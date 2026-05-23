import os
import time
import random
import pickle
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import Pool

from sklearn.base import clone

from dataprocess import DataProcess


class Solver:
    def __init__(self, config) -> None:
        self.cfg = config
        self._set_seed()

    def _set_seed(self):
        self.seed = getattr(self.cfg, "seed", 42)

        random.seed(self.seed)
        np.random.seed(self.seed)

    def ndcg_at_k(self, rel, k=None):
        top_k = rel[:k]
        dcg = sum(1.0 / math.log2(i + 2) for i, g in enumerate(top_k) if g == 1)

        ideal_rel = sorted(rel, reverse=True)[:k]
        idcg = sum(1.0 / math.log2(i + 2) for i, g in enumerate(ideal_rel) if g == 1)

        return dcg / idcg if idcg > 0 else np.nan

    def metric(self, y, y_pred, request_ids):
        metric_k = getattr(self.cfg, "metric_at_k", 5)
        target_col = getattr(self.cfg, "column_target", "is_deal")
        request_col = getattr(self.cfg, "group_column", "request_id")

        metric_df = pd.DataFrame(
            {
                request_col: np.asarray(request_ids),
                target_col: np.asarray(y).astype(int),
                "score": np.asarray(y_pred, dtype=float),
            }
        )

        df_sorted = metric_df.sort_values([request_col, "score"], ascending=[True, False])
        ndcg_per_request = df_sorted.groupby(request_col)[target_col].apply(
            lambda x: self.ndcg_at_k(x.tolist(), k=metric_k)
        )
        return float(np.nanmean(ndcg_per_request))

    def _is_ranking_model(self, model_name):
        return model_name in {"LambdaMART", "CatBoostYetiRank"}

    @staticmethod
    def _is_catboost_ranking_model(model_name):
        return model_name == "CatBoostYetiRank"

    @staticmethod
    def _build_group_sizes(group_values):
        _, counts = np.unique(np.asarray(group_values), return_counts=True)
        return counts.tolist()

    @staticmethod
    def _sorted_group_order(group_values):
        return np.argsort(np.asarray(group_values), kind="stable")

    @staticmethod
    def _restore_order(values, sorted_order):
        restored = np.empty_like(values, dtype=float)
        restored[sorted_order] = np.asarray(values, dtype=float)
        return restored

    def fit_one_model(self, df, model, model_name):
        cv_dataset = DataProcess(self.cfg, model_name=model_name)
        splits = cv_dataset.cv_splits(df)

        oof_preds = np.zeros(len(df), dtype=float)
        oof_labels = df[self.cfg.column_target].to_numpy()
        oof_request_ids = df[self.cfg.group_column].to_numpy()

        fold_models = []
        data_fold_checkpoints = []

        progress = tqdm(splits, desc=f"{model_name} folds", total=len(splits), leave=False)

        for fold_idx, (train_idx, valid_idx) in enumerate(progress, start=1):
            train = df.iloc[train_idx].reset_index(drop=True)
            valid = df.iloc[valid_idx].reset_index(drop=True)

            if self._is_ranking_model(model_name):
                train_order = self._sorted_group_order(train[self.cfg.group_column].to_numpy())
                valid_order = self._sorted_group_order(valid[self.cfg.group_column].to_numpy())
                train = train.iloc[train_order].reset_index(drop=True)
                valid = valid.iloc[valid_order].reset_index(drop=True)

            train_dataset = DataProcess(self.cfg, model_name=model_name)
            train_processed = train_dataset.fit_transform(train)
            data_train_checkpoint = train_dataset.data_checkpoint.copy()
            X_train, y_train = train_dataset.split_Xy(train_processed)

            valid_dataset = DataProcess(self.cfg, model_name=model_name)
            valid_dataset.data_checkpoint = data_train_checkpoint
            valid_processed = valid_dataset.transform(valid)
            X_valid, y_valid = valid_dataset.split_Xy(valid_processed)

            fit_kwargs = getattr(self.cfg, "fit_params", {}).get(model_name, {}).copy()

            if fit_kwargs.get("cat_features") == "__auto__":
                fit_kwargs["cat_features"] = train_dataset.data_checkpoint.get(
                    "cat_cols", []
                )

            if self._is_ranking_model(model_name):
                fit_kwargs["group"] = self._build_group_sizes(train[self.cfg.group_column].to_numpy())
                fit_kwargs["eval_set"] = [(X_valid, y_valid)]
                fit_kwargs["eval_group"] = [
                    self._build_group_sizes(valid[self.cfg.group_column].to_numpy())
                ]
                fit_kwargs.setdefault(
                    "eval_at",
                    [getattr(self.cfg, "metric_at_k", 5)],
                )

            model_fold = clone(model)
            if self._is_catboost_ranking_model(model_name):
                cat_features = fit_kwargs.pop("cat_features", None)
                fit_kwargs.pop("group", None)
                fit_kwargs.pop("eval_group", None)
                fit_kwargs.pop("eval_at", None)
                fit_kwargs.pop("eval_set", None)

                train_pool = Pool(
                    data=X_train,
                    label=y_train,
                    cat_features=cat_features,
                    group_id=train[self.cfg.group_column].to_numpy(),
                )
                valid_pool = Pool(
                    data=X_valid,
                    label=y_valid,
                    cat_features=cat_features,
                    group_id=valid[self.cfg.group_column].to_numpy(),
                )
                model_fold.fit(train_pool, eval_set=valid_pool, **fit_kwargs)
            else:
                model_fold.fit(X_train, y_train, **fit_kwargs)

            if hasattr(model_fold, "predict_proba"):
                preds = model_fold.predict_proba(X_valid)[:, 1]
            else:
                preds = model_fold.predict(X_valid)

            if self._is_ranking_model(model_name):
                preds = self._restore_order(preds, valid_order)

            oof_preds[valid_idx] = preds

            fold_models.append(model_fold)
            data_fold_checkpoints.append(data_train_checkpoint)

        metric = self.metric(oof_labels, oof_preds, oof_request_ids)

        artifacts = {
            "model_name": model_name,
            "seed": self.seed,
            "models": fold_models,
            "data_fold_checkpoints": data_fold_checkpoints,
            "metric": metric,
            "oof_preds": oof_preds,
            "oof_labels": oof_labels,
            "oof_request_ids": oof_request_ids,
            "blending_method": "mean",
            # "threshold": self.cfg.threshold,
        }

        return artifacts

    def fit(self, df):
        artifacts = {}

        for model_name, use_model in self.cfg.model.items():
            if not use_model: continue

            model = self.cfg.models[model_name]

            start_time = time.time()
            artifact = self.fit_one_model(df, model, model_name)

            artifact['train_time_sec'] = time.time() - start_time
            print(
                f"{model_name:<25} | "
                f"{getattr(self.cfg, 'metric_name', 'ndcg').upper()}@{getattr(self.cfg, 'metric_at_k', 5)} = "
                f"{artifact['metric']:.6f} | "
                f"time = {artifact['train_time_sec']:>5.2f}"
            )
            
            artifacts[model_name] = artifact
            self.save_model(model_name, artifact)

        return artifacts

    def predict(self, df):
        preds = {}

        for model_name, use_model in self.cfg.model.items():
            if not use_model:continue

            checkpoint = self.load_model(model_name)

            fold_models = checkpoint["models"]
            fold_checkpoints = checkpoint["data_fold_checkpoints"]
            blending_method = getattr(self.cfg, "blending_method", "mean")

            fold_preds = []

            for model_fold, data_fold_checkpoint in zip(fold_models, fold_checkpoints):
                dataprocess = DataProcess(self.cfg, model_name=model_name)
                dataprocess.data_checkpoint = data_fold_checkpoint

                df_pred = dataprocess.transform(df)
                if hasattr(model_fold, "predict_proba"):
                    pred = model_fold.predict_proba(df_pred)[:, 1]
                else:
                    pred = model_fold.predict(df_pred)

                fold_preds.append(np.asarray(pred, dtype=float))

            fold_preds = np.vstack(fold_preds)

            if blending_method == "mean":
                pred = fold_preds.mean(axis=0)
            else:
                pred = fold_preds.mean(axis=0)

            preds[model_name] = pred

        return pd.DataFrame(preds, index=df.index)

    def save_model(self, model_name, artifact):
        os.makedirs(self.cfg.path_to_save_models, exist_ok=True)
        path = os.path.join(self.cfg.path_to_save_models, f"{model_name}.pkl")

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model_name": artifact["model_name"],
                    "models": artifact["models"],
                    "seed": artifact["seed"],
                    "data_fold_checkpoints": artifact["data_fold_checkpoints"],
                    "metric": artifact["metric"],
                    "metric_name": getattr(self.cfg, "metric_name", "ndcg"),
                    "metric_at_k": getattr(self.cfg, "metric_at_k", 5),
                    "blending_method": artifact.get("blending_method", "mean"),
                    "n_folds": artifact.get("n_folds", len(artifact["models"])),
                    # "threshold": artifact["threshold"],
                },
                f,
            )

    def load_model(self, model_name):
        path = os.path.join(self.cfg.path_to_load_models, f'{model_name}.pkl')
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint
