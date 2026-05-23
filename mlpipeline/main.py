import os
import pandas as pd

from config import Config
from solver import Solver
from ensemble import Ensembler
from sklearn.model_selection import GroupShuffleSplit


def _save_submissions(cfg, test_df, ens_preds):
    os.makedirs(cfg.path_to_submission_dir, exist_ok=True)

    if isinstance(ens_preds, pd.Series):
        ens_preds = ens_preds.to_frame(name="score")

    for method in ens_preds.columns:
        submission_path = os.path.join(
            cfg.path_to_submission_dir,
            f"submission_{cfg.name}_{method}.csv"
        )

        submission_df = pd.DataFrame({
            "request_id": test_df["request_id"],
            "variant_no": test_df["variant_no"],
            "score": ens_preds[method].astype(float).round(6),
        })

        submission_df.to_csv(
            submission_path,
            sep=";",
            index=False
        )

        print(f"saved: {submission_path}")


def train(cfg):
    train_df = pd.read_parquet(cfg.path_to_train)
    feature_df = pd.read_parquet(cfg.path_to_test_features)

    train_df = pd.merge(train_df, feature_df, on=['app_id', 'date_part'])

    holdout_X = None
    holdout_y = None
    holdout_request_ids = None

    if cfg.is_holdout:
        val_size = getattr(cfg, "val_size", 0.1)
        rs = getattr(cfg, "seed", 42)
        group_col = getattr(cfg, "group_column", "request_id")
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=rs)
        train_idx, test_idx = next(
            splitter.split(train_df, train_df[cfg.column_target], groups=train_df[group_col])
        )
        test_df = train_df.iloc[test_idx].reset_index(drop=True)
        train_df = train_df.iloc[train_idx].reset_index(drop=True)
        holdout_X = test_df.drop(columns=[cfg.column_target])
        holdout_y = test_df[cfg.column_target].copy()
        holdout_request_ids = test_df[group_col].copy()

    solver = Solver(cfg)
    artifacts = solver.fit(train_df)

    preds = pd.DataFrame(
        {name: art["oof_preds"] for name, art in artifacts.items()},
        index=train_df.index,
    )
    y = next(iter(artifacts.values()))["oof_labels"]

    ensembler = Ensembler(cfg)
    ensembler.fit(preds, y)
    ensembler.save()

    if cfg.is_holdout:
        holdout_base_preds = solver.predict(holdout_X)
        holdout_ens_preds = ensembler.predict(holdout_base_preds)

        # print("Holdout metrics:")
        # for model_name in holdout_base_preds.columns:
        #     score = solver.metric(holdout_y, holdout_base_preds[model_name], holdout_request_ids)
        #     print(f"  base/{model_name:<20} = {score:.6f}")

        for method_name in holdout_ens_preds.columns:
            score = solver.metric(holdout_y, holdout_ens_preds[method_name], holdout_request_ids)
            print(
                f"  ensemble/{method_name:<16} "
                f"{getattr(cfg, 'metric_name', 'ndcg').upper()}@{getattr(cfg, 'metric_at_k', 5)} = "
                f"{score:.6f}"
            )

    print("training finished")


def inference(cfg):
    test_df = pd.read_parquet(cfg.path_to_test)
    feature_df = pd.read_parquet(cfg.path_to_test_features)

    test_df = pd.merge(test_df, feature_df, on=['app_id', 'date_part'])


    solver = Solver(cfg)
    base_preds = solver.predict(test_df)

    ensembler = Ensembler(cfg)
    ensembler.load()
    ens_preds = ensembler.predict(base_preds)

    _save_submissions(cfg, test_df, ens_preds)
    print("inference finished")


def main():
    cfg = Config()

    if cfg.is_train:
        train(cfg)
    else:
        inference(cfg)


if __name__ == "__main__":
    main()
