import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.linear_model import Ridge, LinearRegression


class Ensembler:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.enabled_methods = [method for method, is_enabled in getattr(cfg, "ensemble_method", {}).items() if is_enabled]

        competition = {
            'threshold': getattr(cfg, "threshold", 0.5),
        }

        self.artifacts = {
            "enabled_methods": self.enabled_methods,
            "models": {},
            'competition': competition,
        }


    def fit(self, X, y):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = np.asarray(y).astype(int)

        self.artifacts["feature_names"] = X_df.columns.tolist()

        if "linear" in self.enabled_methods:
            if self.cfg.is_classificate:
                linear_model = LogisticRegression(max_iter=1000, random_state=self.cfg.seed)
            else:
                linear_model = LinearRegression()
            linear_model.fit(X_df, y)
            self.artifacts["models"]["linear"] = linear_model

        if "ridge" in self.enabled_methods:
            if self.cfg.is_classificate:
                ridge_model = RidgeClassifier(
                    alpha=getattr(self.cfg, "ensemble_alpha", 1.0),
                    random_state=getattr(self.cfg, "seed", 42),
                )
            else:
                ridge_model = Ridge(
                    alpha=getattr(self.cfg, "ensemble_alpha", 1.0),
                    random_state=getattr(self.cfg, "seed", 42),
                )

            ridge_model.fit(X_df, y)
            self.artifacts["models"]["ridge"] = ridge_model

        return self

    def predict(self, X):
        if not self.artifacts["enabled_methods"]:
            raise ValueError("No ensemble methods are enabled in cfg.ensemble_method.")

        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.cfg.is_classificate: 
            threshold = self.artifacts['competition']['threshold']
        feature_names = self.artifacts.get("feature_names")
        if feature_names is not None:
            X_df = X_df.reindex(columns=feature_names)

        preds = {}

        if "mean" in self.enabled_methods:
            mean_pred = X_df.mean(axis=1)
            preds["mean"] = mean_pred.astype(float)

        if "vote" in self.enabled_methods:
            if self.cfg.is_classificate: 
                base_labels = (X_df >= threshold).astype(int)
                vote_pred = (base_labels.mean(axis=1) >= 0.5).astype(int)
                preds["vote"] = vote_pred
            else:
                pass

        if "linear" in self.enabled_methods:
            linear_model = self.artifacts["models"].get("linear")
            if linear_model is None:
                raise ValueError("Linear ensemble is enabled but not fitted/loaded.")
            if self.cfg.is_classificate: 
                preds["linear"] = pd.Series(linear_model.predict(X_df), index=X_df.index).astype(int)
            else:
                preds["linear"] = pd.Series(linear_model.predict(X_df), index=X_df.index).astype(float)

        if "ridge" in self.enabled_methods:
            ridge_model = self.artifacts["models"].get("ridge")
            if ridge_model is None:
                raise ValueError("Ridge ensemble is enabled but not fitted/loaded.")
            if self.cfg.is_classificate: 
                preds["ridge"] = pd.Series(ridge_model.predict(X_df), index=X_df.index).astype(int)
            else:
                preds["ridge"] = pd.Series(ridge_model.predict(X_df), index=X_df.index).astype(float)

        return pd.DataFrame(preds, index=X_df.index)

    def save(self):
        path = self.cfg.path_to_save_ensemble
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.artifacts, f)

        return self

    def load(self):
        path = self.cfg.path_to_load_ensemble
        with open(path, "rb") as f:
            self.artifacts = pickle.load(f)
        return self