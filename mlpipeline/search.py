# hypersearch.py
import os
import json
import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.base import clone
from sklearn.model_selection import ParameterSampler

from config import Config
from solver import Solver

cfg = Config()
metric_fn = Solver(cfg).metric

N_ITER = 200
SAVE_DIR = "./mlpipeline/artifacts/hypersearch"

MODEL_NAME = "LightGBM"

SEARCH_SPACE = [
    {
        "boosting_type": ["gbdt", "dart"],
        "num_leaves": [15, 31, 63],
        "max_depth": [-1, 5, 10],
        "learning_rate": [0.01, 0.03, 0.1],
        "n_estimators": [100, 300, 800],
        "objective": ["binary"],
        "class_weight": [None, "balanced"],
        "min_split_gain": [0.0, 1e-3],
        "min_child_weight": [1e-3, 1e-1, 1.0],
        "min_child_samples": [10, 20, 50],
        "subsample": [0.8, 1.0],
        "subsample_freq": [0, 1],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0.0, 1e-2, 1.0],
        "reg_lambda": [0.0, 1e-2, 1.0],
        "random_state": [42],
        "n_jobs": [-1],
        "verbosity": [-1],
    }
]


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    df = pd.read_parquet(cfg.path_to_train)
    solver = Solver(cfg)
    base_model = cfg.models[MODEL_NAME]
    old_model = cfg.models[MODEL_NAME]
    rows, best_row = [], None

    for idx, params in enumerate(ParameterSampler(SEARCH_SPACE, n_iter=N_ITER, random_state=cfg.seed), start=1):
        model = clone(base_model)
        model.set_params(**params)
        cfg.models[MODEL_NAME] = model

        try:
            art = solver.fit_one_model(df, model, MODEL_NAME)
            ndcg = art["metric"]
            row = {"model": MODEL_NAME, "params": params, "ndcg": ndcg}
            rows.append(row)
            if best_row is None or ndcg > best_row["ndcg"]:
                best_row = deepcopy(row)
            print(f"{idx:>3}/{N_ITER}: ndcg = {ndcg:.5f}, {params}")
        except Exception as e:
            row = {"model": MODEL_NAME, "params": params, "ndcg": None, "error": str(e)}
            rows.append(row)
            print(f"params={params} | error={e}")

    cfg.models[MODEL_NAME] = old_model

    with open(f"{SAVE_DIR}/best_{MODEL_NAME}.json", "w", encoding="utf-8") as f:
        json.dump(best_row, f, ensure_ascii=False, indent=2, default=str)

    print("\nBEST:")
    print(best_row)
