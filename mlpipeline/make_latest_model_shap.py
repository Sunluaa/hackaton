from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool

from config import Config
from dataprocess import DataProcess


SAMPLE_SIZE = 40_000
TOP_FEATURES = 15
OUTPUT_PATH = Path("reports/latest_model_shap_summary.png")


def find_latest_model_path(models_dir: str) -> Path:
    model_paths = sorted(
        Path(models_dir).glob("*.pkl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not model_paths:
        raise FileNotFoundError(f"No model checkpoints found in {models_dir!r}.")
    return model_paths[0]


def load_checkpoint(model_path: Path) -> dict:
    with model_path.open("rb") as fh:
        return pickle.load(fh)


def load_processed_sample(cfg: Config, model_name: str, data_checkpoint: dict) -> tuple[pd.DataFrame, DataProcess]:
    df = pd.read_parquet(cfg.path_to_train)

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=cfg.seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    processor = DataProcess(cfg, model_name=model_name)
    processor.data_checkpoint = data_checkpoint

    processed = processor.transform(df)
    X, _ = processor.split_Xy(processed)
    return X, processor


def compute_shap_values(model_name: str, model, X: pd.DataFrame, processor: DataProcess) -> np.ndarray:
    is_catboost = "catboost" in model_name.lower() or "catboost" in type(model).__name__.lower()

    if is_catboost:
        cat_cols = [
            column
            for column in processor.data_checkpoint.get("cat_cols", [])
            if column in X.columns
        ]
        pool = Pool(X, cat_features=cat_cols or None)
        shap_with_bias = model.get_feature_importance(pool, type="ShapValues")
    else:
        try:
            shap_with_bias = model.predict(X, pred_contrib=True)
        except TypeError:
            shap_with_bias = model.booster_.predict(X, pred_contrib=True)

    shap_with_bias = np.asarray(shap_with_bias, dtype=float)
    if shap_with_bias.ndim != 2 or shap_with_bias.shape[1] != X.shape[1] + 1:
        raise ValueError(
            "Unexpected SHAP shape: "
            f"got {shap_with_bias.shape}, expected ({len(X)}, {X.shape[1] + 1})."
        )

    return shap_with_bias[:, :-1]


def numeric_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    numeric = pd.DataFrame(index=X.index)

    for column in X.columns:
        series = X[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric[column] = series.astype(float)
        else:
            numeric[column] = pd.factorize(series.astype(str), sort=True)[0].astype(float)

    return numeric


def save_summary_plot(model_name: str, X: pd.DataFrame, shap_values: np.ndarray, output_path: Path) -> None:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-TOP_FEATURES:][::-1]
    top_features = [X.columns[idx] for idx in top_idx]
    top_importance = mean_abs_shap[top_idx]

    color_frame = numeric_feature_frame(X)
    rng = np.random.default_rng(42)

    fig, (ax_bar, ax_swarm) = plt.subplots(
        1,
        2,
        figsize=(18, 8),
        gridspec_kw={"width_ratios": [1.0, 1.8]},
    )

    ax_bar.barh(top_features[::-1], top_importance[::-1], color="#1f77b4")
    ax_bar.set_title("Mean |SHAP|")
    ax_bar.set_xlabel("Average absolute contribution")

    cmap = plt.cm.coolwarm
    last_scatter = None

    for y_pos, feature_idx in enumerate(top_idx[::-1]):
        shap_column = shap_values[:, feature_idx]
        feature_name = X.columns[feature_idx]
        feature_values = color_frame.iloc[:, feature_idx].to_numpy(dtype=float)

        lower, upper = np.nanpercentile(feature_values, [5, 95])
        if np.isclose(lower, upper):
            normalized = np.full_like(feature_values, 0.5, dtype=float)
        else:
            normalized = np.clip((feature_values - lower) / (upper - lower), 0.0, 1.0)

        jitter = rng.normal(loc=0.0, scale=0.12, size=len(shap_column))
        last_scatter = ax_swarm.scatter(
            shap_column,
            np.full(len(shap_column), y_pos) + jitter,
            c=normalized,
            cmap=cmap,
            s=18,
            alpha=0.75,
            edgecolors="none",
        )

    ax_swarm.axvline(0.0, color="#555555", linewidth=1, alpha=0.8)
    ax_swarm.set_yticks(range(len(top_features)))
    ax_swarm.set_yticklabels(top_features[::-1])
    ax_swarm.set_title(f"SHAP summary for latest model: {model_name}")
    ax_swarm.set_xlabel("SHAP value")

    cbar = fig.colorbar(last_scatter, ax=ax_swarm, pad=0.02)
    cbar.set_label("Feature value")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = Config()
    model_path = find_latest_model_path(cfg.path_to_load_models)
    checkpoint = load_checkpoint(model_path)
    model_name = checkpoint.get("model_name", model_path.stem)

    if not checkpoint.get("models"):
        raise ValueError(f"Checkpoint {model_path} does not contain fitted folds.")
    if not checkpoint.get("data_fold_checkpoints"):
        raise ValueError(f"Checkpoint {model_path} does not contain preprocessing checkpoints.")

    model = checkpoint["models"][0]
    data_checkpoint = checkpoint["data_fold_checkpoints"][0]

    X, processor = load_processed_sample(cfg, model_name, data_checkpoint)
    shap_values = compute_shap_values(model_name, model, X, processor)
    save_summary_plot(model_name, X, shap_values, OUTPUT_PATH)

    print(f"Saved SHAP summary to: {output_path_to_string(OUTPUT_PATH)}")
    print(f"Model used: {model_name}")
    print(f"Checkpoint used: {model_path}")
    print(f"Rows used: {len(X)}")


def output_path_to_string(path: Path) -> str:
    return str(path.as_posix())


if __name__ == "__main__":
    main()
