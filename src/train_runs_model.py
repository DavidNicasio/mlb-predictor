"""
train_runs_model.py
Entrena y afina XGBoost y LightGBM para carreras totales del partido
(total_runs, para Over/Under). Mismo patrón que train_win_model.py:
búsqueda evaluada en 2024 (val), test (2025-2026) sin tocar todavía.
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import model_data as md
import model_search as ms

XGB_PARAM_SPACE = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 300, 500],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_lambda": [1, 1.5, 2, 3],
}
XGB_FIXED = {"objective": "reg:squarederror", "random_state": 42, "verbosity": 0, "n_jobs": -1}

LGBM_PARAM_SPACE = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 300, 500],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_samples": [10, 20, 30, 50],
    "reg_lambda": [0, 1, 2],
}
LGBM_FIXED = {"random_state": 42, "verbose": -1, "n_jobs": -1}


def score_rmse(model, X_val, y_val) -> float:
    pred = model.predict(X_val)
    return mean_squared_error(y_val, pred) ** 0.5


def report(model, X, y, label: str) -> dict:
    pred = model.predict(X)
    result = {
        "mae": mean_absolute_error(y, pred),
        "rmse": mean_squared_error(y, pred) ** 0.5,
    }
    print(f"\n--- {label} ---")
    print(f"  MAE:  {result['mae']:.3f} carreras")
    print(f"  RMSE: {result['rmse']:.3f} carreras")
    return result


def calibration_by_bucket(model, X_val, y_val, n_bins: int = 10) -> None:
    """Para Over/Under interesa sobre todo si el modelo distingue bien
    partidos de pocas vs muchas carreras -- no solo el error promedio."""
    import pandas as pd

    pred = model.predict(X_val)
    bins = pd.qcut(pred, q=n_bins, duplicates="drop")
    tabla = pd.DataFrame({"pred": pred, "real": np.asarray(y_val), "bin": bins})
    resumen = tabla.groupby("bin", observed=True).agg(
        n=("real", "size"), pred_promedio=("pred", "mean"), real_promedio=("real", "mean")
    )
    print("\n--- Calibración por rango de predicción (10 grupos) ---")
    print(f"{'n':>6}{'prediccion promedio':>22}{'carreras reales promedio':>28}")
    for _, row in resumen.iterrows():
        print(f"{int(row['n']):>6}{row['pred_promedio']:>22.2f}{row['real_promedio']:>28.2f}")


def run(dataset_path: str = "data/training_dataset.parquet", n_iter: int = 25,
        model_out: str = "data/model_runs.joblib") -> None:
    df = md.load_dataset(dataset_path)
    train, val, _test = md.temporal_split(df)
    X_train, y_train = md.prepare_xy(train, md.TARGET_RUNS)
    X_val, y_val = md.prepare_xy(val, md.TARGET_RUNS)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | features: {X_train.shape[1]}")

    print("\n=== Búsqueda de hiperparámetros: XGBoost ===")
    best_xgb, _ = ms.random_search(
        XGBRegressor, XGB_PARAM_SPACE, XGB_FIXED,
        X_train, y_train, X_val, y_val,
        score_fn=score_rmse, higher_is_better=False,
        n_iter=n_iter, label="xgb",
    )

    print("\n=== Búsqueda de hiperparámetros: LightGBM ===")
    best_lgbm, _ = ms.random_search(
        LGBMRegressor, LGBM_PARAM_SPACE, LGBM_FIXED,
        X_train, y_train, X_val, y_val,
        score_fn=score_rmse, higher_is_better=False,
        n_iter=n_iter, label="lgbm",
    )

    print(f"\nMejor XGBoost:  RMSE={best_xgb['score']:.3f}  params={best_xgb['params']}")
    print(f"Mejor LightGBM: RMSE={best_lgbm['score']:.3f}  params={best_lgbm['params']}")

    report(best_xgb["model"], X_val, y_val, "XGBoost (val, mejor combinación)")
    report(best_lgbm["model"], X_val, y_val, "LightGBM (val, mejor combinación)")

    winner_name = "xgboost" if best_xgb["score"] <= best_lgbm["score"] else "lightgbm"
    winner = best_xgb if winner_name == "xgboost" else best_lgbm
    print(f"\n>>> Gana: {winner_name} (RMSE {winner['score']:.3f} en validación) <<<")

    calibration_by_bucket(winner["model"], X_val, y_val)

    joblib.dump({
        "model": winner["model"],
        "model_type": winner_name,
        "params": winner["params"],
        "feature_names": list(X_train.columns),
    }, model_out)
    print(f"\nModelo guardado en {model_out}")

    importances = winner["model"].feature_importances_
    order = np.argsort(importances)[::-1][:15]
    print("\nTop 15 features mas importantes:")
    for idx in order:
        print(f"  {X_train.columns[idx]:<35} {importances[idx]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena XGBoost/LightGBM para total_runs")
    parser.add_argument("--dataset-path", default="data/training_dataset.parquet")
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--model-out", default="data/model_runs.joblib")
    args = parser.parse_args()
    run(args.dataset_path, args.n_iter, args.model_out)
