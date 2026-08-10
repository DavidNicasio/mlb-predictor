"""
train_f5_model.py
Entrena un modelo dedicado de F5 (Primeras 5 Entradas) usando el historial
de carreras de las entradas 1-5 cargadas en game_linescore.

Entrenamiento: 2015-2023
Validación: 2024
"""

from __future__ import annotations

import argparse
import random
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import db
import model_data as md

PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_lambda": [0.1, 1.0, 2.0, 5.0],
}


def load_f5_dataset(db_path: str = "data/mlb.db", dataset_path: str = "data/training_dataset.parquet") -> pd.DataFrame:
    df = md.load_dataset(dataset_path)

    conn = db.get_connection(db_path)
    f5_df = pd.read_sql_query("""
        SELECT game_pk,
               SUM(home_runs) AS home_score_f5,
               SUM(away_runs) AS away_score_f5,
               (SUM(home_runs) + SUM(away_runs)) AS total_runs_f5,
               CASE WHEN SUM(home_runs) > SUM(away_runs) THEN 1 ELSE 0 END AS home_win_f5
        FROM game_linescore
        WHERE inning <= 5
        GROUP BY game_pk
        HAVING COUNT(DISTINCT inning) = 5
    """, conn)
    conn.close()

    f5_cols_in_df = [c for c in df.columns if c.endswith("_f5")]
    if f5_cols_in_df:
        df = df.drop(columns=f5_cols_in_df)

    f5_df["game_pk"] = f5_df["game_pk"].astype(int)
    merged = df.merge(f5_df, on="game_pk", how="inner")
    print(f"Partidos combinados con F5 de linescore: {len(merged)}")
    return merged


def random_search_xgb(X_tr, y_tr, X_va, y_va, n_iter: int = 20, seed: int = 42) -> tuple[dict, float, XGBRegressor]:
    rng = random.Random(seed)
    best_score = float("inf")
    best_params = {}
    best_model = None

    print("\n=== Búsqueda de hiperparámetros F5: XGBoost ===")
    for i in range(1, n_iter + 1):
        params = {
            "n_estimators": rng.choice(PARAM_DIST["n_estimators"]),
            "learning_rate": rng.choice(PARAM_DIST["learning_rate"]),
            "max_depth": rng.choice(PARAM_DIST["max_depth"]),
            "subsample": rng.choice(PARAM_DIST["subsample"]),
            "colsample_bytree": rng.choice(PARAM_DIST["colsample_bytree"]),
            "min_child_weight": rng.choice(PARAM_DIST["min_child_weight"]),
            "reg_lambda": rng.choice(PARAM_DIST["reg_lambda"]),
            "random_state": 42,
            "n_jobs": -1,
        }
        m = XGBRegressor(**params)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va)
        rmse = mean_squared_error(y_va, pred) ** 0.5

        if rmse < best_score:
            best_score = rmse
            best_params = params
            best_model = m
            print(f"  [xgb {i}/{n_iter}] score={rmse:.4f} <- mejor hasta ahora")
        else:
            print(f"  [xgb {i}/{n_iter}] score={rmse:.4f}")

    return best_params, best_score, best_model


def run_training(db_path: str = "data/mlb.db", dataset_path: str = "data/training_dataset.parquet",
                 n_iter: int = 20, model_out: str = "data/model_f5_runs.joblib") -> None:
    merged = load_f5_dataset(db_path, dataset_path)
    train, val, _test = md.temporal_split(merged)
    features = md.feature_columns(merged)

    X_train, y_train = train[features], train["total_runs_f5"]
    X_val, y_val = val[features], val["total_runs_f5"]

    print(f"Dataset F5: Train: {len(train)} | Val: {len(val)} | features: {len(features)}")

    best_params, best_rmse, best_model = random_search_xgb(X_train, y_train, X_val, y_val, n_iter=n_iter)
    val_pred = best_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)

    print(f"\n--- Modelo F5 Dedicado (val) ---")
    print(f"  MAE:  {val_mae:.3f} carreras F5")
    print(f"  RMSE: {best_rmse:.3f} carreras F5")

    joblib.dump({
        "model": best_model,
        "feature_names": features,
        "model_type": "xgboost",
        "best_params": best_params,
        "val_rmse": best_rmse,
        "val_mae": val_mae,
    }, model_out)

    print(f"\nModelo F5 guardado en {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena modelo F5 dedicado")
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--dataset-path", default="data/training_dataset.parquet")
    parser.add_argument("--n-iter", type=int, default=20)
    args = parser.parse_args()

    run_training(args.db_path, args.dataset_path, args.n_iter)
