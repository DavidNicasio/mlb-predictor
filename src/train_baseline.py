"""
train_baseline.py
Baselines antes de XGBoost/LightGBM. Si el modelo grande no le gana a
esto por un margen claro, el problema está en el pipeline, no en el
modelo -- por eso se corre esto primero.

Clasificación (home_win):
  - "el local siempre gana" (tasa histórica de train, constante)
  - regresión logística sobre las features

Regresión (total_runs):
  - promedio histórico de carreras (constante, de train)
  - regresión lineal sobre las features

Evaluado en VALIDACIÓN (2024) -- el set de TEST (2025-2026) se guarda
para el final, cuando ya se haya elegido y afinado el modelo real.
"""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                              mean_absolute_error, mean_squared_error)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import model_data as md


def eval_classification(y_true, y_pred_proba, label: str) -> dict:
    y_pred_proba = np.clip(y_pred_proba, 1e-6, 1 - 1e-6)  # log_loss no acepta 0/1 exactos
    y_pred = (y_pred_proba >= 0.5).astype(int)
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_pred_proba),
        "brier": brier_score_loss(y_true, y_pred_proba),
    }
    print(f"\n--- {label} ---")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Log-loss: {result['log_loss']:.4f}  (mas bajo = mejor; azar puro ~0.693)")
    print(f"  Brier:    {result['brier']:.4f}  (mas bajo = mejor; azar puro = 0.25)")
    return result


def eval_regression(y_true, y_pred, label: str) -> dict:
    result = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }
    print(f"\n--- {label} ---")
    print(f"  MAE:  {result['mae']:.3f} carreras")
    print(f"  RMSE: {result['rmse']:.3f} carreras")
    return result


def run(dataset_path: str = "data/training_dataset.parquet") -> None:
    df = md.load_dataset(dataset_path)
    train, val, test = md.temporal_split(df)
    print(f"Train (2015-2023): {len(train)} | Val (2024): {len(val)} | "
          f"Test (2025-2026, sin tocar todavia): {len(test)}")

    # =================== CLASIFICACION: home_win ===================
    X_train, y_train = md.prepare_xy(train, md.TARGET_WIN)
    X_val, y_val = md.prepare_xy(val, md.TARGET_WIN)

    home_rate = y_train.mean()
    print(f"\n% de victorias locales en train: {home_rate:.4f}")
    pred_const = np.full(len(y_val), home_rate)
    eval_classification(y_val, pred_const, "Baseline 1: 'el local siempre gana' (constante)")

    logit = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    logit.fit(X_train, y_train)
    pred_logit = logit.predict_proba(X_val)[:, 1]
    eval_classification(y_val, pred_logit, "Baseline 2: regresion logistica")

    # =================== REGRESION: total_runs ===================
    X_train_r, y_train_r = md.prepare_xy(train, md.TARGET_RUNS)
    X_val_r, y_val_r = md.prepare_xy(val, md.TARGET_RUNS)

    mean_runs = y_train_r.mean()
    print(f"\nPromedio de carreras totales en train: {mean_runs:.3f}")
    pred_const_r = np.full(len(y_val_r), mean_runs)
    eval_regression(y_val_r, pred_const_r, "Baseline 1: promedio historico (constante)")

    linreg = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression(),
    )
    linreg.fit(X_train_r, y_train_r)
    pred_linreg = linreg.predict(X_val_r)
    eval_regression(y_val_r, pred_linreg, "Baseline 2: regresion lineal")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Corre los baselines antes de XGBoost/LightGBM")
    parser.add_argument("--dataset-path", default="data/training_dataset.parquet")
    args = parser.parse_args()
    run(args.dataset_path)
