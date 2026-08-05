"""
train_win_model.py
Entrena y afina XGBoost y LightGBM para probabilidad de victoria local
(home_win). Búsqueda de hiperparámetros evaluada en 2024 (val); el set
de test (2025-2026) NO se toca aquí -- eso es el último paso, una sola
vez, cuando ya se eligió el modelo final entre todas las fases.

IMPORTANTE: a diferencia del baseline (que sí necesitaba imputar NaN
para la regresión logística), aquí NO se imputa nada -- XGBoost y
LightGBM manejan valores faltantes nativamente y aprenden de la
ausencia del dato como señal (ej. abridor sin historial != promedio).

Uso:
    python src/train_win_model.py
    python src/train_win_model.py --n-iter 40
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

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
XGB_FIXED = {"objective": "binary:logistic", "eval_metric": "logloss",
             "random_state": 42, "verbosity": 0, "n_jobs": -1}

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


def score_logloss(model, X_val, y_val) -> float:
    proba = np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
    return log_loss(y_val, proba)


def calibration_check(model, X_val, y_val, n_bins: int = 10) -> None:
    """¿Entre los partidos donde el modelo dijo '~60% de probabilidad',
    el local de verdad ganó ~60% de las veces? Para un caso de uso de
    probabilidades (no solo accuracy), esto importa tanto o más que el
    log-loss -- un modelo bien calibrado es el que se puede usar para
    tomar decisiones, no solo para "adivinar" el ganador."""
    import pandas as pd

    proba = np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
    bins = pd.qcut(proba, q=n_bins, duplicates="drop")
    tabla = pd.DataFrame({"proba": proba, "y": np.asarray(y_val), "bin": bins})
    resumen = tabla.groupby("bin", observed=True).agg(
        n=("y", "size"), proba_promedio=("proba", "mean"), tasa_real=("y", "mean")
    )
    print("\n--- Calibración (10 grupos por probabilidad predicha) ---")
    print(f"{'rango de proba':<22}{'n':>6}{'proba predicha':>18}{'tasa real de W':>18}")
    for _, row in resumen.iterrows():
        print(f"{'':<22}{int(row['n']):>6}{row['proba_promedio']:>18.3f}{row['tasa_real']:>18.3f}")
    print("(si 'proba predicha' y 'tasa real de W' se parecen en cada fila, el modelo esta bien calibrado)")


def report(model, X, y, label: str) -> dict:
    proba = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    pred = (proba >= 0.5).astype(int)
    result = {
        "accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, proba),
        "brier": brier_score_loss(y, proba),
    }
    print(f"\n--- {label} ---")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Log-loss: {result['log_loss']:.4f}")
    print(f"  Brier:    {result['brier']:.4f}")
    return result


def run(dataset_path: str = "data/training_dataset.parquet", n_iter: int = 25,
        model_out: str = "data/model_win.joblib") -> None:
    df = md.load_dataset(dataset_path)
    train, val, _test = md.temporal_split(df)  # test se deja intacto
    X_train, y_train = md.prepare_xy(train, md.TARGET_WIN)
    X_val, y_val = md.prepare_xy(val, md.TARGET_WIN)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | features: {X_train.shape[1]}")

    print("\n=== Búsqueda de hiperparámetros: XGBoost ===")
    best_xgb, _ = ms.random_search(
        XGBClassifier, XGB_PARAM_SPACE, XGB_FIXED,
        X_train, y_train, X_val, y_val,
        score_fn=score_logloss, higher_is_better=False,
        n_iter=n_iter, label="xgb",
    )

    print("\n=== Búsqueda de hiperparámetros: LightGBM ===")
    best_lgbm, _ = ms.random_search(
        LGBMClassifier, LGBM_PARAM_SPACE, LGBM_FIXED,
        X_train, y_train, X_val, y_val,
        score_fn=score_logloss, higher_is_better=False,
        n_iter=n_iter, label="lgbm",
    )

    print(f"\nMejor XGBoost:  log-loss={best_xgb['score']:.4f}  params={best_xgb['params']}")
    print(f"Mejor LightGBM: log-loss={best_lgbm['score']:.4f}  params={best_lgbm['params']}")

    report(best_xgb["model"], X_val, y_val, "XGBoost (val, mejor combinación)")
    report(best_lgbm["model"], X_val, y_val, "LightGBM (val, mejor combinación)")

    winner_name = "xgboost" if best_xgb["score"] <= best_lgbm["score"] else "lightgbm"
    winner = best_xgb if winner_name == "xgboost" else best_lgbm
    print(f"\n>>> Gana: {winner_name} (log-loss {winner['score']:.4f} en validación) <<<")

    calibration_check(winner["model"], X_val, y_val)

    joblib.dump({
        "model": winner["model"],
        "model_type": winner_name,
        "params": winner["params"],
        "feature_names": list(X_train.columns),
    }, model_out)
    print(f"\nModelo guardado en {model_out}")

    # Importancia de features del ganador (top 15) -- util para sanity-check
    importances = winner["model"].feature_importances_
    order = np.argsort(importances)[::-1][:15]
    print("\nTop 15 features mas importantes:")
    for idx in order:
        print(f"  {X_train.columns[idx]:<35} {importances[idx]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena XGBoost/LightGBM para home_win")
    parser.add_argument("--dataset-path", default="data/training_dataset.parquet")
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--model-out", default="data/model_win.joblib")
    args = parser.parse_args()
    run(args.dataset_path, args.n_iter, args.model_out)
