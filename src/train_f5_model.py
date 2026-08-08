"""
train_f5_model.py
Entrena y afina modelos XGBoost / LightGBM dedicados a las Primeras 5 Entradas:
  - F5 Total Runs (regresión)
  - F5 Home Win Probability (clasificación)

Utiliza la misma división temporal estricta (Train 2015-2023, Val 2024, Test 2025-2026).
Compara el desempeño contra el aproximador heurístico previo (full_game * 0.55 * fip_factor).

Uso:
    python src/train_f5_model.py
    python src/train_f5_model.py --n-iter 25
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

import model_data as md
import model_search as ms

XGB_PARAM_SPACE = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05],
    "n_estimators": [100, 200, 300],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [3, 5, 10],
    "reg_lambda": [1, 2, 3],
}
XGB_RUNS_FIXED = {"objective": "reg:squarederror", "random_state": 42, "verbosity": 0, "n_jobs": -1}
XGB_WIN_FIXED = {"objective": "binary:logistic", "eval_metric": "logloss", "random_state": 42, "verbosity": 0, "n_jobs": -1}

LGBM_PARAM_SPACE = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.03, 0.05],
    "n_estimators": [100, 200, 300],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_samples": [20, 30, 50],
    "reg_lambda": [0, 1, 2],
}
LGBM_FIXED = {"random_state": 42, "verbose": -1, "n_jobs": -1}


def score_rmse(model, X_val, y_val) -> float:
    pred = model.predict(X_val)
    return mean_squared_error(y_val, pred) ** 0.5


def score_logloss(model, X_val, y_val) -> float:
    proba = np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
    return log_loss(y_val, proba)


def run(dataset_path: str = "data/training_dataset.parquet", n_iter: int = 25,
        model_runs_out: str = "data/model_f5_runs.joblib",
        model_win_out: str = "data/model_f5_win.joblib") -> None:
    df = md.load_dataset(dataset_path)

    # Filtrar solo juegos con datos de F5 válidos (no nulos)
    df_f5_runs = df[df["total_runs_f5"].notna()].copy()
    df_f5_win = df[df["home_win_f5"].notna()].copy()

    train_r, val_r, _test_r = md.temporal_split(df_f5_runs)
    train_w, val_w, _test_w = md.temporal_split(df_f5_win)

    X_train_r, y_train_r = md.prepare_xy(train_r, "total_runs_f5")
    X_val_r, y_val_r = md.prepare_xy(val_r, "total_runs_f5")

    X_train_w, y_train_w = md.prepare_xy(train_w, "home_win_f5")
    X_val_w, y_val_w = md.prepare_xy(val_w, "home_win_f5")

    print(f"\n========================================================")
    print(f"  ENTRENAMIENTO MODELO F5 (FIRST 5 INNINGS)")
    print(f"  Train: {len(X_train_r)} | Val: {len(X_val_r)} | Features: {X_train_r.shape[1]}")
    print(f"========================================================\n")

    # --- 1. MODELO F5 CARRERAS (REGRESIÓN) ---
    print("\n--- 1. Búsqueda F5 Total Runs (Regresión) ---")
    best_xgb_r, _ = ms.random_search(
        XGBRegressor, XGB_PARAM_SPACE, XGB_RUNS_FIXED,
        X_train_r, y_train_r, X_val_r, y_val_r,
        score_fn=score_rmse, higher_is_better=False,
        n_iter=n_iter, label="xgb_f5_runs",
    )
    best_lgbm_r, _ = ms.random_search(
        LGBMRegressor, LGBM_PARAM_SPACE, LGBM_FIXED,
        X_train_r, y_train_r, X_val_r, y_val_r,
        score_fn=score_rmse, higher_is_better=False,
        n_iter=n_iter, label="lgbm_f5_runs",
    )

    winner_r_name = "xgboost" if best_xgb_r["score"] <= best_lgbm_r["score"] else "lightgbm"
    winner_r = best_xgb_r if winner_r_name == "xgboost" else best_lgbm_r

    runs_pred = winner_r["model"].predict(X_val_r)
    mae_r = mean_absolute_error(y_val_r, runs_pred)
    rmse_r = mean_squared_error(y_val_r, runs_pred) ** 0.5

    print(f"\n>>> Ganador F5 Runs: {winner_r_name} <<<")
    print(f"  MAE en Val (2024):  {mae_r:.3f} carreras F5")
    print(f"  RMSE en Val (2024): {rmse_r:.3f} carreras F5")

    # Evaluación vs aproximador heurístico previo (full_game * 0.55 * fip_factor)
    import features_f5
    heur_f5_runs = [
        features_f5.calculate_f5_projections(
            r.get("home_fip"), r.get("away_fip"), None, None,
            float(r.get("total_runs", 8.5)), 0.50
        )["f5_total_runs_pred"] for _, r in val_r.iterrows()
    ]
    mae_heur = mean_absolute_error(y_val_r, heur_f5_runs)
    print(f"  MAE Heurística previa: {mae_heur:.3f} carreras F5")
    print(f"  👉 Mejora del modelo dedicado: {mae_heur - mae_r:+.3f} carreras menor error MAE")

    joblib.dump({
        "model": winner_r["model"],
        "model_type": winner_r_name,
        "params": winner_r["params"],
        "feature_names": list(X_train_r.columns),
    }, model_runs_out)
    print(f"Modelo F5 Runs guardado en {model_runs_out}")

    # --- 2. MODELO F5 WIN PROBABILITY (CLASIFICACIÓN) ---
    print("\n--- 2. Búsqueda F5 Home Win (Clasificación) ---")
    best_xgb_w, _ = ms.random_search(
        XGBClassifier, XGB_PARAM_SPACE, XGB_WIN_FIXED,
        X_train_w, y_train_w, X_val_w, y_val_w,
        score_fn=score_logloss, higher_is_better=False,
        n_iter=n_iter, label="xgb_f5_win",
    )
    best_lgbm_w, _ = ms.random_search(
        LGBMClassifier, LGBM_PARAM_SPACE, LGBM_FIXED,
        X_train_w, y_train_w, X_val_w, y_val_w,
        score_fn=score_logloss, higher_is_better=False,
        n_iter=n_iter, label="lgbm_f5_win",
    )

    winner_w_name = "xgboost" if best_xgb_w["score"] <= best_lgbm_w["score"] else "lightgbm"
    winner_w = best_xgb_w if winner_w_name == "xgboost" else best_lgbm_w

    win_proba = np.clip(winner_w["model"].predict_proba(X_val_w)[:, 1], 1e-6, 1 - 1e-6)
    acc_w = accuracy_score(y_val_w, (win_proba >= 0.5).astype(int))
    ll_w = log_loss(y_val_w, win_proba)
    brier_w = brier_score_loss(y_val_w, win_proba)

    print(f"\n>>> Ganador F5 Win: {winner_w_name} <<<")
    print(f"  Accuracy en Val (2024): {acc_w:.4f}")
    print(f"  Log-loss en Val (2024): {ll_w:.4f}")
    print(f"  Brier en Val (2024):    {brier_w:.4f}")

    joblib.dump({
        "model": winner_w["model"],
        "model_type": winner_w_name,
        "params": winner_w["params"],
        "feature_names": list(X_train_w.columns),
    }, model_win_out)
    print(f"Modelo F5 Win guardado en {model_win_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena modelos dedicados XGBoost/LightGBM para F5")
    parser.add_argument("--dataset-path", default="data/training_dataset.parquet")
    parser.add_argument("--n-iter", type=int, default=25)
    args = parser.parse_args()
    run(args.dataset_path, args.n_iter)
