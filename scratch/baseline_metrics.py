"""
baseline_metrics.py
Carga los modelos actuales y evalúa contra el set de validación (2024)
para tener una línea base ANTES de cualquier cambio.
NO toca el test set.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

import model_data as md


def main():
    df = md.load_dataset("data/training_dataset.parquet")
    _train, val, _test = md.temporal_split(df)
    print(f"Val (2024): {len(val)} partidos")
    print(f"Features actuales: {len(md.feature_columns(df))}")
    print(f"Feature list: {md.feature_columns(df)}")

    # --- Win model ---
    win_saved = joblib.load("data/model_win.joblib")
    win_model = win_saved["model"]
    win_feats = win_saved["feature_names"]
    X_val_w = val[win_feats]
    y_val_w = val[md.TARGET_WIN]

    proba = np.clip(win_model.predict_proba(X_val_w)[:, 1], 1e-6, 1 - 1e-6)
    pred = (proba >= 0.5).astype(int)

    print(f"\n=== BASELINE Win Model ({win_saved['model_type']}) en Val (2024) ===")
    print(f"  Accuracy:  {accuracy_score(y_val_w, pred):.4f}")
    print(f"  Log-loss:  {log_loss(y_val_w, proba):.4f}")
    print(f"  Brier:     {brier_score_loss(y_val_w, proba):.4f}")

    # --- Runs model ---
    runs_saved = joblib.load("data/model_runs.joblib")
    runs_model = runs_saved["model"]
    runs_feats = runs_saved["feature_names"]
    X_val_r = val[runs_feats]
    y_val_r = val[md.TARGET_RUNS]

    runs_pred = runs_model.predict(X_val_r)

    print(f"\n=== BASELINE Runs Model ({runs_saved['model_type']}) en Val (2024) ===")
    print(f"  MAE:   {mean_absolute_error(y_val_r, runs_pred):.3f} carreras")
    print(f"  RMSE:  {mean_squared_error(y_val_r, runs_pred) ** 0.5:.3f} carreras")

    # O/U accuracy at 8.5 line
    ou_pred = (runs_pred >= 8.5).astype(int)
    ou_real = (y_val_r >= 9).astype(int)  # real > 8.5 rounds to >= 9
    ou_acc = accuracy_score(ou_real, ou_pred)
    print(f"  O/U accuracy (line 8.5): {ou_acc:.4f}")


if __name__ == "__main__":
    main()
