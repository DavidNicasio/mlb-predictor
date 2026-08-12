"""
evaluate_test_set.py
El ÚLTIMO paso de la Fase 4: evalúa los dos modelos ya elegidos y
afinados (con datos de 2015-2023 para entrenar, 2024 para decidir)
contra el set de TEST (2025-2026), que no se había tocado hasta ahora.

Si las métricas de test se parecen a las de validación, el modelo
generaliza de verdad y no fue solo suerte con 2024. Si test sale mucho
peor que validación, es señal de que hubo algo de overfitting a 2024
en las decisiones de hiperparámetros.
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                              mean_absolute_error, mean_squared_error)

import model_data as md


def evaluate_win_model(model_path: str, test: pd.DataFrame) -> dict:
    saved = joblib.load(model_path)
    model, feats = saved["model"], saved["feature_names"]
    X_test, y_test = test[feats], test[md.TARGET_WIN]

    proba = np.clip(model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    print(f"\n=== Modelo de VICTORIA ({saved['model_type']}) en TEST (2025-2026) ===")
    print(f"  n = {len(test)}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Log-loss: {ll:.4f}")
    print(f"  Brier:    {brier:.4f}")

    bins = pd.qcut(proba, q=10, duplicates="drop")
    tabla = pd.DataFrame({"proba": proba, "y": np.asarray(y_test), "bin": bins})
    resumen = tabla.groupby("bin", observed=True).agg(
        n=("y", "size"), proba_promedio=("proba", "mean"), tasa_real=("y", "mean"))
    print("\n  Calibración en test:")
    print(f"  {'n':>6}{'proba predicha':>18}{'tasa real de W':>18}")
    for _, row in resumen.iterrows():
        print(f"  {int(row['n']):>6}{row['proba_promedio']:>18.3f}{row['tasa_real']:>18.3f}")

    return {"acc": acc, "logloss": ll, "brier": brier}


def evaluate_runs_model(model_path: str, test: pd.DataFrame) -> dict:
    saved = joblib.load(model_path)
    model, feats = saved["model"], saved["feature_names"]
    X_test, y_test = test[feats], test[md.TARGET_RUNS]

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5

    print(f"\n=== Modelo de OVER/UNDER ({saved['model_type']}) en TEST (2025-2026) ===")
    print(f"  n = {len(test)}")
    print(f"  MAE:  {mae:.3f} carreras")
    print(f"  RMSE: {rmse:.3f} carreras")

    bins = pd.qcut(pred, q=10, duplicates="drop")
    tabla = pd.DataFrame({"pred": pred, "real": np.asarray(y_test), "bin": bins})
    resumen = tabla.groupby("bin", observed=True).agg(
        n=("real", "size"), pred_promedio=("pred", "mean"), real_promedio=("real", "mean"))
    print("\n  Calibración en test:")
    print(f"  {'n':>6}{'prediccion promedio':>22}{'carreras reales promedio':>28}")
    for _, row in resumen.iterrows():
        print(f"  {int(row['n']):>6}{row['pred_promedio']:>22.2f}{row['real_promedio']:>28.2f}")

    return {"mae": mae, "rmse": rmse}


def log_test_run(win_metrics: dict, runs_metrics: dict, reason: str, log_path: str = "TEST_SET_LOG.md") -> None:
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = f"""
---

### Corrida ({now_str})
- **Fecha**: {now_str[:10]}
- **Motivo / Qué cambió**: {reason}
- **Resultados**:
  - **Victoria (Win)**: Accuracy = `{win_metrics['acc']:.4f}` | Log-loss = `{win_metrics['logloss']:.4f}` | Brier = `{win_metrics['brier']:.4f}`
  - **Carreras (Runs)**: MAE = `{runs_metrics['mae']:.3f}` | RMSE = `{runs_metrics['rmse']:.3f}`
"""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"\nRegistrada corrida en {log_path}")


def run(dataset_path: str = "data/training_dataset.parquet",
        win_model_path: str = "data/model_win.joblib",
        runs_model_path: str = "data/model_runs.joblib",
        reason: str = "Evaluación manual de modelos en test") -> None:
    df = md.load_dataset(dataset_path)
    if "home_lineup_woba" not in df.columns:
        import db
        from train_lineup_models import prepare_dataset_with_lineups
        conn = db.get_connection("data/mlb.db")
        db.init_db(conn)
        df = prepare_dataset_with_lineups(conn, dataset_path)
        conn.close()

    _train, _val, test = md.temporal_split(df)
    print(f"Set de TEST (2025-2026): {len(test)} partidos -- evaluación registrada")

    win_metrics = evaluate_win_model(win_model_path, test)
    runs_metrics = evaluate_runs_model(runs_model_path, test)

    log_test_run(win_metrics, runs_metrics, reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación final contra el set de test con registro automático")
    parser.add_argument("--dataset-path", default="data/training_dataset.parquet")
    parser.add_argument("--win-model", default="data/model_win.joblib")
    parser.add_argument("--runs-model", default="data/model_runs.joblib")
    parser.add_argument("--reason", default="Evaluación en test set", help="Explicación de qué cambió desde la corrida anterior")
    args = parser.parse_args()
    run(args.dataset_path, args.win_model, args.runs_model, args.reason)
