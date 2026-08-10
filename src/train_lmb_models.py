"""
train_lmb_models.py
Entrenamiento de modelos baseline regularizados (Regresión Logística y Ridge)
diseñados específicamente para la Liga Mexicana de Béisbol (LMB) con datos de `data/lmb.db`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, accuracy_score

import db


def build_lmb_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Extrae características básicas de partidos finalizados de la LMB desde data/lmb.db."""
    query = """
    SELECT
        g.game_pk, g.game_date, g.season, g.home_team_id, g.away_team_id,
        g.home_score, g.away_score, g.venue_id, g.venue_name
    FROM games g
    WHERE g.status = 'Final' AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
    ORDER BY g.game_date ASC, g.game_pk ASC
    """
    games_df = pd.read_sql_query(query, conn)
    if games_df.empty:
        return pd.DataFrame()

    games_df["home_won"] = (games_df["home_score"] > games_df["away_score"]).astype(int)
    games_df["total_runs"] = games_df["home_score"] + games_df["away_score"]

    # Calcular carreras promedio anotadas/recibidas rolling de los últimos 10 partidos
    team_history = {}
    features = []

    for idx, row in games_df.iterrows():
        g_pk = row["game_pk"]
        h_id = row["home_team_id"]
        a_id = row["away_team_id"]

        h_runs = team_history.get(h_id, [])
        a_runs = team_history.get(a_id, [])

        h_avg = np.mean(h_runs[-10:]) if len(h_runs) >= 3 else 5.5
        a_avg = np.mean(a_runs[-10:]) if len(a_runs) >= 3 else 5.5

        # Ajuste de altitud por estadio LMB (ej. Estadio Alfredo Harp Helú / CDMX / Puebla = mayor anotación)
        v_name = str(row["venue_name"]).lower()
        park_boost = 1.15 if ("harp" in v_name or "mexico" in v_name or "puebla" in v_name or "hermanos" in v_name) else 1.0

        feat_row = {
            "game_pk": g_pk,
            "season": row["season"],
            "home_avg_runs": h_avg,
            "away_avg_runs": a_avg,
            "park_boost": park_boost,
            "diff_runs": h_avg - a_avg,
            "proj_total_runs": (h_avg + a_avg) * park_boost * 0.95,
            "home_won": row["home_won"],
            "total_runs": row["total_runs"],
        }
        features.append(feat_row)

        # Actualizar historial
        if h_id not in team_history:
            team_history[h_id] = []
        if a_id not in team_history:
            team_history[a_id] = []

        team_history[h_id].append(row["home_score"])
        team_history[a_id].append(row["away_score"])

    return pd.DataFrame(features)


def train_lmb_models(db_path: str = "data/lmb.db") -> None:
    conn = db.get_connection(db_path)
    df = build_lmb_features(conn)
    conn.close()

    if df.empty or len(df) < 50:
        print(f"[train_lmb_models] Muestra insuficiente en {db_path} ({len(df)} partidos). Entrenando modelo sintetico baseline.")
        # Generar modelo baseline teorico de contingencia si la BD apenas se esta poblando
        df = pd.DataFrame({
            "diff_runs": [0.5, -0.5, 1.2, -1.0, 0.0, 0.8, -0.3, 0.4] * 20,
            "proj_total_runs": [11.0, 9.5, 12.0, 8.5, 10.0, 11.5, 9.0, 10.5] * 20,
            "home_won": [1, 0, 1, 0, 1, 1, 0, 1] * 20,
            "total_runs": [11, 9, 13, 8, 10, 12, 9, 11] * 20,
        })

    feature_cols_win = ["diff_runs"]
    feature_cols_runs = ["proj_total_runs"]

    X_win = df[feature_cols_win]
    y_win = df["home_won"]

    X_runs = df[feature_cols_runs]
    y_runs = df["total_runs"]

    # Entrenar Regresión Logística regularizada para Victoria (evita sobreajuste)
    model_win = LogisticRegression(C=0.5, random_state=42)
    model_win.fit(X_win, y_win)

    # Entrenar Ridge para Carreras
    model_runs = Ridge(alpha=2.0, random_state=42)
    model_runs.fit(X_runs, y_runs)

    p_win = model_win.predict_proba(X_win)[:, 1]
    p_runs = model_runs.predict(X_runs)

    loss_win = log_loss(y_win, p_win)
    acc_win = accuracy_score(y_win, (p_win >= 0.5).astype(int))
    mae_runs = mean_absolute_error(y_runs, p_runs)

    print(f"=== ENTRENAMIENTO MODELOS LMB ({len(df)} partidos) ===")
    print(f"Victoria Log-Loss: {loss_win:.4f} | Accuracy: {acc_win*100:.2f}%")
    print(f"Carreras MAE: {mae_runs:.3f}")

    joblib.dump({"model": model_win, "feature_names": feature_cols_win, "model_type": "LogisticRegression", "league": "LMB"}, "data/model_lmb_win.joblib")
    joblib.dump({"model": model_runs, "feature_names": feature_cols_runs, "model_type": "Ridge", "league": "LMB"}, "data/model_lmb_runs.joblib")
    print("Guardados data/model_lmb_win.joblib y data/model_lmb_runs.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos LMB")
    parser.add_argument("--db-path", default="data/lmb.db")
    args = parser.parse_args()
    train_lmb_models(args.db_path)
