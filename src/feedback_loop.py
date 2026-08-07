"""
feedback_loop.py
Módulo de auto-retroalimentación y calibración continua.
Analiza el desempeño histórico registrado en predictions_log y games
para calibrar el sesgo de carreras (O/U Bias) y ajustar las probabilidades predichas.
"""

from __future__ import annotations

import pandas as pd


def get_feedback_metrics(conn) -> dict:
    """Calcula el sesgo de carreras y el factor de calibración basado en predicciones pasadas finalizadas."""
    query = """
        SELECT
            p.game_pk,
            p.home_win_proba,
            p.total_runs_pred,
            g.home_score,
            g.away_score,
            (g.home_score + g.away_score) AS actual_total,
            CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS home_won
        FROM predictions_log p
        JOIN games g ON g.game_pk = p.game_pk
        WHERE g.status = 'Final' AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        df = pd.DataFrame()

    if df.empty or len(df) < 5:
        return {"runs_bias": 0.0, "proba_scale": 1.0, "n_evaluated": len(df)}

    # 1. Sesgo promedio de carreras (Real - Proyectado)
    df["runs_diff"] = df["actual_total"] - df["total_runs_pred"]
    runs_bias = float(df["runs_diff"].mean())

    # 2. Calibración de probabilidad (Diferencia de Win Rate vs Proba promediada)
    actual_home_win_rate = float(df["home_won"].mean())
    pred_home_win_rate = float(df["home_win_proba"].mean())
    proba_bias = actual_home_win_rate - pred_home_win_rate

    return {
        "runs_bias": round(runs_bias, 3),
        "proba_bias": round(proba_bias, 3),
        "n_evaluated": len(df),
    }


def apply_feedback_corrections(df: pd.DataFrame, conn) -> pd.DataFrame:
    """Aplica correcciones de calibración basadas en la experiencia pasada del sistema."""
    if df.empty:
        return df

    metrics_info = get_feedback_metrics(conn)
    runs_bias = metrics_info.get("runs_bias", 0.0)
    proba_bias = metrics_info.get("proba_bias", 0.0)

    df_corrected = df.copy()

    # Si hay suficiente historial evaluado (>=5 juegos), ajustar sesgo de carreras moderadamente (50% del bias)
    if metrics_info.get("n_evaluated", 0) >= 5:
        adj = max(-1.0, min(1.0, runs_bias * 0.5))
        df_corrected["total_runs_pred"] = df_corrected["total_runs_pred"] + adj

        # Ajuste de probabilidad
        p_adj = max(-0.08, min(0.08, proba_bias * 0.5))
        df_corrected["home_win_proba"] = (df_corrected["home_win_proba"] + p_adj).clip(0.05, 0.95)

    return df_corrected
