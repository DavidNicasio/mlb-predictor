"""
feedback_loop.py
Módulo de auto-retroalimentación y calibración continua con ventana móvil
y bandas de confianza.

Cambios respecto a la versión original:
  1. Ventana móvil de 75 días (no todo el historial acumulado)
  2. Calibración de probabilidad separada por bandas:
     - 50-55%: partidos tipo coin-flip
     - 55-62%: confianza media
     - 62%+: alta confianza
  3. Sesgo de carreras separado por rango de predicción:
     - Bajo (<7.5 carreras proyectadas)
     - Medio (7.5-9.5)
     - Alto (>9.5)
"""

from __future__ import annotations

import pandas as pd


WINDOW_DAYS = 75

# Bandas de probabilidad (borde inferior inclusive, superior exclusivo)
PROBA_BANDS = [
    ("50_55", 0.50, 0.55),
    ("55_62", 0.55, 0.62),
    ("62_plus", 0.62, 1.01),
]

# Rangos de predicción de carreras
RUNS_BANDS = [
    ("low", 0.0, 7.5),
    ("mid", 7.5, 9.5),
    ("high", 9.5, 50.0),
]


def get_feedback_metrics(conn, window_days: int = WINDOW_DAYS) -> dict:
    """Calcula métricas de calibración por banda usando una ventana móvil.

    Retorna un dict con:
      - proba_bias_<band>: sesgo de probabilidad por banda de confianza
      - runs_bias_<band>: sesgo de carreras por rango de predicción
      - n_evaluated: número total de juegos evaluados en la ventana
    """
    query = f"""
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
        WHERE g.status = 'Final'
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
              AND p.predicted_at >= date('now', '-{window_days} days')
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        df = pd.DataFrame()

    if df.empty or len(df) < 5:
        return {"n_evaluated": len(df)}

    result: dict = {"n_evaluated": len(df)}

    # --- Calibración de probabilidad por bandas ---
    # Normalizar a probabilidad del favorito (siempre >= 0.50)
    df["fav_proba"] = df["home_win_proba"].apply(lambda p: p if p >= 0.50 else 1.0 - p)
    df["fav_won"] = df.apply(
        lambda r: (r["home_won"] == 1 and r["home_win_proba"] >= 0.50) or
                  (r["home_won"] == 0 and r["home_win_proba"] < 0.50),
        axis=1,
    ).astype(int)

    for band_name, lo, hi in PROBA_BANDS:
        mask = (df["fav_proba"] >= lo) & (df["fav_proba"] < hi)
        band_df = df[mask]
        if len(band_df) >= 10:
            actual_win_rate = float(band_df["fav_won"].mean())
            pred_avg = float(band_df["fav_proba"].mean())
            bias = actual_win_rate - pred_avg
            result[f"proba_bias_{band_name}"] = round(bias, 4)
            result[f"proba_n_{band_name}"] = len(band_df)
        else:
            result[f"proba_bias_{band_name}"] = 0.0
            result[f"proba_n_{band_name}"] = len(band_df)

    # --- Sesgo de carreras por rango de predicción ---
    df["runs_diff"] = df["actual_total"] - df["total_runs_pred"]

    for band_name, lo, hi in RUNS_BANDS:
        mask = (df["total_runs_pred"] >= lo) & (df["total_runs_pred"] < hi)
        band_df = df[mask]
        if len(band_df) >= 10:
            result[f"runs_bias_{band_name}"] = round(float(band_df["runs_diff"].mean()), 3)
            result[f"runs_n_{band_name}"] = len(band_df)
        else:
            result[f"runs_bias_{band_name}"] = 0.0
            result[f"runs_n_{band_name}"] = len(band_df)

    return result


def apply_feedback_corrections(df: pd.DataFrame, conn) -> pd.DataFrame:
    """Aplica correcciones de calibración por banda basadas en la ventana reciente."""
    if df.empty:
        return df

    metrics = get_feedback_metrics(conn)
    n = metrics.get("n_evaluated", 0)

    if n < 15:
        return df  # no hay suficiente historial para corregir

    df_corrected = df.copy()

    # --- Ajuste de probabilidad por banda ---
    for idx, r in df_corrected.iterrows():
        p = float(r["home_win_proba"])
        fav_p = p if p >= 0.50 else 1.0 - p

        # Encontrar la banda correspondiente
        band_bias = 0.0
        for band_name, lo, hi in PROBA_BANDS:
            if lo <= fav_p < hi:
                band_bias = metrics.get(f"proba_bias_{band_name}", 0.0)
                break

        # Aplicar 50% del sesgo de la banda (conservador)
        if band_bias != 0.0:
            adj = max(-0.06, min(0.06, band_bias * 0.5))
            # Ajustar en la dirección del favorito
            if p >= 0.50:
                df_corrected.at[idx, "home_win_proba"] = max(0.05, min(0.95, p + adj))
            else:
                df_corrected.at[idx, "home_win_proba"] = max(0.05, min(0.95, p - adj))

    # --- Ajuste de carreras por rango ---
    for idx, r in df_corrected.iterrows():
        runs_pred = float(r["total_runs_pred"])

        runs_bias = 0.0
        for band_name, lo, hi in RUNS_BANDS:
            if lo <= runs_pred < hi:
                runs_bias = metrics.get(f"runs_bias_{band_name}", 0.0)
                break

        if runs_bias != 0.0:
            adj = max(-1.0, min(1.0, runs_bias * 0.5))
            df_corrected.at[idx, "total_runs_pred"] = runs_pred + adj

    return df_corrected
