import sqlite3
import pandas as pd

conn = sqlite3.connect("data/mlb.db")

print("=== 1. CONSULTA ANTERIOR (sin deduplicar predictions_log por game_pk) ===")
sql_old = """
    SELECT
        g.game_pk,
        p.home_win_proba,
        p.total_runs_pred,
        g.game_date,
        g.home_score,
        g.away_score,
        (g.home_score + g.away_score) AS actual_total,
        CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS home_won
    FROM games g
    JOIN game_features gf ON gf.game_pk = g.game_pk
    LEFT JOIN predictions_log p ON p.game_pk = g.game_pk
    WHERE g.status = 'Final' AND g.game_type = 'R' AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
          AND p.home_win_proba IS NOT NULL
"""
df_old = pd.read_sql_query(sql_old, conn)
print(f"Total filas recuperadas con SQL anterior: {len(df_old)}")
print(f"Partidos únicos con SQL anterior: {df_old['game_pk'].nunique()}")
print(f"Activaciones ML Local (>=0.60): {(df_old['home_win_proba'] >= 0.60).sum()}")
print(f"Activaciones ML Visitante (<=0.40): {(df_old['home_win_proba'] <= 0.40).sum()}")
print(f"Activaciones Over 8.5 (>=8.5): {(df_old['total_runs_pred'] >= 8.5).sum()}")
print(f"Activaciones Under 8.5 (<8.5): {(df_old['total_runs_pred'] < 8.5).sum()}")

print("\n=== 2. CONSULTA RIGUROSA Y CORREGIDA (deduplicando predictions_log por max(predicted_at)) ===")
sql_clean = """
    SELECT
        g.game_pk,
        p.home_win_proba,
        p.total_runs_pred,
        g.game_date,
        g.home_score,
        g.away_score,
        (g.home_score + g.away_score) AS actual_total,
        CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS home_won
    FROM games g
    JOIN (
        SELECT game_pk, MAX(predicted_at) AS max_pred
        FROM predictions_log
        GROUP BY game_pk
    ) latest ON latest.game_pk = g.game_pk
    JOIN predictions_log p ON p.game_pk = latest.game_pk AND p.predicted_at = latest.max_pred
    WHERE g.status = 'Final' AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
"""
df_clean = pd.read_sql_query(sql_clean, conn)
print(f"Total filas recuperadas con SQL deduplicado: {len(df_clean)}")
print(f"Partidos únicos con SQL deduplicado: {df_clean['game_pk'].nunique()}")
print(f"Activaciones ML Local (>=0.60): {(df_clean['home_win_proba'] >= 0.60).sum()}")
print(f"Activaciones ML Visitante (<=0.40): {(df_clean['home_win_proba'] <= 0.40).sum()}")
print(f"Activaciones Over 8.5 (>=8.5): {(df_clean['total_runs_pred'] >= 8.5).sum()}")
print(f"Activaciones Under 8.5 (<8.5): {(df_clean['total_runs_pred'] < 8.5).sum()}")

conn.close()
