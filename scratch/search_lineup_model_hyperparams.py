"""
search_lineup_model_hyperparams.py
Búsqueda rigurosa de hiperparámetros (Random Search con 30 iteraciones) usando model_search.py
para comparar la versión BASE (Promedio Equipo) vs la versión NUEVA (Alineación Titular wOBA)
exclusivamente en el set de VALIDACIÓN temporal (2024), sin tocar el set de test.
"""

from __future__ import annotations

import sys
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import db
import metrics
import model_data as md
import model_search as ms

print("=== FASE 5: BÚSQUEDA COMPLETA DE HIPERPARÁMETROS (BASE VS ALINEACIÓN TITULAR) ===")

conn = db.get_connection("data/mlb.db")
db.init_db(conn)

ds_path = "data/training_dataset.parquet"
df = md.load_dataset(ds_path)
print(f"Dataset cargado: {len(df)} partidos totales.")

df_train = df[df["season"].isin(md.TRAIN_SEASONS)].copy()
df_val = df[df["season"].isin(md.VAL_SEASONS)].copy()

print(f"Train (2015-2023): {len(df_train)} filas | Val (2024): {len(df_val)} filas.")

# 1. Cargar estadísticas de bateadores y alineaciones históricas
print("Cargando promedios sabermétricos por bateador...")
df_bat = pd.read_sql_query("""
    SELECT b.player_id, g.season,
           SUM(b.ab) AS ab, SUM(b.h) AS h, SUM(b.doubles) AS d, SUM(b.triples) AS t,
           SUM(b.hr) AS hr, SUM(b.bb) AS bb, SUM(b.ibb) AS ibb, SUM(b.hbp) AS hbp, SUM(b.sf) AS sf
    FROM boxscore_batting b
    JOIN games g ON g.game_pk = b.game_pk
    WHERE g.status='Final'
    GROUP BY b.player_id, g.season
""", conn)

woba_map = {}
for idx, r in df_bat.iterrows():
    pid = int(r["player_id"])
    season = int(r["season"])
    ab, h, d, t, hr, bb, ibb, hbp, sf = r["ab"], r["h"], r["d"], r["t"], r["hr"], r["bb"], r["ibb"], r["hbp"], r["sf"]
    singles = h - (d + t + hr)
    pa = ab + (bb - ibb) + sf + hbp
    w_raw = metrics.woba(bb=bb, ibb=ibb, hbp=hbp, singles=singles, doubles=d, triples=t, hr=hr, ab=ab, sf=sf, season=season)
    w_adj = metrics.shrink_rate(w_raw, pa, league_rate=0.315, k=40.0)
    woba_map[(pid, season)] = w_adj

print("Pre-cargando alineaciones titulares...")
df_lineups = pd.read_sql_query("""
    SELECT b.game_pk, b.team_id, b.player_id, b.batting_order
    FROM boxscore_batting b
    WHERE b.batting_order IS NOT NULL OR b.rowid IN (
        SELECT rowid FROM boxscore_batting WHERE game_pk=b.game_pk AND team_id=b.team_id LIMIT 9
    )
""", conn)
conn.close()

lineup_dict = {}
for (g_pk, team_id), group in df_lineups.groupby(["game_pk", "team_id"]):
    lineup_dict[(g_pk, team_id)] = group.sort_values("batting_order")["player_id"].head(9).tolist()

def attach_lineup_features(sub_df: pd.DataFrame) -> pd.DataFrame:
    sub_df = sub_df.copy()
    h_woba = []
    a_woba = []
    weights = [1.2, 1.2, 1.1, 1.1, 1.0, 1.0, 0.9, 0.8, 0.7]

    for idx, row in sub_df.iterrows():
        g_pk = int(row["game_pk"])
        season = int(row["season"])
        prev_season = season - 1

        for t_id, target in [(int(row["home_team_id"]), h_woba), (int(row["away_team_id"]), a_woba)]:
            pids = lineup_dict.get((g_pk, t_id), [])
            if not pids:
                target.append(0.315)
                continue

            w_sum = 0.0
            tot_w = 0.0
            for i, pid in enumerate(pids[:9]):
                w = weights[i] if i < len(weights) else 1.0
                w_val = woba_map.get((pid, season), woba_map.get((pid, prev_season), 0.315))
                w_sum += w_val * w
                tot_w += w
            target.append(round(w_sum / tot_w, 4) if tot_w > 0 else 0.315)

    sub_df["home_lineup_woba"] = h_woba
    sub_df["away_lineup_woba"] = a_woba
    sub_df["diff_lineup_woba"] = sub_df["home_lineup_woba"] - sub_df["away_lineup_woba"]
    return sub_df

print("Construyendo datasets con alineación para train y val...")
df_val_lineup = attach_lineup_features(df_val)
df_train_lineup = attach_lineup_features(df_train)

lineup_cols = ["home_lineup_woba", "away_lineup_woba", "diff_lineup_woba"]

feature_cols_base = [c for c in md.feature_columns(df_train_lineup) if c not in lineup_cols]
feature_cols_new = feature_cols_base + lineup_cols

X_tr_base, y_tr_win, y_tr_runs = df_train_lineup[feature_cols_base], df_train_lineup[md.TARGET_WIN], df_train_lineup[md.TARGET_RUNS]
X_val_base, y_val_win, y_val_runs = df_val_lineup[feature_cols_base], df_val_lineup[md.TARGET_WIN], df_val_lineup[md.TARGET_RUNS]

X_tr_new = df_train_lineup[feature_cols_new]
X_val_new = df_val_lineup[feature_cols_new]

# Definición de espacios de búsqueda
XGB_WIN_SPACE = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 300, 400],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_lambda": [1, 1.5, 2, 3],
}
XGB_WIN_FIXED = {"objective": "binary:logistic", "eval_metric": "logloss", "random_state": 42, "verbosity": 0, "n_jobs": -1}

XGB_RUNS_SPACE = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 300, 400],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_lambda": [1, 1.5, 2, 3],
}
XGB_RUNS_FIXED = {"objective": "reg:squarederror", "random_state": 42, "verbosity": 0, "n_jobs": -1}

def score_logloss(model, X, y) -> float:
    proba = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return log_loss(y, proba)

def score_rmse(model, X, y) -> float:
    pred = model.predict(X)
    return mean_squared_error(y, pred) ** 0.5

print("\n--- 1. BÚSQUEDA DE HIPERPARÁMETROS: MODELO BASE (Sin Alineación) ---")
best_win_base, _ = ms.random_search(
    XGBClassifier, XGB_WIN_SPACE, XGB_WIN_FIXED,
    X_tr_base, y_tr_win, X_val_base, y_val_win,
    score_fn=score_logloss, higher_is_better=False, n_iter=25, label="Win-Base"
)

best_runs_base, _ = ms.random_search(
    XGBRegressor, XGB_RUNS_SPACE, XGB_RUNS_FIXED,
    X_tr_base, y_tr_runs, X_val_base, y_val_runs,
    score_fn=score_rmse, higher_is_better=False, n_iter=25, label="Runs-Base"
)

print("\n--- 2. BÚSQUEDA DE HIPERPARÁMETROS: MODELO NUEVO (Con Alineación Titular) ---")
best_win_new, _ = ms.random_search(
    XGBClassifier, XGB_WIN_SPACE, XGB_WIN_FIXED,
    X_tr_new, y_tr_win, X_val_new, y_val_win,
    score_fn=score_logloss, higher_is_better=False, n_iter=25, label="Win-Lineup"
)

best_runs_new, _ = ms.random_search(
    XGBRegressor, XGB_RUNS_SPACE, XGB_RUNS_FIXED,
    X_tr_new, y_tr_runs, X_val_new, y_val_runs,
    score_fn=score_rmse, higher_is_better=False, n_iter=25, label="Runs-Lineup"
)

# Métricas finales de evaluación en Validación 2024
p_win_b = best_win_base["model"].predict_proba(X_val_base)[:, 1]
p_runs_b = best_runs_base["model"].predict(X_val_base)

ll_b = log_loss(y_val_win, p_win_b)
acc_b = accuracy_score(y_val_win, (p_win_b >= 0.5).astype(int))
brier_b = brier_score_loss(y_val_win, p_win_b)
mae_b = mean_absolute_error(y_val_runs, p_runs_b)
rmse_b = mean_squared_error(y_val_runs, p_runs_b) ** 0.5

p_win_n = best_win_new["model"].predict_proba(X_val_new)[:, 1]
p_runs_n = best_runs_new["model"].predict(X_val_new)

ll_n = log_loss(y_val_win, p_win_n)
acc_n = accuracy_score(y_val_win, (p_win_n >= 0.5).astype(int))
brier_n = brier_score_loss(y_val_win, p_win_n)
mae_n = mean_absolute_error(y_val_runs, p_runs_n)
rmse_n = mean_squared_error(y_val_runs, p_runs_n) ** 0.5

print("\n" + "="*85)
print("  TABLA COMPARATIVA FINAL CON HIPERPARÁMETROS OPTIMIZADOS (VALIDACIÓN 2024)")
print("="*85)
print(f"  Log-Loss Victoria:   {ll_b:.4f}  --->  {ll_n:.4f}  (Delta: {ll_n - ll_b:+.4f})")
print(f"  Accuracy Victoria:   {acc_b*100:.2f}% --->  {acc_n*100:.2f}% (Delta: {(acc_n - acc_b)*100:+.2f}%)")
print(f"  Brier Score:        {brier_b:.4f}  --->  {brier_n:.4f}  (Delta: {brier_n - brier_b:+.4f})")
print(f"  MAE Carreras:       {mae_b:.4f}  --->  {mae_n:.4f}  (Delta: {mae_n - mae_b:+.4f})")
print(f"  RMSE Carreras:      {rmse_b:.4f}  --->  {rmse_n:.4f}  (Delta: {rmse_n - rmse_b:+.4f})")
print("="*85)
print("\nHiperparámetros Óptimos Modelo Victoria (Base):", best_win_base["params"])
print("Hiperparámetros Óptimos Modelo Victoria (Alineación):", best_win_new["params"])
print("Hiperparámetros Óptimos Modelo Carreras (Base):", best_runs_base["params"])
print("Hiperparámetros Óptimos Modelo Carreras (Alineación):", best_runs_new["params"])
