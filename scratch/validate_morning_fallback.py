"""
validate_morning_fallback.py
Simula la predicción matutina real en la temporada de Validación 2024 para evaluar si un modelo
entrenado con wOBA de alineación se degrada cuando se ejecuta por la mañana sin alineaciones confirmadas
(fallback a promedio de equipo / wOBA de liga), comparándolo contra el modelo base de producción.
"""

from __future__ import annotations

import sys
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import db
import metrics
import model_data as md

print("=== SIMULACIÓN RIGUROSA: ESCENARIO MATUTINO REAL EN VALIDACIÓN 2024 ===")

conn = db.get_connection("data/mlb.db")
db.init_db(conn)

ds_path = "data/training_dataset.parquet"
df = md.load_dataset(ds_path)

df_train = df[df["season"].isin(md.TRAIN_SEASONS)].copy()
df_val = df[df["season"].isin(md.VAL_SEASONS)].copy()

# Buscar columna de wOBA de equipo
woba_team_cols = [c for c in df.columns if "woba" in c]
print("Columnas de wOBA encontradas en dataset:", woba_team_cols)

h_col = [c for c in woba_team_cols if "home" in c][0]
a_col = [c for c in woba_team_cols if "away" in c][0]
print(f"Usando {h_col} y {a_col} como fallback de equipo en la mañana.")

# Cargar promedios de wOBA por bateador y alineaciones históricas
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

df_val_lineup = attach_lineup_features(df_val)
df_train_lineup = attach_lineup_features(df_train)

lineup_cols = ["home_lineup_woba", "away_lineup_woba", "diff_lineup_woba"]
feature_cols_base = [c for c in md.feature_columns(df_train_lineup) if c not in lineup_cols]
feature_cols_new = feature_cols_base + lineup_cols

X_tr_base = df_train_lineup[feature_cols_base]
X_tr_new = df_train_lineup[feature_cols_new]
y_tr_win = df_train_lineup[md.TARGET_WIN]
y_tr_runs = df_train_lineup[md.TARGET_RUNS]

# Entrenar Modelo Base (Hiperparámetros optimizados)
params_win_base = {'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 400, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5, 'reg_lambda': 1.5, 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1}
params_runs_base = {'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 400, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5, 'reg_lambda': 1.5, 'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1}

xgb_win_base = XGBClassifier(**params_win_base).fit(X_tr_base, y_tr_win)
xgb_runs_base = XGBRegressor(**params_runs_base).fit(X_tr_base, y_tr_runs)

# Entrenar Modelo Nuevo (Hiperparámetros optimizados para Alineación)
params_win_new = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 0.6, 'colsample_bytree': 0.6, 'min_child_weight': 3, 'reg_lambda': 1.5, 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1}
params_runs_new = {'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 400, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5, 'reg_lambda': 1.5, 'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1}

xgb_win_new = XGBClassifier(**params_win_new).fit(X_tr_new, y_tr_win)
xgb_runs_new = XGBRegressor(**params_runs_new).fit(X_tr_new, y_tr_runs)

# --- EVALUACIÓN 1: MODELO BASE EN VALIDACIÓN 2024 ---
X_val_base = df_val_lineup[feature_cols_base]
y_val_win = df_val_lineup[md.TARGET_WIN]
y_val_runs = df_val_lineup[md.TARGET_RUNS]

p_win_base = xgb_win_base.predict_proba(X_val_base)[:, 1]
p_runs_base = xgb_runs_base.predict(X_val_base)

ll_base = log_loss(y_val_win, p_win_base)
acc_base = accuracy_score(y_val_win, (p_win_base >= 0.5).astype(int))
brier_base = brier_score_loss(y_val_win, p_win_base)
mae_base = mean_absolute_error(y_val_runs, p_runs_base)

# --- EVALUACIÓN 2: NUEVO MODELO EN TARDE (Alineación Titular Disponible) ---
X_val_tarde = df_val_lineup[feature_cols_new]
p_win_tarde = xgb_win_new.predict_proba(X_val_tarde)[:, 1]
p_runs_tarde = xgb_runs_new.predict(X_val_tarde)

ll_tarde = log_loss(y_val_win, p_win_tarde)
acc_tarde = accuracy_score(y_val_win, (p_win_tarde >= 0.5).astype(int))
brier_tarde = brier_score_loss(y_val_win, p_win_tarde)
mae_tarde = mean_absolute_error(y_val_runs, p_runs_tarde)

# --- EVALUACIÓN 3: NUEVO MODELO EN MAÑANA (Simulación sin Alineación Confirmada -> Reemplazo por team_woba de fallback) ---
df_val_manana = df_val_lineup.copy()
df_val_manana["home_lineup_woba"] = df_val_manana[h_col]
df_val_manana["away_lineup_woba"] = df_val_manana[a_col]
df_val_manana["diff_lineup_woba"] = df_val_manana["home_lineup_woba"] - df_val_manana["away_lineup_woba"]

X_val_manana = df_val_manana[feature_cols_new]
p_win_manana = xgb_win_new.predict_proba(X_val_manana)[:, 1]
p_runs_manana = xgb_runs_new.predict(X_val_manana)

ll_manana = log_loss(y_val_win, p_win_manana)
acc_manana = accuracy_score(y_val_win, (p_win_manana >= 0.5).astype(int))
brier_manana = brier_score_loss(y_val_win, p_win_manana)
mae_manana = mean_absolute_error(y_val_runs, p_runs_manana)

print("\n" + "="*85)
print("  EVALUACIÓN COMPARATIVA: ESCENARIO MATUTINO VS VESPERTINO EN VALIDACIÓN 2024")
print("="*85)
print(f"  1. Modelo BASE Producción (Sin alineación): LogLoss={ll_base:.4f} | Acc={acc_base*100:.2f}% | Brier={brier_base:.4f} | MAE={mae_base:.4f}")
print(f"  2. Modelo NUEVO en la MAÑANA (Simulando Fallback): LogLoss={ll_manana:.4f} | Acc={acc_manana*100:.2f}% | Brier={brier_manana:.4f} | MAE={mae_manana:.4f}")
print(f"  3. Modelo NUEVO en la TARDE (Alineación Confirmada): LogLoss={ll_tarde:.4f} | Acc={acc_tarde*100:.2f}% | Brier={brier_tarde:.4f} | MAE={mae_tarde:.4f}")
print("="*85)
