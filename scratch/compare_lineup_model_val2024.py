import sys
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import db
import metrics
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

print("=== FASE 5: EVALUACIÓN DE MODELO POR ALINEACIÓN TITULAR EN VALIDACIÓN 2024 ===")

conn = db.get_connection("data/mlb.db")
db.init_db(conn)

ds_path = "data/training_dataset.parquet"
df = pd.read_parquet(ds_path)
df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
df["total_runs"] = df["home_score"] + df["away_score"]
print(f"Dataset cargado: {len(df)} partidos totales.")

df_train = df[(df["season"] >= 2015) & (df["season"] <= 2023)].copy()
df_val = df[df["season"] == 2024].copy()

# Cargar promedios de wOBA por bateador por temporada previa
print("Calculando promedios sabermetricos rolling por bateador...")
df_bat = pd.read_sql_query("""
    SELECT b.player_id, g.season,
           SUM(b.ab) AS ab, SUM(b.h) AS h, SUM(b.doubles) AS d, SUM(b.triples) AS t,
           SUM(b.hr) AS hr, SUM(b.bb) AS bb, SUM(b.ibb) AS ibb, SUM(b.hbp) AS hbp, SUM(b.sf) AS sf
    FROM boxscore_batting b
    JOIN games g ON g.game_pk = b.game_pk
    WHERE g.status='Final'
    GROUP BY b.player_id, g.season
""", conn)

# Pre-calcular wOBA por bateador por temporada
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

# Obtener los 9 titulares históricos por partido
print("Pre-cargando alineaciones titulares de partidos...")
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

def process_features(sub_df: pd.DataFrame) -> pd.DataFrame:
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
                # Buscar wOBA de la temporada actual o previa
                w_val = woba_map.get((pid, season), woba_map.get((pid, prev_season), 0.315))
                w_sum += w_val * w
                tot_w += w
            target.append(round(w_sum / tot_w, 4) if tot_w > 0 else 0.315)

    sub_df["home_lineup_woba"] = h_woba
    sub_df["away_lineup_woba"] = a_woba
    sub_df["diff_lineup_woba"] = sub_df["home_lineup_woba"] - sub_df["away_lineup_woba"]
    return sub_df

print("Procesando wOBA por alineacion para validacion 2024...")
df_val_lineup = process_features(df_val)

print("Procesando wOBA por alineacion para entrenamiento (2015-2023)...")
df_train_lineup = process_features(df_train)

import model_data

lineup_cols = ["home_lineup_woba", "away_lineup_woba", "diff_lineup_woba"]

# 1. MODELO BASE (Sin alineación individual - wOBA equipo)
feature_cols_base = [c for c in model_data.feature_columns(df_train_lineup) if c not in lineup_cols]

X_tr_base = df_train_lineup[feature_cols_base]
y_tr_win = df_train_lineup["home_win"]
y_tr_runs = df_train_lineup["total_runs"]

X_val_base = df_val_lineup[feature_cols_base]
y_val_win = df_val_lineup["home_win"]
y_val_runs = df_val_lineup["total_runs"]

xgb_win_base = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42, eval_metric="logloss")
xgb_win_base.fit(X_tr_base, y_tr_win)

xgb_runs_base = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42)
xgb_runs_base.fit(X_tr_base, y_tr_runs)

p_win_base = xgb_win_base.predict_proba(X_val_base)[:, 1]
p_runs_base = xgb_runs_base.predict(X_val_base)

ll_base = log_loss(y_val_win, p_win_base)
acc_base = accuracy_score(y_val_win, (p_win_base >= 0.5).astype(int))
brier_base = brier_score_loss(y_val_win, p_win_base)
mae_base = mean_absolute_error(y_val_runs, p_runs_base)
rmse_base = np.sqrt(mean_squared_error(y_val_runs, p_runs_base))

# 2. MODELO NUEVO (Con wOBA por alineación titular)
feature_cols_new = feature_cols_base + lineup_cols

X_tr_new = df_train_lineup[feature_cols_new]
X_val_new = df_val_lineup[feature_cols_new]

xgb_win_new = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42, eval_metric="logloss")
xgb_win_new.fit(X_tr_new, y_tr_win)

xgb_runs_new = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42)
xgb_runs_new.fit(X_tr_new, y_tr_runs)

p_win_new = xgb_win_new.predict_proba(X_val_new)[:, 1]
p_runs_new = xgb_runs_new.predict(X_val_new)

ll_new = log_loss(y_val_win, p_win_new)
acc_new = accuracy_score(y_val_win, (p_win_new >= 0.5).astype(int))
brier_new = brier_score_loss(y_val_win, p_win_new)
mae_new = mean_absolute_error(y_val_runs, p_runs_new)
rmse_new = np.sqrt(mean_squared_error(y_val_runs, p_runs_new))

print("\n" + "="*80)
print("  RESULTADOS COMPARATIVOS EN VALIDACIÓN 2024 (ANTES VS DESPUÉS)")
print("="*80)
print(f"  Log-Loss Victoria:   {ll_base:.4f}  --->  {ll_new:.4f}  (Delta: {ll_new - ll_base:+.4f})")
print(f"  Accuracy Victoria:   {acc_base*100:.2f}% --->  {acc_new*100:.2f}% (Delta: {(acc_new - acc_base)*100:+.2f}%)")
print(f"  Brier Score:        {brier_base:.4f}  --->  {brier_new:.4f}  (Delta: {brier_new - brier_base:+.4f})")
print(f"  MAE Carreras:       {mae_base:.4f}  --->  {mae_new:.4f}  (Delta: {mae_new - mae_base:+.4f})")
print(f"  RMSE Carreras:      {rmse_base:.4f}  --->  {rmse_new:.4f}  (Delta: {rmse_new - rmse_base:+.4f})")
print("="*80)
