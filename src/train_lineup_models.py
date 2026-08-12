"""
train_lineup_models.py
Entrena los modelos vespertinos de alineación titular (model_win_lineup.joblib y model_runs_lineup.joblib)
con los hiperparámetros óptimos encontrados en la búsqueda aleatoria sobre 2015-2023 (train).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

import db
import metrics
import model_data as md


def prepare_dataset_with_lineups(conn: sqlite3.Connection, ds_path: str = "data/training_dataset.parquet") -> pd.DataFrame:
    df = md.load_dataset(ds_path)

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

    lineup_dict = {}
    for (g_pk, team_id), group in df_lineups.groupby(["game_pk", "team_id"]):
        lineup_dict[(g_pk, team_id)] = group.sort_values("batting_order")["player_id"].head(9).tolist()

    sub_df = df.copy()
    h_woba, a_woba = [], []
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


def train_and_save(
    ds_path: str = "data/training_dataset.parquet",
    db_path: str = "data/mlb.db",
    out_win_path: str = "data/model_win_lineup.joblib",
    out_runs_path: str = "data/model_runs_lineup.joblib",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = db.get_connection(db_path)
    db.init_db(conn)

    df_full = prepare_dataset_with_lineups(conn, ds_path)
    conn.close()

    train, val, test = md.temporal_split(df_full)

    lineup_cols = ["home_lineup_woba", "away_lineup_woba", "diff_lineup_woba"]
    feature_cols = [c for c in md.feature_columns(train) if c not in lineup_cols] + lineup_cols

    X_train, y_train_win, y_train_runs = train[feature_cols], train[md.TARGET_WIN], train[md.TARGET_RUNS]
    X_val, y_val_win, y_val_runs = val[feature_cols], val[md.TARGET_WIN], val[md.TARGET_RUNS]

    # Hiperparámetros Óptimos de Alineación
    params_win = {
        'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100,
        'subsample': 0.6, 'colsample_bytree': 0.6, 'min_child_weight': 3,
        'reg_lambda': 1.5, 'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'random_state': 42, 'n_jobs': -1
    }
    params_runs = {
        'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 400,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5,
        'reg_lambda': 1.5, 'objective': 'reg:squarederror',
        'random_state': 42, 'n_jobs': -1
    }

    win_model = XGBClassifier(**params_win)
    win_model.fit(X_train, y_train_win)

    runs_model = XGBRegressor(**params_runs)
    runs_model.fit(X_train, y_train_runs)

    # Validar en 2024
    p_val_win = win_model.predict_proba(X_val)[:, 1]
    p_val_runs = runs_model.predict(X_val)

    print("=== MODELOS VESPERTINOS DE ALINEACIÓN ENTRENADOS EXITOSAMENTE ===")
    print(f"  Validation 2024 Log-Loss: {log_loss(y_val_win, p_val_win):.4f}")
    print(f"  Validation 2024 Accuracy: {accuracy_score(y_val_win, (p_val_win >= 0.5).astype(int))*100:.2f}%")
    print(f"  Validation 2024 MAE Runs: {mean_absolute_error(y_val_runs, p_val_runs):.4f}")

    joblib.dump({"model": win_model, "feature_names": feature_cols, "model_type": "XGBoost (Lineup)"}, out_win_path)
    joblib.dump({"model": runs_model, "feature_names": feature_cols, "model_type": "XGBoost (Lineup)"}, out_runs_path)

    print(f"Guardados: {out_win_path} y {out_runs_path}")
    return train, val, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds-path", default="data/training_dataset.parquet")
    parser.add_argument("--db-path", default="data/mlb.db")
    args = parser.parse_args()
    train_and_save(args.ds_path, args.db_path)
