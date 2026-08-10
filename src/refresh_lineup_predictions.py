"""
refresh_lineup_predictions.py
Fase 5: Segunda pasada de inferencia en tiempo real (1-3 horas antes del primer lanzamiento).
Si la alineación titular está confirmada en la Stats API (hydrate=lineups), calcula la proyección
ofensiva ponderada por los 9 bateadores específicos y guarda una NUEVA fila en `predictions_log`
marcada con `prediction_stage = 'lineup_confirmed'`. No sobrescribe la predicción matutina.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import joblib
import pandas as pd

import db
import features
import features_lineup
import pdf_generator
import predict_today


def run_lineup_refresh(
    target_date: str | None = None,
    db_path: str = "data/mlb.db",
    win_model_path: str = "data/model_win.joblib",
    runs_model_path: str = "data/model_runs.joblib",
    league: str = "MLB",
) -> int:
    target_date = target_date or str(date.today())
    conn = db.get_connection(db_path)
    db.init_db(conn)

    # 1. Obtener partidos programados para la fecha
    games_rows = conn.execute("""
        SELECT game_pk, home_team_id, away_team_id, status, game_date
        FROM games
        WHERE game_date = ? AND status != 'Final'
    """, (target_date,)).fetchall()

    if not games_rows:
        print(f"[refresh_lineup] No hay partidos pendientes para {target_date}.")
        conn.close()
        return 0

    print(f"=== REVISANDO ALINEACIONES CONFIRMADAS ({len(games_rows)} partidos) ===")

    win_saved = joblib.load(win_model_path)
    runs_saved = joblib.load(runs_model_path)

    refreshed_count = 0
    now_iso = datetime.now().isoformat(timespec="seconds")

    for g_pk, h_id, a_id, status, g_date in games_rows:
        # Consultar alineacion confirmada en vivo
        lineup_pids = features_lineup.fetch_confirmed_lineup_live(g_date, g_pk)
        if not lineup_pids:
            continue

        h_pids, a_pids = lineup_pids
        h_lineup_woba = features_lineup.get_lineup_projected_woba(conn, h_pids, g_date)
        a_lineup_woba = features_lineup.get_lineup_projected_woba(conn, a_pids, g_date)

        # Construir fila de caracteristicas para el partido
        rows = features.build_features_for_date(conn, g_date, league=league)
        game_feat = [r for r in rows if r["game_pk"] == g_pk]
        if not game_feat:
            continue

        feat_dict = game_feat[0]

        # Sobrescribir wOBA de equipo por el wOBA de la alineacion titular confirmada
        feat_dict["home_woba_rolling"] = h_lineup_woba
        feat_dict["away_woba_rolling"] = a_lineup_woba
        feat_dict["woba_diff"] = h_lineup_woba - a_lineup_woba

        df_pred = predict_today.predict_games(conn, [feat_dict], win_saved, runs_saved)

        if not df_pred.empty:
            r_pred = df_pred.iloc[0]
            conn.execute("""
                INSERT OR REPLACE INTO predictions_log
                (game_pk, predicted_at, home_win_proba, total_runs_pred,
                 win_model_type, runs_model_type, weather_temp, weather_wind, prediction_stage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'lineup_confirmed')
            """, (
                g_pk,
                now_iso,
                float(r_pred["home_win_proba"]),
                float(r_pred["total_runs_pred"]),
                win_saved.get("model_type", "XGBoost"),
                runs_saved.get("model_type", "XGBoost"),
                int(r_pred["weather_temp"]) if pd.notna(r_pred.get("weather_temp")) else None,
                str(r_pred["weather_wind"]) if pd.notna(r_pred.get("weather_wind")) else None,
            ))
            conn.commit()
            refreshed_count += 1
            print(f"  ⚡ Alineación confirmada y refrescada para game_pk {g_pk}: Proba={r_pred['home_win_proba']:.3f}, Runs={r_pred['total_runs_pred']:.1f}")

    conn.close()
    print(f"Refrescadas {refreshed_count} predicciones con alineación confirmada.")
    return refreshed_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresco de predicciones con alineación confirmada")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--league", default="MLB")
    args = parser.parse_args()

    run_lineup_refresh(args.date, args.db_path, league=args.league)
