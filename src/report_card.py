"""
report_card.py
Genera un reporte en PDF que compara las predicciones guardadas en
`predictions_log` contra el resultado real de los partidos (games.home_score
/ away_score). Incluye soporte para partidos pendientes y finalizados.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import config
import db
import pdf_generator

OU_TOL = config.OU_TOL


def fetch_predictions_with_results(
        conn, start_date: str | None = None, end_date: str | None = None,
        include_pending: bool = False, league: str | None = None
) -> pd.DataFrame:
    """Predicciones guardadas por partido (game_pk). Si include_pending es False,
    solo retorna partidos que ya tienen status='Final' y marcador cargado."""
    query = """
            SELECT
                g.game_pk, g.game_date, g.game_date_utc, g.status, g.home_score, g.away_score,
                g.weather_condition, g.weather_temp, g.weather_wind, COALESCE(g.league, 'MLB') AS league,
                ht.name AS home_name, at.name AS away_name,
                ht.abbreviation AS home_abbr, at.abbreviation AS away_abbr,
                ht.team_id AS home_team_id, at.team_id AS away_team_id,
                p.home_win_proba, p.total_runs_pred, p.predicted_at,
                COALESCE(p.prediction_stage, 'pregame_team_avg') AS prediction_stage
            FROM games g
                     JOIN teams ht ON ht.team_id = g.home_team_id
                     JOIN teams at ON at.team_id = g.away_team_id
                JOIN (
                SELECT game_pk, MAX(predicted_at) AS max_pred
                FROM predictions_log
                GROUP BY game_pk
                ) latest ON latest.game_pk = g.game_pk
                JOIN predictions_log p
                ON p.game_pk = latest.game_pk AND p.predicted_at = latest.max_pred
            WHERE 1=1
            """
    params: list = []
    if league:
        query += " AND COALESCE(g.league, 'MLB') = ?"
        params.append(league)
    if not include_pending:
        query += " AND g.status = 'Final' AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL"
    if start_date:
        query += " AND g.game_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND g.game_date <= ?"
        params.append(end_date)
    query += " ORDER BY g.game_date ASC, g.game_date_utc ASC, g.game_pk ASC"
    return pd.read_sql_query(query, conn, params=params)


def count_ungraded(conn, target_date: str, league: str | None = None) -> int:
    """Partidos de la fecha con predicción guardada que TODAVÍA no están en
    status='Final' (para avisar que faltan por calificar)."""
    query = """SELECT COUNT(DISTINCT p.game_pk)
               FROM predictions_log p
               JOIN games g ON g.game_pk = p.game_pk
               WHERE g.game_date = ? AND (g.status != 'Final' OR g.home_score IS NULL)"""
    params: list = [target_date]
    if league:
        query += " AND COALESCE(g.league, 'MLB') = ?"
        params.append(league)
    row = conn.execute(query, params).fetchone()
    return row[0] if row else 0


def compute_grades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["is_final"] = (df["status"] == "Final") & df["home_score"].notna() & df["away_score"].notna()

    import features_f5
    f5_res = [
        features_f5.calculate_f5_projections(
            r.get("home_fip"), r.get("away_fip"),
            r.get("home_woba_vs_hand"), r.get("away_woba_vs_hand"),
            float(r["total_runs_pred"]), float(r["home_win_proba"])
        ) for _, r in df.iterrows()
    ]
    df["f5_total_runs_pred"] = [f["f5_total_runs_pred"] for f in f5_res]
    df["f5_home_win_proba"] = [f["f5_home_win_proba"] for f in f5_res]

    df["actual_total"] = None
    df["home_won"] = None
    df["predicted_winner"] = df.apply(
        lambda r: r["home_name"] if r["home_win_proba"] >= 0.5 else r["away_name"], axis=1
    )
    df["predicted_winner_abbr"] = df.apply(
        lambda r: r["home_abbr"] if r["home_win_proba"] >= 0.5 else r["away_abbr"], axis=1
    )
    df["favorito_proba"] = df["home_win_proba"].apply(lambda p: p if p >= 0.5 else 1 - p)

    df["actual_winner"] = None
    df["actual_winner_abbr"] = None
    df["win_hit"] = None
    df["ou_diff"] = None
    df["ou_label"] = None

    for i, r in df.iterrows():
        if r["is_final"]:
            hs = float(r["home_score"])
            aws = float(r["away_score"])
            total = hs + aws
            home_won = hs > aws

            df.at[i, "actual_total"] = total
            df.at[i, "home_won"] = home_won

            act_win_name = r["home_name"] if home_won else r["away_name"]
            act_win_abbr = r["home_abbr"] if home_won else r["away_abbr"]
            df.at[i, "actual_winner"] = act_win_name
            df.at[i, "actual_winner_abbr"] = act_win_abbr

            predicted_home_win = r["home_win_proba"] >= 0.5
            df.at[i, "win_hit"] = (predicted_home_win == home_won)

            diff = total - float(r["total_runs_pred"])
            df.at[i, "ou_diff"] = diff
            if diff > OU_TOL:
                df.at[i, "ou_label"] = "SOBRE"
            elif diff < -OU_TOL:
                df.at[i, "ou_label"] = "BAJO"
            else:
                df.at[i, "ou_label"] = "IGUAL"
    return df


def summarize(final_df: pd.DataFrame) -> dict:
    if final_df.empty:
        return {}
    n = len(final_df)
    win_hits = int(final_df["win_hit"].sum())
    bias = float(final_df["ou_diff"].mean())
    return {
        "n_games": n,
        "win_hits": win_hits,
        "win_pct": win_hits / n * 100,
        "mae_runs": float(final_df["ou_diff"].abs().mean()),
        "bias_runs": bias,
        "n_over": int((final_df["ou_diff"] > OU_TOL).sum()),
        "n_under": int((final_df["ou_diff"] < -OU_TOL).sum()),
        "n_igual": int((final_df["ou_diff"].abs() <= OU_TOL).sum()),
    }


def run(
        target_date: str | None = None,
        db_path: str = "data/mlb.db",
        output_path: str | None = None,
        league: str = "MLB",
) -> str:
    target_date = target_date or str(date.today() - timedelta(days=1))
    prefix = f"report_{league.lower()}_" if league != "MLB" else "report_"
    output_path = output_path or f"reports/{prefix}{target_date}.pdf"

    conn = db.get_connection(db_path)

    # Incluir partidos finalizados y pendientes para el cuadro del día
    daily_raw = fetch_predictions_with_results(conn, start_date=target_date, end_date=target_date, include_pending=True, league=league)
    daily_df = compute_grades(daily_raw)
    n_pendientes = count_ungraded(conn, target_date, league=league)

    # Para el acumulado histórico, solo partidos finalizados
    cumulative_raw = fetch_predictions_with_results(conn, include_pending=False, league=league)
    cumulative_df = compute_grades(cumulative_raw)

    pdf_generator.build_report_card_pdf(output_path, target_date, daily_df, cumulative_df, n_pendientes)
    conn.close()

    print(f"Reporte generado: {output_path}")
    final_daily = daily_df[daily_df["is_final"] == True] if not daily_df.empty else daily_df
    if final_daily.empty:
        print(f"  ({n_pendientes} partido(s) pendientes para {target_date})")
    else:
        s = summarize(final_daily)
        print(f"  {target_date}: {s['win_hits']}/{s['n_games']} aciertos de ganador "
              f"({s['win_pct']:.1f}%), error O/U promedio {s['mae_runs']:.2f} carreras. Pendientes: {n_pendientes}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reporte PDF: predicciones vs resultado real")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD a calificar; por defecto, ayer")
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--output", default=None, help="Ruta del PDF de salida")
    args = parser.parse_args()
    run(args.date, args.db_path, args.output)