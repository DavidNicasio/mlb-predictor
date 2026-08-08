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

import db
import pdf_generator

OU_TOL = 0.25


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
                p.home_win_proba, p.total_runs_pred, p.predicted_at
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


def count_ungraded(conn, target_date: str) -> int:
    """Partidos de la fecha con predicción guardada que TODAVÍA no están en
    status='Final' (para avisar que faltan por calificar)."""
    row = conn.execute(
        """SELECT COUNT(DISTINCT p.game_pk)
           FROM predictions_log p
                    JOIN games g ON g.game_pk = p.game_pk
           WHERE g.game_date = ? AND (g.status != 'Final' OR g.home_score IS NULL)""",
        (target_date,),
    ).fetchone()
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

    final_mask = df["is_final"]
    if final_mask.any():
        df.loc[final_mask, "actual_total"] = df.loc[final_mask, "home_score"] + df.loc[final_mask, "away_score"]
        df.loc[final_mask, "home_won"] = df.loc[final_mask, "home_score"] > df.loc[final_mask, "away_score"]

        df.loc[final_mask, "actual_winner"] = df[final_mask].apply(
            lambda r: r["home_name"] if r["home_won"] else r["away_name"], axis=1
        )
        df.loc[final_mask, "actual_winner_abbr"] = df[final_mask].apply(
            lambda r: r["home_abbr"] if r["home_won"] else r["away_abbr"], axis=1
        )
        df.loc[final_mask, "win_hit"] = df[final_mask]["predicted_winner"] == df[final_mask]["actual_winner"]

        df.loc[final_mask, "ou_diff"] = df.loc[final_mask, "actual_total"] - df.loc[final_mask, "total_runs_pred"]
        df.loc[final_mask, "ou_label"] = df.loc[final_mask, "ou_diff"].apply(
            lambda d: "SOBRE" if d > OU_TOL else ("BAJO" if d < -OU_TOL else "IGUAL")
        )
    return df


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_games": 0}

    # Filtrar si hay columna is_final
    if "is_final" in df.columns:
        final_df = df[df["is_final"] == True]
    else:
        final_df = df

    if final_df.empty:
        return {"n_games": 0}

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
) -> str:
    target_date = target_date or str(date.today() - timedelta(days=1))
    output_path = output_path or f"reports/report_{target_date}.pdf"

    conn = db.get_connection(db_path)

    # Incluir partidos finalizados y pendientes para el cuadro del día
    daily_raw = fetch_predictions_with_results(conn, start_date=target_date, end_date=target_date, include_pending=True)
    daily_df = compute_grades(daily_raw)
    n_pendientes = count_ungraded(conn, target_date)

    # Para el acumulado histórico, solo partidos finalizados
    cumulative_raw = fetch_predictions_with_results(conn, include_pending=False)
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