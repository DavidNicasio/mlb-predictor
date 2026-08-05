"""
report_card.py
Genera un reporte en PDF que compara las predicciones guardadas en
`predictions_log` contra el resultado real de los partidos (games.home_score
/ away_score), una vez que ya se jugaron.

Requiere que los partidos de la fecha a calificar ya tengan status='Final'
en la base local -- corre pipeline.py (o backfill.py) antes si hace falta
traer los resultados de esa fecha.

Cómo se califica cada partido:
  - Ganador: se compara el favorito del modelo (home_win_proba >= 0.5 -> local,
    si no, visitante) contra quién ganó de verdad.
  - Over/Under: la "línea" es la propia proyección del modelo
    (total_runs_pred). Se compara el total real contra esa proyección y se
    marca SOBRE / BAJO / IGUAL (dentro de +/- 0.25 carreras se considera
    IGUAL). El error y el sesgo promedio (bias) son las métricas clave para
    saber si el modelo tiende a sobreestimar o subestimar el total.

Uso:
    python src/report_card.py                     # califica el día de ayer
    python src/report_card.py --date 2026-08-01    # una fecha específica
    python src/report_card.py --output reports/mi_reporte.pdf
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import db

# Diferencia (en carreras) por debajo de la cual el total real se considera
# "igual" a la proyección en vez de SOBRE/BAJO.
OU_TOL = 0.25


def fetch_predictions_with_results(
        conn, start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    """Última predicción guardada por partido (game_pk), solo de partidos
    que ya tienen status='Final' y marcador cargado."""
    query = """
            SELECT
                g.game_pk, g.game_date, g.home_score, g.away_score,
                ht.name AS home_name, at.name AS away_name,
                ht.abbreviation AS home_abbr, at.abbreviation AS away_abbr,
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
            WHERE g.status = 'Final' AND g.game_type = 'R'
              AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL \
            """
    params: list = []
    if start_date:
        query += " AND g.game_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND g.game_date <= ?"
        params.append(end_date)
    query += " ORDER BY g.game_date, g.game_pk"
    return pd.read_sql_query(query, conn, params=params)


def count_ungraded(conn, target_date: str) -> int:
    """Partidos de la fecha con predicción guardada que TODAVÍA no están en
    status='Final' (para avisar que faltan por calificar, no que fallaron)."""
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
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["home_won"] = df["home_score"] > df["away_score"]

    df["predicted_winner"] = df.apply(
        lambda r: r["home_name"] if r["home_win_proba"] >= 0.5 else r["away_name"], axis=1
    )
    df["actual_winner"] = df.apply(
        lambda r: r["home_name"] if r["home_won"] else r["away_name"], axis=1
    )
    df["predicted_winner_abbr"] = df.apply(
        lambda r: r["home_abbr"] if r["home_win_proba"] >= 0.5 else r["away_abbr"], axis=1
    )
    df["actual_winner_abbr"] = df.apply(
        lambda r: r["home_abbr"] if r["home_won"] else r["away_abbr"], axis=1
    )
    df["favorito_proba"] = df["home_win_proba"].apply(lambda p: p if p >= 0.5 else 1 - p)
    df["win_hit"] = df["predicted_winner"] == df["actual_winner"]

    df["ou_diff"] = df["actual_total"] - df["total_runs_pred"]
    df["ou_label"] = df["ou_diff"].apply(
        lambda d: "SOBRE" if d > OU_TOL else ("BAJO" if d < -OU_TOL else "IGUAL")
    )
    return df


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_games": 0}
    n = len(df)
    win_hits = int(df["win_hit"].sum())
    bias = df["ou_diff"].mean()
    return {
        "n_games": n,
        "win_hits": win_hits,
        "win_pct": win_hits / n * 100,
        "mae_runs": df["ou_diff"].abs().mean(),
        "bias_runs": bias,
        "n_over": int((df["ou_diff"] > OU_TOL).sum()),
        "n_under": int((df["ou_diff"] < -OU_TOL).sum()),
        "n_igual": int((df["ou_diff"].abs() <= OU_TOL).sum()),
    }


def _summary_paragraphs(stats: dict, styles) -> list:
    if stats.get("n_games", 0) == 0:
        return [Paragraph("No hay partidos calificables en este rango todavia.", styles["Normal"])]

    if stats["bias_runs"] > OU_TOL:
        sesgo_txt = "(el modelo tiende a quedarse CORTO frente al total real)"
    elif stats["bias_runs"] < -OU_TOL:
        sesgo_txt = "(el modelo tiende a proyectar POR ENCIMA del total real)"
    else:
        sesgo_txt = "(sin sesgo notable)"

    lines = [
        f"Partidos evaluados: <b>{stats['n_games']}</b>",
        f"Aciertos de ganador: <b>{stats['win_hits']} de {stats['n_games']} "
        f"({stats['win_pct']:.1f}%)</b>",
        f"Error promedio en la proyeccion de carreras totales: "
        f"<b>{stats['mae_runs']:.2f} carreras</b>",
        f"Sesgo promedio del modelo: <b>{stats['bias_runs']:+.2f} carreras</b> {sesgo_txt}",
        f"Partidos SOBRE la proyeccion: {stats['n_over']}  |  "
        f"BAJO la proyeccion: {stats['n_under']}  |  IGUAL: {stats['n_igual']}",
    ]
    return [Paragraph(line, styles["Normal"]) for line in lines]


def build_game_table(df: pd.DataFrame) -> Table:
    header = [
        "Partido", "Favorito", "Marcador\n(V-L)", "Gano",
        "Acerto", "Proy.\nO/U", "Total\nreal", "Comparacion",
    ]
    rows = [header]
    ordered = df.sort_values("game_date")
    for _, r in ordered.iterrows():
        rows.append([
            f"{r['away_abbr']} @ {r['home_abbr']}",
            f"{r['predicted_winner_abbr']}  {r['favorito_proba']:.0%}",
            f"{int(r['away_score'])}-{int(r['home_score'])}",
            r["actual_winner_abbr"],
            "SI" if r["win_hit"] else "NO",
            f"{r['total_runs_pred']:.1f}",
            f"{int(r['actual_total'])}",
            f"{r['ou_label']}  {r['ou_diff']:+.1f}",
        ])

    col_widths = [
        1.5 * inch, 1.2 * inch, 1.0 * inch, 0.8 * inch,
        0.8 * inch, 0.9 * inch, 0.9 * inch, 1.4 * inch,
        ]
    table = Table(rows, colWidths=col_widths, repeatRows=1)

    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
    ]
    for i, (_, r) in enumerate(ordered.iterrows(), start=1):
        color = colors.HexColor("#15803d") if r["win_hit"] else colors.HexColor("#b91c1c")
        style.append(("TEXTCOLOR", (4, i), (4, i), color))
        style.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))
    table.setStyle(TableStyle(style))
    return table


def build_pdf(
        output_path: str,
        target_date: str,
        daily_df: pd.DataFrame,
        cumulative_df: pd.DataFrame,
        n_pendientes: int,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        output_path, pagesize=landscape(letter),
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        leftMargin=0.5 * inch, rightMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Reporte de Predicciones MLB", styles["Title"]))
    story.append(Paragraph(f"Generado el {date.today().isoformat()}", styles["Normal"]))
    story.append(Spacer(1, 18))

    story.append(Paragraph(f"Cuadro del dia: {target_date}", styles["Heading2"]))
    if n_pendientes:
        story.append(Paragraph(
            f"({n_pendientes} partido(s) de esa fecha aun no tienen resultado final "
            "y no se incluyen en este cuadro)", styles["Italic"],
        ))
    story.append(Spacer(1, 8))
    daily_stats = summarize(daily_df)
    story.extend(_summary_paragraphs(daily_stats, styles))
    story.append(Spacer(1, 14))
    if not daily_df.empty:
        story.append(build_game_table(daily_df))

    story.append(PageBreak())
    story.append(Paragraph("Acumulado historico", styles["Heading2"]))
    cum_stats = summarize(cumulative_df)
    if cum_stats.get("n_games", 0) > 0:
        date_min = cumulative_df["game_date"].min()
        date_max = cumulative_df["game_date"].max()
        story.append(Paragraph(f"Rango de fechas: {date_min} a {date_max}", styles["Normal"]))
    story.append(Spacer(1, 6))
    story.extend(_summary_paragraphs(cum_stats, styles))

    doc.build(story)


def run(
        target_date: str | None = None,
        db_path: str = "data/mlb.db",
        output_path: str | None = None,
) -> str:
    target_date = target_date or str(date.today() - timedelta(days=1))
    output_path = output_path or f"reports/report_{target_date}.pdf"

    conn = db.get_connection(db_path)

    daily_raw = fetch_predictions_with_results(conn, start_date=target_date, end_date=target_date)
    daily_df = compute_grades(daily_raw)
    n_pendientes = count_ungraded(conn, target_date)

    cumulative_raw = fetch_predictions_with_results(conn)
    cumulative_df = compute_grades(cumulative_raw)

    build_pdf(output_path, target_date, daily_df, cumulative_df, n_pendientes)
    conn.close()

    print(f"Reporte generado: {output_path}")
    if daily_df.empty:
        print(f"  (sin partidos calificables para {target_date} -- revisa que ya "
              f"corriste pipeline.py/backfill.py para traer los resultados de esa fecha)")
    else:
        s = summarize(daily_df)
        print(f"  {target_date}: {s['win_hits']}/{s['n_games']} aciertos de ganador "
              f"({s['win_pct']:.1f}%), error O/U promedio {s['mae_runs']:.2f} carreras")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reporte PDF: predicciones vs resultado real")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD a calificar; por defecto, ayer")
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--output", default=None, help="Ruta del PDF de salida")
    args = parser.parse_args()
    run(args.date, args.db_path, args.output)