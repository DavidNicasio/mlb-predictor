"""
pdf_generator.py
Módulo de generación de reportes en PDF (Predicciones del día y Calificación histórica/Report Card)
utilizando ReportLab con diseño moderno, textos auto-ajustables (Paragraph) y tablas limpias.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _get_custom_styles():
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e293b"),
        alignment=0, # Izquierda
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#64748b"),
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=6,
    )

    cell_style = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1e293b"),
        alignment=0,
    )

    cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=cell_style,
        fontName="Helvetica-Bold",
    )

    cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=cell_style,
        alignment=1,
    )

    cell_center_bold = ParagraphStyle(
        "TableCellCenterBold",
        parent=cell_bold,
        alignment=1,
    )

    cell_header = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.white,
        alignment=1,
    )

    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "h2": h2_style,
        "cell": cell_style,
        "cell_bold": cell_bold,
        "cell_center": cell_center,
        "cell_center_bold": cell_center_bold,
        "header": cell_header,
        "normal": styles["Normal"],
    }


def build_daily_predictions_pdf(
    output_path: str,
    target_date: str,
    predictions_df: pd.DataFrame,
) -> str:
    """Genera un PDF visual moderno con las predicciones del día (partidos programados,
    en progreso o finalizados).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(letter),
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
        leftMargin=0.4 * inch,
        rightMargin=0.4 * inch,
    )

    st = _get_custom_styles()
    story = []

    # Encabezado principal
    story.append(Paragraph(f"MLB Predictor - Predicciones del Día", st["title"]))
    story.append(
        Paragraph(
            f"Fecha de partidos: <b>{target_date}</b> | Reporte generado el {date.today().isoformat()} | Partidos programados: {len(predictions_df)}",
            st["subtitle"],
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#0284c7"),
            spaceBefore=0,
            spaceAfter=12,
        )
    )

    if predictions_df.empty:
        story.append(
            Paragraph(
                f"No hay partidos programados o con datos para la fecha <b>{target_date}</b>.",
                st["normal"],
            )
        )
        doc.build(story)
        return output_path

    # Formatear la tabla de predicciones
    # Columnas: Partido, Favorito (Prob), Proyección O/U, Abridor Local, Abridor Visitante
    table_data = [
        [
            Paragraph("Enfrentamiento<br/>(Visitante @ Local)", st["header"]),
            Paragraph("Favorito y Probabilidad", st["header"]),
            Paragraph("Proyección O/U<br/>(Carreras)", st["header"]),
            Paragraph("Abridor Local (Lanzador / FIP / Ap.)", st["header"]),
            Paragraph("Abridor Visitante (Lanzador / FIP / Ap.)", st["header"]),
        ]
    ]

    # Ordenar por nivel de confianza del modelo
    df_sorted = predictions_df.copy()
    df_sorted["_confianza"] = (df_sorted["home_win_proba"] - 0.5).abs()
    df_sorted = df_sorted.sort_values("_confianza", ascending=False)

    for _, r in df_sorted.iterrows():
        proba = float(r["home_win_proba"])
        fav_team = r["home_name"] if proba >= 0.5 else r["away_name"]
        fav_proba = proba if proba >= 0.5 else 1.0 - proba

        # Enfrentamiento
        matchup_html = f"<b>{r['away_name']}</b><br/>@ <b>{r['home_name']}</b>"

        # Favorito
        fav_html = f"<b>{fav_team}</b><br/><font color='#0369a1'><b>{fav_proba:.1%}</b></font> (Local: {proba:.1%})"

        # Over/Under
        ou_html = f"<font size='10'><b>{r['total_runs_pred']:.1f}</b></font> carreras"

        # Abridor Local
        if pd.notna(r.get("home_abridor_id")) and r.get("home_abridor_id"):
            h_fip = f"{r.get('home_fip'):.2f}" if pd.notna(r.get("home_fip")) else "N/A"
            h_starts = int(r.get("home_n_starts") or 0)
            h_pitcher = f"Brazo: {r.get('home_abridor_throws') or '?'}<br/>FIP: <b>{h_fip}</b> ({h_starts} ap.)"
        else:
            h_pitcher = "<font color='#64748b'><i>Sin confirmar</i></font>"

        # Abridor Visitante
        if pd.notna(r.get("away_abridor_id")) and r.get("away_abridor_id"):
            a_fip = f"{r.get('away_fip'):.2f}" if pd.notna(r.get("away_fip")) else "N/A"
            a_starts = int(r.get("away_n_starts") or 0)
            a_pitcher = f"Brazo: {r.get('away_abridor_throws') or '?'}<br/>FIP: <b>{a_fip}</b> ({a_starts} ap.)"
        else:
            a_pitcher = "<font color='#64748b'><i>Sin confirmar</i></font>"

        table_data.append(
            [
                Paragraph(matchup_html, st["cell"]),
                Paragraph(fav_html, st["cell_center"]),
                Paragraph(ou_html, st["cell_center"]),
                Paragraph(h_pitcher, st["cell"]),
                Paragraph(a_pitcher, st["cell"]),
            ]
        )

    # Ancho total disponible en landscape letter: 11in - 0.8in = 10.2 in = 734.4 points
    # Distribución de anchos: [2.2 in, 1.8 in, 1.4 in, 2.4 in, 2.4 in] -> sum 10.2 in
    col_widths = [
        2.2 * inch,
        1.8 * inch,
        1.4 * inch,
        2.4 * inch,
        2.4 * inch,
    ]

    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [colors.white, colors.HexColor("#f8fafc")],
        ),
    ]
    table.setStyle(TableStyle(t_style))
    story.append(table)

    doc.build(story)
    return output_path


def build_report_card_pdf(
    output_path: str,
    target_date: str,
    daily_df: pd.DataFrame,
    cumulative_df: pd.DataFrame,
    n_pendientes: int = 0,
) -> str:
    """Genera el PDF de calificación histórica (Report Card) comparando predicciones
    contra los marcadores reales.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(letter),
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
        leftMargin=0.4 * inch,
        rightMargin=0.4 * inch,
    )

    st = _get_custom_styles()
    story = []

    # Encabezado principal
    story.append(Paragraph(f"MLB Predictor - Evaluación de Resultados (Report Card)", st["title"]))
    story.append(
        Paragraph(
            f"Fecha evaluada: <b>{target_date}</b> | Reporte generado el {date.today().isoformat()}",
            st["subtitle"],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#059669"),
            spaceBefore=0,
            spaceAfter=10,
        )
    )

    # --- Sección 1: Cuadro del Día ---
    story.append(Paragraph(f"Resumen del Día: {target_date}", st["h2"]))
    if n_pendientes > 0:
        story.append(
            Paragraph(
                f"<font color='#d97706'><b>Nota:</b> {n_pendientes} partido(s) de esta fecha aún no tienen resultado final registrado.</font>",
                st["normal"],
            )
        )
        story.append(Spacer(1, 4))

    # Resumen numérico del día
    from report_card import summarize

    daily_stats = summarize(daily_df)
    _add_summary_block(story, daily_stats, st)
    story.append(Spacer(1, 10))

    if not daily_df.empty:
        story.append(_build_results_table(daily_df, st))

    # --- Sección 2: Acumulado Histórico ---
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceBefore=6, spaceAfter=8))
    story.append(Paragraph("Acumulado Histórico de Predicciones", st["h2"]))

    cum_stats = summarize(cumulative_df)
    if cum_stats.get("n_games", 0) > 0:
        date_min = cumulative_df["game_date"].min()
        date_max = cumulative_df["game_date"].max()
        story.append(
            Paragraph(
                f"Rango de datos evaluados: <b>{date_min}</b> a <b>{date_max}</b>",
                st["subtitle"],
            )
        )
        story.append(Spacer(1, 4))

    _add_summary_block(story, cum_stats, st)

    doc.build(story)
    return output_path


def _add_summary_block(story: list, stats: dict, st: dict) -> None:
    if stats.get("n_games", 0) == 0:
        story.append(
            Paragraph(
                "<i>No hay partidos finalizados calificables en este rango todavía.</i>",
                st["normal"],
            )
        )
        return

    bias = stats["bias_runs"]
    if bias > 0.25:
        sesgo_txt = "<font color='#dc2626'><b>(El modelo proyectó de menos / anotaron más)</b></font>"
    elif bias < -0.25:
        sesgo_txt = "<font color='#d97706'><b>(El modelo proyectó de más / anotaron menos)</b></font>"
    else:
        sesgo_txt = "<font color='#16a34a'><b>(Sin sesgo significativo)</b></font>"

    win_pct = stats['win_pct']
    win_color = "#16a34a" if win_pct >= 55.0 else ("#0284c7" if win_pct >= 50.0 else "#dc2626")

    summary_html = f"""
    Partidos evaluados: <b>{stats['n_games']}</b> &nbsp;|&nbsp;
    Aciertos de Ganador: <font color='{win_color}'><b>{stats['win_hits']}/{stats['n_games']} ({win_pct:.1f}%)</b></font><br/>
    Error prom. Total Carreras (MAE): <b>{stats['mae_runs']:.2f}</b> &nbsp;|&nbsp;
    Sesgo promedio: <b>{stats['bias_runs']:+.2f} carreras</b> {sesgo_txt}<br/>
    Desglose Over/Under: <b>{stats['n_over']} SOBRE</b> &nbsp;|&nbsp; <b>{stats['n_under']} BAJO</b> &nbsp;|&nbsp; <b>{stats['n_igual']} IGUAL</b>
    """
    story.append(Paragraph(summary_html, st["cell"]))


def _build_results_table(df: pd.DataFrame, st: dict) -> Table:
    headers = [
        Paragraph("Partido", st["header"]),
        Paragraph("Predicción Favorito", st["header"]),
        Paragraph("Marcador<br/>Real (V-L)", st["header"]),
        Paragraph("Ganador Real", st["header"]),
        Paragraph("¿Acierto?", st["header"]),
        Paragraph("Proy.<br/>O/U", st["header"]),
        Paragraph("Total<br/>Real", st["header"]),
        Paragraph("Comparación O/U", st["header"]),
    ]
    rows = [headers]

    ordered = df.sort_values("game_date")
    for _, r in ordered.iterrows():
        hit = bool(r["win_hit"])
        hit_html = "<font color='#16a34a'><b>SÍ</b></font>" if hit else "<font color='#dc2626'><b>NO</b></font>"

        matchup = f"<b>{r['away_abbr']} @ {r['home_abbr']}</b>"
        pred_fav = f"<b>{r['predicted_winner_abbr']}</b> ({r['favorito_proba']:.0%})"
        score = f"{int(r['away_score'])}-{int(r['home_score'])}"
        winner = f"<b>{r['actual_winner_abbr']}</b>"
        ou_pred = f"{r['total_runs_pred']:.1f}"
        total_actual = f"<b>{int(r['actual_total'])}</b>"
        ou_comp = f"{r['ou_label']} ({r['ou_diff']:+.1f})"

        rows.append(
            [
                Paragraph(matchup, st["cell"]),
                Paragraph(pred_fav, st["cell_center"]),
                Paragraph(score, st["cell_center"]),
                Paragraph(winner, st["cell_center"]),
                Paragraph(hit_html, st["cell_center"]),
                Paragraph(ou_pred, st["cell_center"]),
                Paragraph(total_actual, st["cell_center"]),
                Paragraph(ou_comp, st["cell_center"]),
            ]
        )

    # Ancho total: 10.2 in = 734.4 points
    # Distribución: [1.3, 1.6, 1.0, 1.0, 0.9, 0.9, 0.9, 1.6] -> sum = 9.2 in (entra holgadamente)
    col_widths = [
        1.4 * inch,
        1.7 * inch,
        1.1 * inch,
        1.1 * inch,
        0.9 * inch,
        0.9 * inch,
        0.9 * inch,
        1.6 * inch,
    ]

    table = Table(rows, colWidths=col_widths, repeatRows=1)
    t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [colors.white, colors.HexColor("#f8fafc")],
        ),
    ]
    table.setStyle(TableStyle(t_style))
    return table
