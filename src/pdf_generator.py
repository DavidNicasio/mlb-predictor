"""
pdf_generator.py
Módulo de generación de reportes en PDF (Predicciones del día y Calificación histórica/Report Card)
utilizando ReportLab con diseño moderno, nombres completos de equipos, logos PNG,
especificación explícita de Over/Under, sugerencia de Hándicap (-1.5 / +1.5),
información de Clima/Viento y proyecciones para las Primeras 5 Entradas (F5).
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
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

LOGOS_DIR = Path("assets/logos")


def _team_html(team_name: str, team_abbr: str | None = None, team_id: int | None = None) -> str:
    """Retorna código HTML para mostrar el logo PNG (si existe) seguido del nombre completo."""
    img_path = None
    if team_abbr and (LOGOS_DIR / f"{team_abbr}.png").exists():
        img_path = LOGOS_DIR / f"{team_abbr}.png"
    elif team_id and (LOGOS_DIR / f"{team_id}.png").exists():
        img_path = LOGOS_DIR / f"{team_id}.png"

    clean_path = str(img_path).replace("\\", "/") if img_path else None
    if clean_path:
        return f'<img src="{clean_path}" width="15" height="15" valign="middle"/> &nbsp;<b>{team_name}</b>'
    return f"<b>{team_name}</b>"


def _ou_html(total_runs: float) -> str:
    """Formatea la predicción Over/Under con etiqueta y color explícito."""
    if total_runs >= 8.75:
        return f"<font color='#16a34a'><b>OVER</b></font><br/><b>{total_runs:.1f} carreras</b>"
    elif total_runs <= 8.25:
        return f"<font color='#dc2626'><b>UNDER</b></font><br/><b>{total_runs:.1f} carreras</b>"
    else:
        return f"<font color='#0284c7'><b>LÍNEA</b></font><br/><b>{total_runs:.1f} carreras</b>"


def _handicap_suggestion(
    home_name: str, home_abbr: str | None, home_id: int | None,
    away_name: str, away_abbr: str | None, away_id: int | None,
    home_proba: float
) -> str:
    """La sugerencia de hándicap se expresa siempre sobre el equipo FAVORITO a ganar."""
    if home_proba >= 0.50:
        fav_html = _team_html(home_name, home_abbr, home_id)
        hcap_str = "-1.5" if home_proba >= 0.58 else "-0.5 (ML)"
    else:
        fav_html = _team_html(away_name, away_abbr, away_id)
        away_proba = 1.0 - home_proba
        hcap_str = "-1.5" if away_proba >= 0.58 else "-0.5 (ML)"

    color_hex = "#16a34a" if "-1.5" in hcap_str else "#0284c7"
    return f"{fav_html} <font color='{color_hex}'><b>{hcap_str}</b></font>"


def _weather_html(temp: int | None, wind: str | None, condition: str | None) -> str:
    """Formatea la información metereológica oficial."""
    parts = []
    if temp:
        parts.append(f"<b>{temp}°F</b>")
    if wind:
        w_clean = wind.replace("In From", "In").replace("Out To", "Out")
        parts.append(f"Viento: <b>{w_clean}</b>")
    elif condition:
        parts.append(f"<b>{condition}</b>")

    if not parts:
        return "<font color='#94a3b8'>Domo / Normal</font>"
    return "<br/>".join(parts)


def _f5_html(
    f5_runs: float, f5_home_proba: float,
    home_name: str, home_abbr: str | None, home_id: int | None,
    away_name: str, away_abbr: str | None, away_id: int | None
) -> str:
    """Formatea la recomendación de Primeras 5 Entradas (F5)."""
    fav_name = home_name if f5_home_proba >= 0.50 else away_name
    fav_abbr = home_abbr if f5_home_proba >= 0.50 else away_abbr
    fav_id = home_id if f5_home_proba >= 0.50 else away_id

    fav_html = _team_html(fav_name, fav_abbr, fav_id)
    return f"O/U: <b>{f5_runs:.1f}</b><br/>F5: {fav_html}"


def _get_custom_styles():
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=4,
    )

    cell_style = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1e293b"),
        alignment=0,
    )

    cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=cell_style,
        alignment=1,
    )

    cell_header = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1,
    )

    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "h2": h2_style,
        "cell": cell_style,
        "cell_center": cell_center,
        "header": cell_header,
        "normal": styles["Normal"],
    }


def build_daily_predictions_pdf(
    output_path: str,
    target_date: str,
    predictions_df: pd.DataFrame,
) -> str:
    """Genera un PDF visual con las predicciones de la fecha especificada."""
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

    story.append(Paragraph("MLB Predictor - Reporte de Predicciones Diarias", st["title"]))
    story.append(
        Paragraph(
            f"Fecha de partidos: <b>{target_date}</b> | Generado el: {date.today().isoformat()} | Total partidos: {len(predictions_df)}",
            st["subtitle"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#0284c7"),
            spaceBefore=0,
            spaceAfter=10,
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

    headers = [
        Paragraph("Enfrentamiento<br/>(V @ L)", st["header"]),
        Paragraph("Predicción Favorito", st["header"]),
        Paragraph("Proyección O/U<br/>(Línea)", st["header"]),
        Paragraph("Sugerencia Hándicap<br/>(Run Line)", st["header"]),
        Paragraph("Primeras 5 Entradas<br/>(F5)", st["header"]),
        Paragraph("Clima / Viento", st["header"]),
        Paragraph("Abridor Local", st["header"]),
        Paragraph("Abridor Visitante", st["header"]),
    ]
    rows = [headers]

    df_sorted = predictions_df.copy()
    if "game_date_utc" in df_sorted.columns and df_sorted["game_date_utc"].notna().any():
        df_sorted = df_sorted.sort_values("game_date_utc", ascending=True)
    else:
        df_sorted["_confianza"] = (df_sorted["home_win_proba"] - 0.5).abs()
        df_sorted = df_sorted.sort_values("_confianza", ascending=False)

    for _, r in df_sorted.iterrows():
        proba = float(r["home_win_proba"])
        fav_name = r["home_name"] if proba >= 0.5 else r["away_name"]
        fav_abbr = r["home_abbr"] if proba >= 0.5 else r["away_abbr"]
        fav_id = r.get("home_team_id") if proba >= 0.5 else r.get("away_team_id")
        fav_proba = proba if proba >= 0.5 else 1.0 - proba

        home_html = _team_html(r["home_name"], r.get("home_abbr"), r.get("home_team_id"))
        away_html = _team_html(r["away_name"], r.get("away_abbr"), r.get("away_team_id"))
        matchup_html = f"{away_html}<br/><font color='#64748b'><b>@</b></font> {home_html}"

        fav_team_html = _team_html(fav_name, fav_abbr, fav_id)
        fav_cell_html = f"{fav_team_html}<br/><font color='#0284c7'><b>{fav_proba:.1%} victoria</b></font>"

        ou_label = _ou_html(float(r["total_runs_pred"]))

        hcap_html = _handicap_suggestion(
            r["home_name"], r.get("home_abbr"), r.get("home_team_id"),
            r["away_name"], r.get("away_abbr"), r.get("away_team_id"),
            proba
        )

        f5_runs = float(r.get("f5_total_runs_pred", float(r["total_runs_pred"]) * 0.55))
        f5_home_proba = float(r.get("f5_home_win_proba", proba))
        f5_cell = _f5_html(
            f5_runs, f5_home_proba,
            r["home_name"], r.get("home_abbr"), r.get("home_team_id"),
            r["away_name"], r.get("away_abbr"), r.get("away_team_id")
        )

        weather_cell = _weather_html(
            r.get("weather_temp"), r.get("weather_wind"), r.get("weather_condition")
        )

        if pd.notna(r.get("home_abridor_id")) and r.get("home_abridor_id"):
            h_fip = f"{r.get('home_fip'):.2f}" if pd.notna(r.get("home_fip")) else "N/A"
            h_starts = int(r.get("home_n_starts") or 0)
            h_pitcher = f"Brazo: {r.get('home_abridor_throws') or '?'}<br/>FIP: <b>{h_fip}</b> ({h_starts} ap.)"
        else:
            h_pitcher = "<font color='#94a3b8'><i>Sin confirmar</i></font>"

        if pd.notna(r.get("away_abridor_id")) and r.get("away_abridor_id"):
            a_fip = f"{r.get('away_fip'):.2f}" if pd.notna(r.get("away_fip")) else "N/A"
            a_starts = int(r.get("away_n_starts") or 0)
            a_pitcher = f"Brazo: {r.get('away_abridor_throws') or '?'}<br/>FIP: <b>{a_fip}</b> ({a_starts} ap.)"
        else:
            a_pitcher = "<font color='#94a3b8'><i>Sin confirmar</i></font>"

        rows.append(
            [
                Paragraph(matchup_html, st["cell"]),
                Paragraph(fav_cell_html, st["cell"]),
                Paragraph(ou_label, st["cell_center"]),
                Paragraph(hcap_html, st["cell"]),
                Paragraph(f5_cell, st["cell"]),
                Paragraph(weather_cell, st["cell_center"]),
                Paragraph(h_pitcher, st["cell"]),
                Paragraph(a_pitcher, st["cell"]),
            ]
        )

    # Anchos de columnas (total 10.2 pulgadas = 734.4 pt)
    col_widths = [
        1.7 * inch,
        1.5 * inch,
        1.1 * inch,
        1.4 * inch,
        1.3 * inch,
        1.0 * inch,
        1.1 * inch,
        1.1 * inch,
    ]

    table = Table(rows, colWidths=col_widths, repeatRows=1)
    t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
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
    """Genera el PDF de calificación histórica (Report Card) incluyendo partidos finalizados y pendientes."""
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

    story.append(Paragraph("MLB Predictor - Evaluación de Resultados (Report Card)", st["title"]))
    story.append(
        Paragraph(
            f"Fecha evaluada: <b>{target_date}</b> | Reporte generado el: {date.today().isoformat()}",
            st["subtitle"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#059669"),
            spaceBefore=0,
            spaceAfter=8,
        )
    )

    # --- Sección 1: Cuadro del Día ---
    story.append(Paragraph(f"Resumen del Día: {target_date}", st["h2"]))

    finalized_daily = daily_df[daily_df["is_final"] == True] if not daily_df.empty and "is_final" in daily_df.columns else daily_df
    from report_card import summarize

    daily_stats = summarize(finalized_daily)
    _add_summary_block(story, daily_stats, st, n_pendientes)
    story.append(Spacer(1, 8))

    if not daily_df.empty:
        story.append(_build_results_table(daily_df, st))

    # --- Sección 2: Acumulado Histórico ---
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceBefore=4, spaceAfter=6))
    story.append(Paragraph("Acumulado Histórico de Predicciones Finalizadas", st["h2"]))

    cum_stats = summarize(cumulative_df)
    if cum_stats.get("n_games", 0) > 0:
        date_min = cumulative_df["game_date"].min()
        date_max = cumulative_df["game_date"].max()
        story.append(
            Paragraph(
                f"Rango evaluado: <b>{date_min}</b> a <b>{date_max}</b>",
                st["subtitle"],
            )
        )
        story.append(Spacer(1, 4))

    _add_summary_block(story, cum_stats, st, 0)

    doc.build(story)
    return output_path


def _add_summary_block(story: list, stats: dict, st: dict, n_pendientes: int = 0) -> None:
    if stats.get("n_games", 0) == 0:
        msg = "<i>No hay partidos finalizados calificables en este rango todavía.</i>"
        if n_pendientes > 0:
            msg += f" <font color='#d97706'>({n_pendientes} partido(s) pendientes por jugar o terminar).</font>"
        story.append(Paragraph(msg, st["normal"]))
        return

    bias = stats["bias_runs"]
    if bias > 0.25:
        sesgo_txt = "<font color='#dc2626'><b>(El modelo proyectó de menos / anotaron más)</b></font>"
    elif bias < -0.25:
        sesgo_txt = "<font color='#d97706'><b>(El modelo proyectó de más / anotaron menos)</b></font>"
    else:
        sesgo_txt = "<font color='#16a34a'><b>(Sin sesgo significativo)</b></font>"

    win_pct = stats["win_pct"]
    win_color = "#16a34a" if win_pct >= 55.0 else ("#0284c7" if win_pct >= 50.0 else "#dc2626")

    pending_html = f" &nbsp;|&nbsp; <font color='#d97706'><b>{n_pendientes} Pendientes</b></font>" if n_pendientes > 0 else ""

    summary_html = f"""
    Partidos finalizados evaluados: <b>{stats['n_games']}</b>{pending_html} &nbsp;|&nbsp;
    Aciertos de Ganador: <font color='{win_color}'><b>{stats['win_hits']}/{stats['n_games']} ({win_pct:.1f}%)</b></font><br/>
    Error prom. Total Carreras (MAE): <b>{stats['mae_runs']:.2f}</b> &nbsp;|&nbsp;
    Sesgo promedio: <b>{stats['bias_runs']:+.2f} carreras</b> {sesgo_txt}<br/>
    Desglose Over/Under: <b>{stats['n_over']} SOBRE</b> &nbsp;|&nbsp; <b>{stats['n_under']} BAJO</b> &nbsp;|&nbsp; <b>{stats['n_igual']} IGUAL</b>
    """
    story.append(Paragraph(summary_html, st["cell"]))


def _build_results_table(df: pd.DataFrame, st: dict) -> Table:
    headers = [
        Paragraph("Enfrentamiento (V @ L)", st["header"]),
        Paragraph("Predicción Favorito", st["header"]),
        Paragraph("Sugerencia Hándicap", st["header"]),
        Paragraph("Marcador Real", st["header"]),
        Paragraph("Ganador Real", st["header"]),
        Paragraph("¿Acierto?", st["header"]),
        Paragraph("Proyección Over / Under", st["header"]),
        Paragraph("Total Real", st["header"]),
        Paragraph("Resultado Real O/U", st["header"]),
    ]
    rows = [headers]

    if "game_date_utc" in df.columns and df["game_date_utc"].notna().any():
        ordered = df.sort_values("game_date_utc", ascending=True)
    else:
        ordered = df.sort_values("game_date", ascending=True)

    for _, r in ordered.iterrows():
        is_final = bool(r.get("is_final", True))

        away_html = _team_html(r["away_name"], r.get("away_abbr"), r.get("away_team_id"))
        home_html = _team_html(r["home_name"], r.get("home_abbr"), r.get("home_team_id"))
        matchup = f"{away_html}<br/><font color='#64748b'><b>@</b></font> {home_html}"

        pred_fav_name = r["predicted_winner"]
        pred_fav_abbr = r.get("predicted_winner_abbr")
        pred_fav_html = _team_html(pred_fav_name, pred_fav_abbr)
        pred_cell = f"{pred_fav_html}<br/><font color='#0284c7'><b>{r['favorito_proba']:.0%}</b></font>"

        hcap_html = _handicap_suggestion(
            r["home_name"], r.get("home_abbr"), r.get("home_team_id"),
            r["away_name"], r.get("away_abbr"), r.get("away_team_id"),
            float(r["home_win_proba"])
        )

        ou_pred = _ou_html(float(r["total_runs_pred"]))

        if is_final:
            score = f"<b>{int(r['away_score'])}-{int(r['home_score'])}</b>"
            actual_fav_html = _team_html(r["actual_winner"], r.get("actual_winner_abbr"))
            hit = bool(r["win_hit"])
            hit_html = "<font color='#16a34a'><b>SÍ</b></font>" if hit else "<font color='#dc2626'><b>NO</b></font>"
            total_actual = f"<b>{int(r['actual_total'])}</b>"

            diff = float(r["ou_diff"])
            label = r["ou_label"]
            if label == "SOBRE":
                ou_comp = f"<font color='#16a34a'><b>SOBRE ({diff:+.1f})</b></font>"
            elif label == "BAJO":
                ou_comp = f"<font color='#dc2626'><b>BAJO ({diff:+.1f})</b></font>"
            else:
                ou_comp = f"<font color='#0284c7'><b>IGUAL ({diff:+.1f})</b></font>"
        else:
            status_text = r.get("status") or "Scheduled"
            if pd.notna(r.get("away_score")) and pd.notna(r.get("home_score")):
                score = f"{int(r['away_score'])}-{int(r['home_score'])}<br/><font color='#d97706'>({status_text})</font>"
            else:
                score = f"<font color='#d97706'>PENDIENTE</font>"
            actual_fav_html = "<font color='#94a3b8'>-</font>"
            hit_html = "<font color='#d97706'><b>PENDIENTE</b></font>"
            total_actual = "<font color='#94a3b8'>-</font>"
            ou_comp = "<font color='#94a3b8'>PENDIENTE</font>"

        rows.append(
            [
                Paragraph(matchup, st["cell"]),
                Paragraph(pred_cell, st["cell"]),
                Paragraph(hcap_html, st["cell"]),
                Paragraph(score, st["cell_center"]),
                Paragraph(actual_fav_html, st["cell"]),
                Paragraph(hit_html, st["cell_center"]),
                Paragraph(ou_pred, st["cell_center"]),
                Paragraph(total_actual, st["cell_center"]),
                Paragraph(ou_comp, st["cell_center"]),
            ]
        )

    # Ancho total: 10.2 pulgadas = 734.4 pt
    col_widths = [
        1.7 * inch,
        1.4 * inch,
        1.4 * inch,
        0.8 * inch,
        1.4 * inch,
        0.6 * inch,
        1.2 * inch,
        0.6 * inch,
        1.1 * inch,
    ]

    table = Table(rows, colWidths=col_widths, repeatRows=1)
    t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [colors.white, colors.HexColor("#f8fafc")],
        ),
    ]
    table.setStyle(TableStyle(t_style))
    return table
