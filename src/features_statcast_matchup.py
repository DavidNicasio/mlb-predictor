"""
features_statcast_matchup.py
Fase 6: Mapeo de arsenal de pitcheo Statcast (pitch_type) del abridor vs bateadores titulares.
"""

from __future__ import annotations

import sqlite3
import pandas as pd


def get_pitcher_arsenal_mix(conn: sqlite3.Connection, pitcher_id: int, as_of_date: str, days: int = 60) -> dict[str, float]:
    """Calcula la proporción de tipos de pitcheo lanzados por el abridor en los últimos N días."""
    start_dt = str(pd.to_datetime(as_of_date) - pd.Timedelta(days=days))[:10]

    rows = conn.execute("""
        SELECT pitch_type, COUNT(*) AS n
        FROM statcast_batted_balls
        WHERE pitcher_id = ? AND game_date >= ? AND game_date < ?
          AND pitch_type IS NOT NULL
        GROUP BY pitch_type
    """, (pitcher_id, start_dt, as_of_date)).fetchall()

    total = sum(r[1] for r in rows)
    if not total:
        # Mezcla por defecto de la liga si no hay datos de Statcast
        return {"FF": 0.45, "SL": 0.25, "CH": 0.15, "SI": 0.15}

    return {r[0]: round(r[1] / total, 4) for r in rows}


def compute_lineup_vs_pitch_mix(
    conn: sqlite3.Connection, lineup_player_ids: list[int], pitcher_id: int, as_of_date: str
) -> float:
    """Calcula un índice de ventaja situacional de la alineación contra la mezcla de pitcheo del abridor."""
    arsenal = get_pitcher_arsenal_mix(conn, pitcher_id, as_of_date)
    # Por omisión, un índice neutro es 1.0 (ajustable con Statcast a futuro)
    slider_ratio = arsenal.get("SL", 0.25)
    fastball_ratio = arsenal.get("FF", 0.45)

    # Índice de ventaja del arsenal (ej. abridores con alto slider vs alineaciones con alta tasa de abanico a slider)
    arsenal_effect = 1.0 + (fastball_ratio * 0.05) - (slider_ratio * 0.03)
    return round(arsenal_effect, 4)
