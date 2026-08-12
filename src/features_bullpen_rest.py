"""
features_bullpen_rest.py
Fase 6: Medición de la carga de trabajo y fatiga del bullpen (relevistas).
Suma los lanzamientos tirados por el bullpen en los últimos 1, 2 y 3 días para determinar
si los relevistas principales llegan indispuestos por cansancio.
"""

from __future__ import annotations

import sqlite3
import pandas as pd


def compute_bullpen_fatigue(conn: sqlite3.Connection, team_id: int, as_of_date: str) -> dict:
    """Calcula el índice de fatiga del bullpen para un equipo en los últimos 3 días antes de as_of_date."""
    dt_target = pd.to_datetime(as_of_date)
    dt_1d = str(dt_target - pd.Timedelta(days=1))[:10]
    dt_3d = str(dt_target - pd.Timedelta(days=3))[:10]

    rows = conn.execute("""
        SELECT g.game_date, SUM(p.pitches_thrown) AS pitches, SUM(p.outs) AS outs, COUNT(DISTINCT p.player_id) AS n_rel
        FROM boxscore_pitching p
        JOIN games g ON g.game_pk = p.game_pk
        WHERE p.team_id = ? AND p.is_starter = 0
          AND g.game_date >= ? AND g.game_date < ? AND g.status = 'Final'
        GROUP BY g.game_date
    """, (team_id, dt_3d, as_of_date)).fetchall()

    pitches_1d, pitches_2d, pitches_3d = 0, 0, 0
    outs_1d, outs_2d, outs_3d = 0, 0, 0

    for r_date, p_count, o_count, _ in rows:
        p_val = p_count or 0
        o_val = o_count or 0
        days_diff = (dt_target - pd.to_datetime(r_date)).days

        if days_diff == 1:
            pitches_1d += p_val
            outs_1d += o_val
        elif days_diff == 2:
            pitches_2d += p_val
            outs_2d += o_val
        elif days_diff == 3:
            pitches_3d += p_val
            outs_3d += o_val

    # Ponderar la recarga de trabajo reciente (1d pesa 1.5x, 2d 1.0x, 3d 0.5x)
    weighted_pitches = (pitches_1d * 1.5) + (pitches_2d * 1.0) + (pitches_3d * 0.5)
    fatigue_index = round(weighted_pitches / 100.0, 4)

    return {
        "team_id": team_id,
        "pitches_1d": pitches_1d,
        "pitches_2d": pitches_2d,
        "pitches_3d": pitches_3d,
        "fatigue_index": fatigue_index,
    }
