"""
features_bullpen.py
FIP del bullpen ponderado por qué tan reciente fue cada apertura de
relevo (un relevista que lanzó ayer pesa más que uno de hace 3 semanas,
porque es más probable que sea el que salga hoy), más un índice de
fatiga basado en innings lanzados en el/los últimos días.

Simplificación consciente: no identificamos "el cerrador" como rol
individual (necesitaría rastrear saves, que no capturamos todavía).
En su lugar usamos un índice de fatiga general del bullpen (innings
lanzados en los últimos 1-3 días) como proxy de qué tan fresco está.
"""

from __future__ import annotations

from datetime import date, timedelta

import features_pitching
import metrics

DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_HALF_LIFE_DAYS = 10  # una apertura de hace 10 días pesa la mitad que una de hoy
BULLPEN_STABILIZATION_K_OUTS = 40  # ~13 IP equivalentes; menos que eso, se jala hacia la liga


def _recency_weight(days_ago: int, half_life_days: float) -> float:
    return 0.5 ** (days_ago / half_life_days)


def bullpen_appearances(conn, team_id: int, as_of_date: str,
                          lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> list[dict]:
    start_d = date.fromisoformat(as_of_date) - timedelta(days=lookback_days)
    rows = conn.execute(
        """SELECT g.game_date, bp.outs, bp.hr, bp.bb, bp.hbp, bp.so
           FROM boxscore_pitching bp
           JOIN games g ON g.game_pk = bp.game_pk
           WHERE bp.team_id = ? AND bp.is_starter = 0
                 AND g.game_date >= ? AND g.game_date < ?""",
        (team_id, str(start_d), as_of_date),
    ).fetchall()
    cols = ["game_date", "outs", "hr", "bb", "hbp", "so"]
    return [dict(zip(cols, r)) for r in rows]


def rolling_bullpen_fip(conn, team_id: int, as_of_date: str, season: int,
                          lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                          half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> dict:
    appearances = bullpen_appearances(conn, team_id, as_of_date, lookback_days)
    if not appearances:
        return {"fip_ponderado": None, "fip_ponderado_shrunk": None,
                "outs_totales": 0, "apariciones": 0}

    as_of = date.fromisoformat(as_of_date)
    w_outs = w_hr = w_bb = w_hbp = w_so = 0.0
    for a in appearances:
        days_ago = (as_of - date.fromisoformat(a["game_date"])).days
        w = _recency_weight(max(days_ago, 0), half_life_days)
        w_outs += (a["outs"] or 0) * w
        w_hr += (a["hr"] or 0) * w
        w_bb += (a["bb"] or 0) * w
        w_hbp += (a["hbp"] or 0) * w
        w_so += (a["so"] or 0) * w

    lc = features_pitching._league_constants_with_fallback(conn, season)
    fip_ponderado = metrics.fip(w_hr, w_bb, w_hbp, w_so, w_outs, lc["fip_constant"])

    # Shrinkage: con pocas entradas ponderadas, jala hacia el ERA de liga
    # de esa temporada (proxy razonable del FIP promedio, por construcción).
    league_row = conn.execute(
        "SELECT league_era FROM league_constants WHERE season=?", (season,)
    ).fetchone()
    league_anchor = league_row[0] if league_row and league_row[0] is not None else 4.20
    fip_shrunk = metrics.shrink_rate(fip_ponderado, w_outs, league_anchor,
                                       BULLPEN_STABILIZATION_K_OUTS)

    return {
        "fip_ponderado": fip_ponderado,
        "fip_ponderado_shrunk": fip_shrunk,
        "outs_totales": round(sum(a["outs"] or 0 for a in appearances), 1),
        "apariciones": len(appearances),
    }


def fatigue_index(conn, team_id: int, as_of_date: str) -> dict:
    """Innings de bullpen lanzados en el/los últimos días -- entre más
    alto, más 'quemado' está el pen para el partido de hoy."""
    result = {}
    for days in (1, 2, 3):
        appearances = bullpen_appearances(conn, team_id, as_of_date, lookback_days=days)
        outs = sum(a["outs"] or 0 for a in appearances)
        result[f"bullpen_outs_last_{days}d"] = outs
    return result


def build_bullpen_features(conn, team_id: int, as_of_date: str, season: int,
                             lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                             half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> dict:
    roll = rolling_bullpen_fip(conn, team_id, as_of_date, season, lookback_days, half_life_days)
    fatigue = fatigue_index(conn, team_id, as_of_date)
    return {**roll, **fatigue}
