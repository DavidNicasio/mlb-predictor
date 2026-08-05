"""
features_pitching.py
Rolling del abridor probable de un partido: FIP/xFIP en sus últimas N
aperturas (no por ventana de días -- un abridor lanza cada 5-6 días, así
que "últimas N salidas" es el marco correcto, no un rango de fechas fijo),
más descanso y carga de la salida anterior.
"""

from __future__ import annotations

from datetime import date

import extract_statcast
import metrics

DEFAULT_N_STARTS = 8

# Fallback si todavía no hay league_constants calculadas para la temporada
# en curso (ej. primeros días de abril, antes de tener suficientes Final).
FALLBACK_FIP_CONSTANT = 3.10
FALLBACK_LG_HR_FB = 0.12


def get_pitcher_hand(conn, pitcher_id: int) -> str | None:
    """Lee de la tabla `players`, NO de boxscore_pitching.throws -- ese
    campo nunca se pudo poblar porque el boxscore no trae biografía del
    jugador (confirmado contra la API real). Requiere haber corrido
    extract_players.py al menos una vez."""
    row = conn.execute(
        "SELECT throws FROM players WHERE player_id = ?", (pitcher_id,)
    ).fetchone()
    return row[0] if row else None


def _league_constants_with_fallback(conn, season: int) -> dict:
    row = conn.execute(
        "SELECT fip_constant, lg_hr_fb FROM league_constants WHERE season=?", (season,)
    ).fetchone()
    if row and row[0] is not None:
        return {"fip_constant": row[0], "lg_hr_fb": row[1] or FALLBACK_LG_HR_FB}

    row_prev = conn.execute(
        "SELECT fip_constant, lg_hr_fb FROM league_constants WHERE season=?", (season - 1,)
    ).fetchone()
    if row_prev and row_prev[0] is not None:
        return {"fip_constant": row_prev[0], "lg_hr_fb": row_prev[1] or FALLBACK_LG_HR_FB}

    return {"fip_constant": FALLBACK_FIP_CONSTANT, "lg_hr_fb": FALLBACK_LG_HR_FB}


def last_n_starts(conn, pitcher_id: int, as_of_date: str, n: int = DEFAULT_N_STARTS) -> list[dict]:
    rows = conn.execute(
        """SELECT bp.game_pk, g.game_date, bp.outs, bp.hr, bp.bb, bp.hbp, bp.so, bp.pitches_thrown
           FROM boxscore_pitching bp
           JOIN games g ON g.game_pk = bp.game_pk
           WHERE bp.player_id = ? AND bp.is_starter = 1 AND g.game_date < ?
           ORDER BY g.game_date DESC LIMIT ?""",
        (pitcher_id, as_of_date, n),
    ).fetchall()
    cols = ["game_pk", "game_date", "outs", "hr", "bb", "hbp", "so", "pitches_thrown"]
    return [dict(zip(cols, r)) for r in rows]


def _fly_balls_in_starts(conn, pitcher_id: int, game_pks: list[int]) -> int:
    if not game_pks:
        return 0
    placeholders = ",".join("?" * len(game_pks))
    rows = conn.execute(
        f"""SELECT launch_angle FROM statcast_batted_balls
            WHERE pitcher_id = ? AND game_pk IN ({placeholders})
              AND launch_angle IS NOT NULL""",
        (pitcher_id, *game_pks),
    ).fetchall()
    return sum(1 for (la,) in rows if extract_statcast.classify_batted_ball(la) == "FB")


def rolling_starter_fip_xfip(conn, pitcher_id: int, as_of_date: str, season: int,
                              n_starts: int = DEFAULT_N_STARTS) -> dict:
    starts = last_n_starts(conn, pitcher_id, as_of_date, n_starts)
    if not starts:
        return {"fip": None, "xfip": None, "outs": 0, "n_starts": 0, "avg_ip_per_start": None}

    outs = sum(s["outs"] or 0 for s in starts)
    hr = sum(s["hr"] or 0 for s in starts)
    bb = sum(s["bb"] or 0 for s in starts)
    hbp = sum(s["hbp"] or 0 for s in starts)
    so = sum(s["so"] or 0 for s in starts)

    lc = _league_constants_with_fallback(conn, season)
    fip = metrics.fip(hr, bb, hbp, so, outs, lc["fip_constant"])

    fb = _fly_balls_in_starts(conn, pitcher_id, [s["game_pk"] for s in starts])
    xfip = metrics.xfip(fb, lc["lg_hr_fb"], bb, hbp, so, outs, lc["fip_constant"])

    return {
        "fip": fip, "xfip": xfip, "outs": outs, "n_starts": len(starts),
        "avg_ip_per_start": round(metrics.innings_from_outs(outs) / len(starts), 2),
    }


def rest_and_last_workload(conn, pitcher_id: int, as_of_date: str) -> dict:
    """Días de descanso y pitch count de la última apertura (cualquiera,
    no solo dentro de las últimas N usadas para FIP)."""
    last = last_n_starts(conn, pitcher_id, as_of_date, n=1)
    if not last:
        return {"abridor_dias_descanso": None, "last_start_pitches": None, "last_start_outs": None}

    last_date = date.fromisoformat(last[0]["game_date"])
    days_rest = (date.fromisoformat(as_of_date) - last_date).days
    return {
        "abridor_dias_descanso": days_rest,
        "last_start_pitches": last[0]["pitches_thrown"],
        "last_start_outs": last[0]["outs"],
    }


def build_pitching_features(conn, pitcher_id: int | None, as_of_date: str, season: int,
                              n_starts: int = DEFAULT_N_STARTS) -> dict:
    if pitcher_id is None:
        return {"abridor_id": None, "abridor_throws": None, "fip": None, "xfip": None,
                "n_starts": 0, "avg_ip_per_start": None, "abridor_dias_descanso": None,
                "last_start_pitches": None}

    hand = get_pitcher_hand(conn, pitcher_id)
    roll = rolling_starter_fip_xfip(conn, pitcher_id, as_of_date, season, n_starts)
    rest = rest_and_last_workload(conn, pitcher_id, as_of_date)

    return {
        "abridor_id": pitcher_id,
        "abridor_throws": hand,
        "fip": roll["fip"],
        "xfip": roll["xfip"],
        "n_starts": roll["n_starts"],
        "avg_ip_per_start": roll["avg_ip_per_start"],
        "abridor_dias_descanso": rest["abridor_dias_descanso"],
        "last_start_pitches": rest["last_start_pitches"],
    }
