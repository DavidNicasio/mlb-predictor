"""
features_lineup.py
Cálculo de la proyección ofensiva ponderada por alineación real (Confirmed Lineup)
y wOBA por bateador individual con shrinkage bayesiano ($k=40$ PA).
"""

from __future__ import annotations

import sqlite3
import pandas as pd
import metrics


def fetch_confirmed_lineup_live(target_date: str, game_pk: int) -> tuple[list[int], list[int]] | None:
    """Consulta en vivo si la alineación confirmada ya está publicada en la Stats API para el partido."""
    import requests
    url = f"https://statsapi.mlb.com/api/v1/schedule?gamePk={game_pk}&hydrate=lineups"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            dates = r.json().get("dates", [])
            if dates and dates[0].get("games"):
                g = dates[0]["games"][0]
                lineups = g.get("lineups")
                if lineups:
                    home_pids = [p["id"] for p in lineups.get("homePlayers", [])[:9]]
                    away_pids = [p["id"] for p in lineups.get("awayPlayers", [])[:9]]
                    if len(home_pids) == 9 and len(away_pids) == 9:
                        return home_pids, away_pids
    except Exception:
        pass
    return None


def get_historical_lineup(conn: sqlite3.Connection, game_pk: int, team_id: int) -> list[int]:
    """Obtiene los 9 bateadores titulares del partido histórico de boxscore_batting."""
    rows = conn.execute("""
        SELECT player_id
        FROM boxscore_batting
        WHERE game_pk=? AND team_id=?
        ORDER BY CASE WHEN batting_order IS NOT NULL THEN batting_order ELSE 999 END ASC
        LIMIT 9
    """, (game_pk, team_id)).fetchall()
    return [r[0] for r in rows]


def compute_batter_woba_rolling(
    conn: sqlite3.Connection, player_id: int, as_of_date: str, days: int = 30
) -> tuple[float | None, int]:
    """Calcula el wOBA individual de un bateador en las apariciones en el plato de los últimos N días."""
    start_dt = str(pd.to_datetime(as_of_date) - pd.Timedelta(days=days))[:10]
    season = int(as_of_date[:4])

    row = conn.execute("""
        SELECT
            SUM(b.ab) AS ab, SUM(b.h) AS h, SUM(b.doubles) AS d,
            SUM(b.triples) AS t, SUM(b.hr) AS hr, SUM(b.bb) AS bb,
            SUM(b.ibb) AS ibb, SUM(b.hbp) AS hbp, SUM(b.sf) AS sf
        FROM boxscore_batting b
        JOIN games g ON g.game_pk = b.game_pk
        WHERE b.player_id=? AND g.game_date >= ? AND g.game_date < ? AND g.status='Final'
    """, (player_id, start_dt, as_of_date)).fetchone()

    if not row or not row[0]:
        return None, 0

    ab, h, d, t, hr, bb, ibb, hbp, sf = [x or 0 for x in row]
    singles = h - (d + t + hr)
    pa = ab + (bb - ibb) + sf + hbp

    val = metrics.woba(
        bb=bb, ibb=ibb, hbp=hbp, singles=singles, doubles=d,
        triples=t, hr=hr, ab=ab, sf=sf, season=season
    )
    return val, pa


def get_lineup_projected_woba(
    conn: sqlite3.Connection, player_ids: list[int], as_of_date: str, league_woba: float = 0.315
) -> float:
    """Calcula el wOBA esperado ponderado de una alineación titular de 9 bateadores."""
    if not player_ids:
        return league_woba

    # Pesos por orden al bat (posiciones 1-4 pesan más que 8-9)
    weights = [1.2, 1.2, 1.1, 1.1, 1.0, 1.0, 0.9, 0.8, 0.7]
    weighted_woba_sum = 0.0
    weight_total = 0.0

    for i, pid in enumerate(player_ids[:9]):
        raw_woba, pa = compute_batter_woba_rolling(conn, pid, as_of_date, days=45)
        # Shrinkage agresivo hacia wOBA de liga ($k=40$ PA para novatos/apariciones cortas)
        adj_woba = metrics.shrink_rate(raw_woba, pa, league_rate=league_woba, k=40.0)

        w = weights[i] if i < len(weights) else 1.0
        weighted_woba_sum += adj_woba * w
        weight_total += w

    return round(weighted_woba_sum / weight_total, 4) if weight_total > 0 else league_woba
