"""
features_rest.py
Descanso del equipo (no del pitcher -- eso ya está en features_pitching).
En MLB casi no hay días libres de verdad (se juega casi diario), así que
lo que importa es: ¿tuvieron un día libre antes de este partido? ¿cuántos
partidos han jugado en la última semana (dobles incluidos)? ¿acaban de
cambiar de sede (proxy simple de viaje, sin ir a distancias/husos horarios)?
"""

from __future__ import annotations

from datetime import date, timedelta


def team_rest_features(conn, team_id: int, as_of_date: str,
                         todays_venue_id: int | None = None) -> dict:
    row = conn.execute(
        """SELECT game_date, venue_id FROM games
           WHERE (home_team_id = ? OR away_team_id = ?) AND status = 'Final'
                 AND game_date < ?
           ORDER BY game_date DESC LIMIT 1""",
        (team_id, team_id, as_of_date),
    ).fetchone()

    if not row:
        return {"days_rest": None, "tuvo_dia_libre": None,
                "juegos_ultimos_7d": 0, "cambio_de_sede": None}

    last_date, last_venue = row
    days_rest = (date.fromisoformat(as_of_date) - date.fromisoformat(last_date)).days

    start_7 = str(date.fromisoformat(as_of_date) - timedelta(days=7))
    juegos_7d = conn.execute(
        """SELECT COUNT(*) FROM games
           WHERE (home_team_id = ? OR away_team_id = ?) AND status = 'Final'
                 AND game_date >= ? AND game_date < ?""",
        (team_id, team_id, start_7, as_of_date),
    ).fetchone()[0]

    cambio_sede = None
    if todays_venue_id is not None and last_venue is not None:
        cambio_sede = int(todays_venue_id != last_venue)

    return {
        "days_rest": days_rest,
        "tuvo_dia_libre": int(days_rest >= 2),
        "juegos_ultimos_7d": juegos_7d,
        "cambio_de_sede": cambio_sede,
    }
