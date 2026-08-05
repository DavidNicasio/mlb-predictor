"""
features.py
El compilador de la Fase 3: build_features_for_date(fecha) arma UNA FILA
por partido programado ese día, combinando los 4 módulos:
  - features_offense   (wOBA rolling, split vs mano del abridor rival)
  - features_pitching   (FIP/xFIP rolling del abridor, descanso, carga)
  - features_bullpen    (FIP ponderado por uso reciente, fatiga)
  - features_rest       (descanso de equipo, densidad de calendario)

Cada columna queda prefijada `home_` / `away_` para dejar una sola fila
lista para el modelo de la Fase 4. Si el partido ya se jugó (status
Final), home_score/away_score quedan pobladas -- eso es lo que Fase 4
va a usar como variable objetivo al entrenar con fechas históricas.
"""

from __future__ import annotations

import features_bullpen
import features_offense
import features_pitching
import features_rest


def _prefix_keys(d: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def _team_block(conn, team_id: int, as_of_date: str, season: int, venue_id: int | None,
                 opponent_starter_throws: str | None) -> dict:
    offense = features_offense.build_offense_features(
        conn, team_id, as_of_date, season, opponent_starter_throws)
    bullpen = features_bullpen.build_bullpen_features(conn, team_id, as_of_date, season)
    rest = features_rest.team_rest_features(conn, team_id, as_of_date, todays_venue_id=venue_id)
    return {**offense, **bullpen, **rest}


def build_features_for_game(conn, game_row: dict, season: int) -> dict:
    game_pk = game_row["game_pk"]
    game_date = game_row["game_date"]
    venue_id = game_row["venue_id"]
    home_id, away_id = game_row["home_team_id"], game_row["away_team_id"]

    probables = conn.execute(
        "SELECT is_home, pitcher_id FROM probable_pitchers WHERE game_pk=?", (game_pk,)
    ).fetchall()
    home_pitcher_id = next((pid for is_home, pid in probables if is_home == 1), None)
    away_pitcher_id = next((pid for is_home, pid in probables if is_home == 0), None)

    home_pitching = features_pitching.build_pitching_features(conn, home_pitcher_id, game_date, season)
    away_pitching = features_pitching.build_pitching_features(conn, away_pitcher_id, game_date, season)

    # La ofensiva de cada equipo se mide contra la mano del ABRIDOR RIVAL
    # (lo que enfrentan hoy), no la propia.
    home_block = _team_block(conn, home_id, game_date, season, venue_id,
                              opponent_starter_throws=away_pitching["abridor_throws"])
    away_block = _team_block(conn, away_id, game_date, season, venue_id,
                              opponent_starter_throws=home_pitching["abridor_throws"])

    park_row = conn.execute(
        "SELECT factor_runs FROM park_factors WHERE venue_id=? AND season=?",
        (venue_id, season),
    ).fetchone()

    row = {
        "game_pk": game_pk,
        "game_date": game_date,
        "game_date_utc": game_row.get("game_date_utc"),
        "season": season,
        "venue_id": venue_id,
        "park_factor_runs": park_row[0] if park_row else None,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "status": game_row["status"],
        "home_score": game_row["home_score"],
        "away_score": game_row["away_score"],
        "weather_condition": game_row.get("weather_condition"),
        "weather_temp": game_row.get("weather_temp"),
        "weather_wind": game_row.get("weather_wind"),
    }
    row.update(_prefix_keys(home_pitching, "home"))
    row.update(_prefix_keys(away_pitching, "away"))
    row.update(_prefix_keys(home_block, "home"))
    row.update(_prefix_keys(away_block, "away"))
    return row


def build_features_for_date(conn, target_date: str, season: int | None = None) -> list[dict]:
    season = season or int(target_date[:4])
    cols = [d[0] for d in conn.execute("SELECT * FROM games LIMIT 0").description]
    games = conn.execute(
        "SELECT * FROM games WHERE game_date = ? AND game_type = 'R' ORDER BY game_date_utc ASC, game_pk ASC", (target_date,)
    ).fetchall()

    rows = []
    for g in games:
        game_row = dict(zip(cols, g))
        try:
            rows.append(build_features_for_game(conn, game_row, season))
        except Exception as err:  # noqa: BLE001
            print(f"[features] partido {game_row.get('game_pk')} omitido por error: {err}")
    return rows


if __name__ == "__main__":
    import argparse
    import sys
    from datetime import date

    import db

    parser = argparse.ArgumentParser(description="Arma features de todos los partidos de una fecha")
    parser.add_argument("--date", default=str(date.today()))
    parser.add_argument("--db-path", default="data/mlb.db")
    args = parser.parse_args()

    conn = db.get_connection(args.db_path)
    rows = build_features_for_date(conn, args.date)
    print(f"{len(rows)} partido(s) encontrados para {args.date}\n")
    for r in rows:
        print(r)
    conn.close()
