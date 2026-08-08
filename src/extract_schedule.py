"""
extract_schedule.py
Fuente: MLB Stats API (statsapi.mlb.com) - endpoint publico, no autenticado,
no documentado oficialmente por MLB pero de uso estandar en la comunidad.
Uso de los datos sujeto al aviso de copyright de MLB en gdx.mlb.com.

Qué trae:
  - Calendario del dia (o rango) con estado, marcador, venue
  - Abridores probables por equipo
  - Boxscore por partido terminado (bateo y pitcheo, a nivel de linea,
    no play-by-play)

No incluye llamadas a FanGraphs ni Baseball-Reference: esta fuente es
publica y de bajo riesgo, ideal para correr todos los dias sin restriccion.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from typing import Any

import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "personal-mlb-model/1.0 (uso no comercial, contacto: tu_email@ejemplo.com)"}
TIMEOUT = 15
MAX_RETRIES = 3


def _get(url: str, params: dict | None = None) -> dict:
    """GET con reintentos simples y backoff, para tolerar caidas puntuales."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as err:
            last_err = err
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Fallo al llamar {url} tras {MAX_RETRIES} intentos: {last_err}")


# ---------------------------------------------------------------------------
# Calendario + abridores probables
# ---------------------------------------------------------------------------

def fetch_schedule(date_str: str, sport_id: int = 1, league_id: int | None = None) -> dict:
    """date_str en formato YYYY-MM-DD. sport_id=1 (MLB), sport_id=23 y league_id=125 (LMB)."""
    params = {
        "sportId": sport_id,
        "date": date_str,
        "hydrate": "weather,team,linescore,probablePitcher",
    }
    if league_id:
        params["leagueId"] = league_id
    return _get(f"{BASE_URL}/schedule", params=params)


def fetch_schedule_range(start_date: str, end_date: str, sport_id: int = 1, league_id: int | None = None) -> dict:
    """Como fetch_schedule pero para un rango."""
    params = {
        "sportId": sport_id,
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "weather,team,linescore,probablePitcher",
    }
    if league_id:
        params["leagueId"] = league_id
    return _get(f"{BASE_URL}/schedule", params=params)


def parse_schedule(schedule_json: dict, league: str = "MLB") -> tuple[list[dict], list[dict]]:
    """Devuelve (games_rows, probable_pitcher_rows)."""
    games_rows: list[dict] = []
    probable_rows: list[dict] = []

    for day in schedule_json.get("dates", []):
        for g in day.get("games", []):
            try:
                game_pk = g["gamePk"]
                teams = g.get("teams", {})
                home = teams.get("home", {})
                away = teams.get("away", {})
                venue = g.get("venue", {})
                w = g.get("weather", {})
                temp_val = int(w.get("temp")) if w.get("temp") and str(w.get("temp")).isdigit() else None

                games_rows.append({
                    "game_pk": game_pk,
                    "game_date": day.get("date"),
                    "game_date_utc": g.get("gameDate"),
                    "season": int(g.get("season", day.get("date", "0000")[:4])),
                    "game_type": g.get("gameType"),
                    "status": g.get("status", {}).get("detailedState"),
                    "home_team_id": home.get("team", {}).get("id"),
                    "away_team_id": away.get("team", {}).get("id"),
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                    "venue_id": venue.get("id"),
                    "venue_name": venue.get("name"),
                    "weather_condition": w.get("condition"),
                    "weather_temp": temp_val,
                    "weather_wind": w.get("wind"),
                    "league": league,
                })

                for side, is_home in (("home", 1), ("away", 0)):
                    prob = teams.get(side, {}).get("probablePitcher")
                    if prob:
                        probable_rows.append({
                            "game_pk": game_pk,
                            "team_id": teams.get(side, {}).get("team", {}).get("id"),
                            "is_home": is_home,
                            "pitcher_id": prob.get("id"),
                            "pitcher_name": prob.get("fullName"),
                        })
            except (KeyError, TypeError) as err:
                # Un partido con forma inesperada no debe tumbar todo el batch
                print(f"[extract_schedule] fila de calendario omitida: {err}")
                continue

    return games_rows, probable_rows


# ---------------------------------------------------------------------------
# Boxscore (bateo y pitcheo por partido terminado)
# ---------------------------------------------------------------------------

def fetch_boxscore(game_pk: int) -> dict:
    return _get(f"{BASE_URL}/game/{game_pk}/boxscore")


def _ip_to_outs(ip_str: str | None) -> int | None:
    """Convierte innings pitcheados tipo '6.1' (6 y 1/3) a outs totales."""
    if ip_str in (None, ""):
        return None
    whole, _, frac = str(ip_str).partition(".")
    outs = int(whole) * 3 + int(frac or 0)
    return outs


def parse_boxscore(boxscore_json: dict, game_pk: int) -> tuple[list[dict], list[dict]]:
    """Devuelve (batting_rows, pitching_rows) para un game_pk."""
    batting_rows: list[dict] = []
    pitching_rows: list[dict] = []

    teams = boxscore_json.get("teams", {})
    for side in ("home", "away"):
        team_block = teams.get(side, {})
        team_id = team_block.get("team", {}).get("id")
        starter_ids = set(team_block.get("pitchers", [])[:1])  # primer id = abridor
        players = team_block.get("players", {})

        for _, pdata in players.items():
            person = pdata.get("person", {})
            player_id = person.get("id")
            stats = pdata.get("stats", {})

            batting = stats.get("batting", {})
            if batting.get("atBats") is not None or batting.get("plateAppearances"):
                batting_rows.append({
                    "game_pk": game_pk,
                    "team_id": team_id,
                    "player_id": player_id,
                    "player_name": person.get("fullName"),
                    "bats": pdata.get("batSide", {}).get("code"),
                    "ab": batting.get("atBats"),
                    "h": batting.get("hits"),
                    "doubles": batting.get("doubles"),
                    "triples": batting.get("triples"),
                    "hr": batting.get("homeRuns"),
                    "bb": batting.get("baseOnBalls"),
                    "ibb": batting.get("intentionalWalks"),
                    "hbp": batting.get("hitByPitch"),
                    "sf": batting.get("sacFlies"),
                    "so": batting.get("strikeOuts"),
                    "sb": batting.get("stolenBases"),
                    "cs": batting.get("caughtStealing"),
                })

            pitching = stats.get("pitching", {})
            if pitching.get("inningsPitched") not in (None, "0.0"):
                pitching_rows.append({
                    "game_pk": game_pk,
                    "team_id": team_id,
                    "player_id": player_id,
                    "player_name": person.get("fullName"),
                    "throws": pdata.get("pitchHand", {}).get("code"),
                    "is_starter": 1 if player_id in starter_ids else 0,
                    "outs": _ip_to_outs(pitching.get("inningsPitched")),
                    "h": pitching.get("hits"),
                    "r": pitching.get("runs"),
                    "er": pitching.get("earnedRuns"),
                    "bb": pitching.get("baseOnBalls"),
                    "ibb": pitching.get("intentionalWalks"),
                    "hbp": pitching.get("hitByPitch"),
                    "so": pitching.get("strikeOuts"),
                    "hr": pitching.get("homeRuns"),
                    "pitches_thrown": pitching.get("numberOfPitches"),
                })

    return batting_rows, pitching_rows


# ---------------------------------------------------------------------------
# Carga a SQLite (upsert simple con INSERT OR REPLACE)
# ---------------------------------------------------------------------------

def upsert_games(conn, rows: list[dict]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO games
           (game_pk, game_date, game_date_utc, season, game_type, status, home_team_id,
            away_team_id, home_score, away_score, venue_id, venue_name,
            weather_condition, weather_temp, weather_wind, league)
           VALUES (:game_pk, :game_date, :game_date_utc, :season, :game_type, :status,
                   :home_team_id, :away_team_id, :home_score, :away_score,
                   :venue_id, :venue_name,
                   :weather_condition, :weather_temp, :weather_wind, :league)""",
        rows,
    )
    conn.commit()


def upsert_probables(conn, rows: list[dict]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO probable_pitchers
           (game_pk, team_id, is_home, pitcher_id, pitcher_name)
           VALUES (:game_pk, :team_id, :is_home, :pitcher_id, :pitcher_name)""",
        rows,
    )
    conn.commit()


def upsert_batting(conn, rows: list[dict]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO boxscore_batting
           (game_pk, team_id, player_id, player_name, bats, ab, h, doubles,
            triples, hr, bb, ibb, hbp, sf, so, sb, cs)
           VALUES (:game_pk, :team_id, :player_id, :player_name, :bats, :ab,
                   :h, :doubles, :triples, :hr, :bb, :ibb, :hbp, :sf, :so,
                   :sb, :cs)""",
        rows,
    )
    conn.commit()


def upsert_pitching(conn, rows: list[dict]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO boxscore_pitching
           (game_pk, team_id, player_id, player_name, throws, is_starter,
            outs, h, r, er, bb, ibb, hbp, so, hr, pitches_thrown)
           VALUES (:game_pk, :team_id, :player_id, :player_name, :throws,
                   :is_starter, :outs, :h, :r, :er, :bb, :ibb, :hbp, :so,
                   :hr, :pitches_thrown)""",
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Orquestacion para un solo dia (usada por pipeline.py)
# ---------------------------------------------------------------------------

def run_for_date(conn, date_str: str, fetch_boxscores: bool = True,
                 sport_id: int = 1, league_id: int | None = None,
                 league: str = "MLB") -> dict:
    """Descarga calendario+probables de date_str para la liga dada."""
    schedule_json = fetch_schedule(date_str, sport_id=sport_id, league_id=league_id)
    games_rows, probable_rows = parse_schedule(schedule_json, league=league)
    upsert_games(conn, games_rows)
    upsert_probables(conn, probable_rows)

    n_boxscores = 0
    if fetch_boxscores:
        for g in games_rows:
            if g["status"] == "Final":
                time.sleep(0.5)  # ritmo cortes, no hay limite oficial publicado
                try:
                    box_json = fetch_boxscore(g["game_pk"])
                    batting_rows, pitching_rows = parse_boxscore(box_json, g["game_pk"])
                    upsert_batting(conn, batting_rows)
                    upsert_pitching(conn, pitching_rows)
                    n_boxscores += 1
                except Exception as err:
                    print(f"  [fallo boxscore {g['game_pk']}]: {err}")

    return {
        "games": len(games_rows),
        "probables": len(probable_rows),
        "boxscores_cargados": n_boxscores,
    }


if __name__ == "__main__":
    import db

    parser = argparse.ArgumentParser(description="Extrae calendario/boxscores de un dia")
    parser.add_argument("--date", default=str(date.today() - timedelta(days=1)))
    parser.add_argument("--db-path", default="data/mlb.db")
    args = parser.parse_args()

    conn = db.get_connection(args.db_path)
    db.init_db(conn)
    resumen = run_for_date(conn, args.date)
    print(f"[{args.date}] {resumen}")
    conn.close()
