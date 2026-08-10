"""
extract_schedule_lmb.py
Extractor de calendarios, abridores probables, marcadores y boxscores para la
Liga Mexicana de Béisbol (LMB) usando la MLB Stats API oficial (sportId=23, leagueId=125).
"""

from __future__ import annotations

from typing import Any
import requests

LMB_SPORT_ID = 23
LMB_LEAGUE_ID = 125
BASE_URL = "https://statsapi.mlb.com/api/v1"


def fetch_lmb_schedule_range(start_date: str, end_date: str) -> dict[str, Any]:
    url = f"{BASE_URL}/schedule"
    params = {
        "sportId": LMB_SPORT_ID,
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "team,probablePitcher,linescore,boxscore",
    }
    r = requests.get(url, params=params, timeout=30.0)
    r.raise_for_status()
    return r.json()


def parse_lmb_games(schedule_data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    games = []
    probables = []

    for d in schedule_data.get("dates", []):
        g_date = d.get("date")
        for g in d.get("games", []):
            game_type = g.get("gameType", "R")
            game_pk = g.get("gamePk")
            status = g.get("status", {}).get("detailedState", "Scheduled")
            venue = g.get("venue", {})

            home_team = g.get("teams", {}).get("home", {})
            away_team = g.get("teams", {}).get("away", {})

            home_id = home_team.get("team", {}).get("id")
            away_id = away_team.get("team", {}).get("id")

            home_score = home_team.get("score")
            away_score = away_team.get("score")

            games.append({
                "game_pk": game_pk,
                "game_date": g_date,
                "game_date_utc": g.get("gameDate"),
                "season": int(g.get("season", g_date[:4])),
                "game_type": game_type,
                "status": "Final" if status in ["Final", "Completed Early"] else status,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_score": home_score,
                "away_score": away_score,
                "venue_id": venue.get("id"),
                "venue_name": venue.get("name"),
                "league": "LMB",
            })

            for side, is_home in (("home", 1), ("away", 0)):
                prob = g.get("teams", {}).get(side, {}).get("probablePitcher")
                if prob:
                    probables.append({
                        "game_pk": game_pk,
                        "team_id": home_id if is_home else away_id,
                        "is_home": is_home,
                        "pitcher_id": prob.get("id"),
                        "pitcher_name": prob.get("fullName"),
                    })

    return games, probables
