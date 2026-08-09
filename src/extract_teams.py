"""
extract_teams.py
Nombres de equipo -- /api/v1/teams los trae todos en una sola llamada.
Sin esto, los reportes solo tendrían team_id (147, 111...) en vez de
"Yankees", "Red Sox".
"""

from __future__ import annotations

import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "personal-mlb-model/1.0 (uso no comercial, contacto: tu_email@ejemplo.com)"}
TIMEOUT = 20


def fetch_teams(sport_id: int = 1, league_id: int | None = None) -> dict:
    params = {"sportId": sport_id}
    if league_id:
        params["leagueId"] = league_id
    resp = requests.get(f"{BASE_URL}/teams", params=params,
                        headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def parse_teams(teams_json: dict, league: str = "MLB") -> list[dict]:
    rows = []
    for t in teams_json.get("teams", []):
        rows.append({
            "team_id": t["id"],
            "name": t.get("name"),
            "abbreviation": t.get("abbreviation"),
            "league": league,
        })
    return rows


def upsert_teams(conn, rows: list[dict]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO teams (team_id, name, abbreviation, league)
           VALUES (:team_id, :name, :abbreviation, :league)""",
        rows,
    )
    conn.commit()


def run(conn) -> int:
    mlb_rows = parse_teams(fetch_teams(sport_id=1), league="MLB")
    lmb_rows = parse_teams(fetch_teams(league_id=125), league="LMB")
    all_rows = mlb_rows + lmb_rows
    upsert_teams(conn, all_rows)
    print(f"[extract_teams] {len(all_rows)} equipos cargados ({len(mlb_rows)} MLB, {len(lmb_rows)} LMB)")
    return len(all_rows)


if __name__ == "__main__":
    import db

    conn = db.get_connection("data/mlb.db")
    db.init_db(conn)
    run(conn)
    conn.close()
