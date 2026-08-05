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


def fetch_teams(sport_id: int = 1) -> dict:
    resp = requests.get(f"{BASE_URL}/teams", params={"sportId": sport_id},
                        headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def parse_teams(teams_json: dict) -> list[dict]:
    rows = []
    for t in teams_json.get("teams", []):
        rows.append({
            "team_id": t["id"],
            "name": t.get("name"),
            "abbreviation": t.get("abbreviation"),
        })
    return rows


def upsert_teams(conn, rows: list[dict]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO teams (team_id, name, abbreviation)
           VALUES (:team_id, :name, :abbreviation)""",
        rows,
    )
    conn.commit()


def run(conn) -> int:
    teams_json = fetch_teams()
    rows = parse_teams(teams_json)
    upsert_teams(conn, rows)
    print(f"[extract_teams] {len(rows)} equipos cargados")
    return len(rows)


if __name__ == "__main__":
    import db

    conn = db.get_connection("data/mlb.db")
    db.init_db(conn)
    run(conn)
    conn.close()
