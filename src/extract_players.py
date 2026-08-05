"""
extract_players.py
Bats/throws NO vienen en el boxscore -- confirmado contra la API real
que el 'person' embebido ahi es solo {id, fullName, link, boxscoreName}.
Esos datos biograficos viven en el endpoint /people, que sí soporta
consultar muchos IDs de una sola vez via el parametro personIds.

Como solo hay unos pocos miles de jugadores unicos en toda la historia
(no uno por partido), esto es MUCHO mas barato que re-descargar boxscores:
~20-30 llamadas en vez de ~29,000.
"""

from __future__ import annotations

import argparse
import time

import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "personal-mlb-model/1.0 (uso no comercial, contacto: tu_email@ejemplo.com)"}
TIMEOUT = 20
CHUNK_SIZE = 100


def fetch_people(player_ids: list[int]) -> dict:
    ids_str = ",".join(str(i) for i in player_ids)
    resp = requests.get(f"{BASE_URL}/people", params={"personIds": ids_str},
                        headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def parse_people(people_json: dict) -> list[dict]:
    rows = []
    for p in people_json.get("people", []):
        rows.append({
            "player_id": p["id"],
            "full_name": p.get("fullName"),
            "bats": p.get("batSide", {}).get("code"),
            "throws": p.get("pitchHand", {}).get("code"),
        })
    return rows


def upsert_players(conn, rows: list[dict]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO players (player_id, full_name, bats, throws)
           VALUES (:player_id, :full_name, :bats, :throws)""",
        rows,
    )
    conn.commit()


def missing_player_ids(conn) -> list[int]:
    """IDs que aparecen en boxscores pero todavia no tenemos en players."""
    rows = conn.execute(
        """SELECT DISTINCT player_id FROM (
               SELECT player_id FROM boxscore_batting
               UNION
               SELECT player_id FROM boxscore_pitching
           ) WHERE player_id NOT IN (SELECT player_id FROM players)"""
    ).fetchall()
    return [r[0] for r in rows]


def run(conn, chunk_size: int = CHUNK_SIZE) -> int:
    ids = missing_player_ids(conn)
    if not ids:
        print("[extract_players] no hay jugadores nuevos por consultar")
        return 0

    print(f"[extract_players] {len(ids)} jugadores nuevos por consultar "
          f"({(len(ids) + chunk_size - 1) // chunk_size} llamadas)")

    total = 0
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i + chunk_size]
        try:
            people_json = fetch_people(chunk)
            rows = parse_people(people_json)
            upsert_players(conn, rows)
            total += len(rows)
        except Exception as err:  # noqa: BLE001
            print(f"  [fallo] bloque {i}-{i+len(chunk)}: {err}")
        time.sleep(0.3)

    print(f"[extract_players] {total} jugadores cargados")
    return total


if __name__ == "__main__":
    import db

    parser = argparse.ArgumentParser(description="Carga bats/throws de jugadores faltantes")
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    conn = db.get_connection(args.db_path)
    db.init_db(conn)
    run(conn, args.chunk_size)
    conn.close()
