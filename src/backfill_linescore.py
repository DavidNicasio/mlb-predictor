"""
backfill_linescore.py
Backfill de linescore inning-by-inning para todos los partidos finalizados.
Usa GET /api/v1/game/{game_pk}/linescore de la MLB Stats API.

Resumible: salta game_pks que ya tienen filas en game_linescore.
Paralelizable: usa ThreadPool como el backfill de boxscores.

Uso:
    python src/backfill_linescore.py
    python src/backfill_linescore.py --workers 8
    python src/backfill_linescore.py --season 2024
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import requests

import db

BASE_URL = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "personal-mlb-model/1.0 (uso no comercial)"}
TIMEOUT = 15
MAX_RETRIES = 3


def _get(url: str) -> dict:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as err:
            last_err = err
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Fallo al llamar {url} tras {MAX_RETRIES} intentos: {last_err}")


def fetch_linescore(game_pk: int) -> list[dict]:
    """Retorna lista de dicts con {game_pk, inning, home_runs, away_runs}."""
    data = _get(f"{BASE_URL}/game/{game_pk}/linescore")
    innings = data.get("innings", [])
    rows = []
    for inn in innings:
        rows.append({
            "game_pk": game_pk,
            "inning": inn.get("num"),
            "home_runs": inn.get("home", {}).get("runs"),
            "away_runs": inn.get("away", {}).get("runs"),
        })
    return rows


def upsert_linescore(conn, rows: list[dict]) -> None:
    if not rows:
        return
    conn.executemany(
        """INSERT OR REPLACE INTO game_linescore
           (game_pk, inning, home_runs, away_runs)
           VALUES (:game_pk, :inning, :home_runs, :away_runs)""",
        rows,
    )
    conn.commit()


def _fetch_linescore_safe(game_pk: int) -> tuple[int, list[dict] | None]:
    try:
        return game_pk, fetch_linescore(game_pk)
    except Exception as err:
        return game_pk, None


def run(db_path: str = "data/mlb.db", max_workers: int = 6,
        season: int | None = None) -> None:
    conn = db.get_connection(db_path)
    db.init_db(conn)

    # Game PKs que ya tienen linescore
    already = {r[0] for r in conn.execute("SELECT DISTINCT game_pk FROM game_linescore")}

    # Game PKs que necesitan linescore
    if season:
        target_pks = [
            r[0] for r in conn.execute(
                "SELECT game_pk FROM games WHERE game_type='R' AND status='Final' AND season=?",
                (season,),
            ) if r[0] not in already
        ]
    else:
        target_pks = [
            r[0] for r in conn.execute(
                "SELECT game_pk FROM games WHERE game_type='R' AND status='Final'"
            ) if r[0] not in already
        ]

    print(f"Linescore backfill: {len(target_pks)} partidos pendientes "
          f"({len(already)} ya cargados)")

    if not target_pks:
        print("Nada que hacer.")
        conn.close()
        return

    done, fallos = 0, 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_linescore_safe, pk): pk for pk in target_pks}
        for fut in as_completed(futures):
            pk = futures[fut]
            game_pk_result, rows = fut.result()
            if rows is not None:
                upsert_linescore(conn, rows)
            else:
                fallos += 1
            done += 1

            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(target_pks) - done) / rate / 60 if rate > 0 else 0
                print(f"  ... {done}/{len(target_pks)} linescores "
                      f"({fallos} fallos, {elapsed:.0f}s, ~{remaining:.1f}min restantes)")

    elapsed = time.time() - t0
    print(f"\nLinescore backfill terminado: {done} procesados, {fallos} fallos, "
          f"{elapsed:.0f}s total")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill de linescore inning-by-inning")
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--season", type=int, default=None,
                        help="Solo backfillear una temporada específica")
    args = parser.parse_args()
    run(args.db_path, args.workers, args.season)
