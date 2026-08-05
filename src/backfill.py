"""
backfill.py
Carga histórica ÚNICA (2015-presente por defecto). Esto NO lo corre
GitHub Actions -- lo corres tú, una vez, en tu máquina local.

Es resumible: si lo detienes (Ctrl+C) o se corta la conexión, puedes
volver a correr el mismo comando y va a saltarse todo lo que ya esté
cargado (revisa qué game_pk ya tienen boxscore antes de pedirlo de nuevo).

Por qué tarda: el calendario de una temporada completa se trae en UNA
sola llamada (el endpoint acepta rango de fechas), pero no existe un
endpoint masivo de boxscores -- hay que pedirlos uno por uno. Para no
tardar horas se descargan en paralelo con varios hilos (I/O-bound, no
CPU-bound, así que threads alcanzan sin problema).

Uso:
    python src/backfill.py                              # 2015 -> temporada actual
    python src/backfill.py --season 2021                # una sola temporada
    python src/backfill.py --start-season 2015 --end-season 2019
    python src/backfill.py --workers 10                 # más paralelismo

Tiempo estimado con 6 hilos: del orden de 1-2 horas para todo 2015-presente,
dependiendo de tu conexión. Statcast es rápido (se trae en bloques de
fechas); los boxscores son lo lento.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import db
import extract_players
import extract_schedule
import extract_statcast
import metrics


def _season_date_range(season: int) -> tuple[str, str]:
    today = date.today()
    start = f"{season}-02-01"
    end = str(today) if season == today.year else f"{season}-11-30"
    return start, end


def backfill_schedule(conn, season: int) -> list[dict]:
    """Trae TODO el calendario de la temporada en una sola llamada."""
    start, end = _season_date_range(season)
    schedule_json = extract_schedule.fetch_schedule_range(start, end)
    games_rows, probable_rows = extract_schedule.parse_schedule(schedule_json)
    extract_schedule.upsert_games(conn, games_rows)
    extract_schedule.upsert_probables(conn, probable_rows)
    print(f"[{season}] calendario: {len(games_rows)} partidos, {len(probable_rows)} probables")
    return games_rows


def _fetch_and_parse_boxscore(game_pk: int):
    box_json = extract_schedule.fetch_boxscore(game_pk)
    return extract_schedule.parse_boxscore(box_json, game_pk)


def backfill_boxscores(conn, games_rows: list[dict], max_workers: int = 6) -> int:
    """Descarga boxscores en paralelo, saltando los que ya estén en la base.
    Las escrituras a SQLite se hacen todas desde el hilo principal (SQLite
    no maneja bien escrituras concurrentes desde varios hilos)."""
    already = {r[0] for r in conn.execute("SELECT DISTINCT game_pk FROM boxscore_pitching")}
    target_pks = [
        g["game_pk"] for g in games_rows
        if g["game_type"] == "R" and g["status"] == "Final" and g["game_pk"] not in already
    ]

    print(f"  {len(target_pks)} boxscores por descargar "
          f"({len(games_rows) - len(target_pks)} ya estaban en la base o no son Final/regular)")
    if not target_pks:
        return 0

    done, fallos = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_and_parse_boxscore, pk): pk for pk in target_pks}
        for fut in as_completed(futures):
            pk = futures[fut]
            try:
                batting_rows, pitching_rows = fut.result()
                extract_schedule.upsert_batting(conn, batting_rows)
                extract_schedule.upsert_pitching(conn, pitching_rows)
            except Exception as err:  # noqa: BLE001
                fallos += 1
                print(f"  [fallo] boxscore {pk}: {err}")
            done += 1
            if done % 200 == 0:
                print(f"  ... {done}/{len(target_pks)} boxscores procesados ({fallos} fallos)")

    print(f"  boxscores terminados: {done} procesados, {fallos} fallos")
    return done


def backfill_statcast(conn, season: int, chunk_days: int = 3) -> int:
    """Trae batted balls en bloques de `chunk_days` días (Baseball Savant
    limita filas por request; bloques chicos evitan resultados truncados).

    La resumibilidad se controla con `ingestion_log`, NO con si hay filas
    en statcast_batted_balls: un día sin partidos (ej. día libre, All-Star
    break) legítimamente no genera batted balls, y si nos guiáramos solo
    por eso volveríamos a pedir ese bloque en cada corrida para siempre."""
    start, end = _season_date_range(season)
    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)

    done_ranges = {
        (r[0], r[1]) for r in conn.execute(
            "SELECT range_start, range_end FROM ingestion_log WHERE source='statcast'"
        )
    }

    total_rows = 0
    cur = start_d
    while cur <= end_d:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end_d)
        key = (str(cur), str(chunk_end))

        if key in done_ranges:
            cur = chunk_end + timedelta(days=1)
            continue

        try:
            resumen = extract_statcast.run_for_range(conn, str(cur), str(chunk_end))
            total_rows += resumen["batted_balls_cargados"]
            conn.execute(
                """INSERT OR REPLACE INTO ingestion_log (source, range_start, range_end)
                   VALUES ('statcast', ?, ?)""",
                key,
            )
            conn.commit()
        except Exception as err:  # noqa: BLE001
            # OJO: si falla, NO se marca en ingestion_log, así que la
            # próxima corrida sí va a reintentar este bloque.
            print(f"  [fallo] statcast {cur} a {chunk_end}: {err}")

        time.sleep(0.3)  # ritmo cortés entre bloques
        cur = chunk_end + timedelta(days=1)

    print(f"[{season}] statcast: {total_rows} batted balls nuevos")
    return total_rows


def run(start_season: int, end_season: int, db_path: str = "data/mlb.db",
        max_workers: int = 6, chunk_days: int = 3) -> None:
    conn = db.get_connection(db_path)
    db.init_db(conn)

    for season in range(start_season, end_season + 1):
        print(f"\n=== Temporada {season} ===")
        try:
            games_rows = backfill_schedule(conn, season)
            backfill_boxscores(conn, games_rows, max_workers=max_workers)
            backfill_statcast(conn, season, chunk_days=chunk_days)
            metrics.compute_league_constants(conn, season)
            extract_players.run(conn)
        except Exception as err:  # noqa: BLE001
            print(f"[{season}] la temporada falló por completo: {err}")
            print(f"  puedes reintentarla sola con: python src/backfill.py --season {season}")

    conn.close()
    print("\nBackfill terminado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill histórico de datos MLB")
    parser.add_argument("--season", type=int, default=None, help="cargar una sola temporada")
    parser.add_argument("--start-season", type=int, default=2015)
    parser.add_argument("--end-season", type=int, default=date.today().year)
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--workers", type=int, default=6,
                         help="hilos paralelos para boxscores (baja esto si ves muchos fallos)")
    parser.add_argument("--chunk-days", type=int, default=3,
                         help="tamaño de bloque de fechas para Statcast")
    args = parser.parse_args()

    if args.season:
        run(args.season, args.season, args.db_path, args.workers, args.chunk_days)
    else:
        run(args.start_season, args.end_season, args.db_path, args.workers, args.chunk_days)
