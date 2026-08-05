"""
export_recent_window.py
La base completa (2015-presente) es demasiado grande para GitHub tal
cual (cientos de MB con Statcast, sobre el límite práctico de 100MB por
archivo). Para la automatización diaria en Actions no hace falta toda
la historia -- los rolling de Fase 3 miran hacia atrás cuando mucho
30-60 días (wOBA) o un puñado de aperturas (FIP del abridor, que rara
vez retrocede más de 1-2 meses incluso con rotación lenta).

Esto exporta una copia recortada (`data/mlb_recent.db` por defecto) con:
  - games / probable_pitchers / boxscore_batting / boxscore_pitching /
    statcast_batted_balls: solo los últimos `--days` días (100 por
    defecto, con margen de sobra sobre el rolling de 30 días)
  - players / teams / league_constants / park_factors: completas
    (son chicas, no hace falta recortarlas)

La base COMPLETA se queda solo en tu PC -- se usa para reentrenar los
modelos (Fase 4) cuando quieras, no para la automatización diaria.

Uso:
    python src/export_recent_window.py
    python src/export_recent_window.py --days 120 --dest data/mlb_recent.db
"""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta

import db

TABLES_BY_GAME_PK = ["probable_pitchers", "boxscore_batting", "boxscore_pitching"]
TABLES_BY_GAME_DATE = ["statcast_batted_balls"]
TABLES_FULL = ["players", "teams", "league_constants", "park_factors"]


def run(source_db: str = "data/mlb.db", dest_db: str = "data/mlb_recent.db",
        days: int = 100) -> None:
    cutoff = str(date.today() - timedelta(days=days))

    if os.path.exists(dest_db):
        os.remove(dest_db)

    conn = db.get_connection(dest_db)
    db.init_db(conn)
    conn.execute("ATTACH DATABASE ? AS src", (source_db,))

    n_games = conn.execute(
        "INSERT INTO games SELECT * FROM src.games WHERE game_date >= ?", (cutoff,)
    ).rowcount
    print(f"games: {n_games} filas (desde {cutoff})")

    for table in TABLES_BY_GAME_PK:
        n = conn.execute(
            f"""INSERT INTO {table} SELECT * FROM src.{table}
                WHERE game_pk IN (SELECT game_pk FROM games)"""
        ).rowcount
        print(f"{table}: {n} filas")

    for table in TABLES_BY_GAME_DATE:
        n = conn.execute(
            f"INSERT INTO {table} SELECT * FROM src.{table} WHERE game_date >= ?", (cutoff,)
        ).rowcount
        print(f"{table}: {n} filas")

    for table in TABLES_FULL:
        n = conn.execute(f"INSERT INTO {table} SELECT * FROM src.{table}").rowcount
        print(f"{table}: {n} filas (completa)")

    conn.commit()
    conn.execute("DETACH DATABASE src")
    conn.execute("VACUUM")  # compacta el archivo final
    conn.close()

    size_mb = os.path.getsize(dest_db) / (1024 * 1024)
    print(f"\n{dest_db} listo: {size_mb:.1f} MB")
    if size_mb > 90:
        print("AVISO: sigue cerca del límite de 100MB de GitHub -- considera bajar --days")


def prune(db_path: str = "data/mlb_recent.db", days: int = 100) -> None:
    """Borra filas más viejas que `days` de una base YA recortada -- se
    usa en la automatización diaria (después de pipeline.py) para que
    mlb_recent.db no crezca sin límite con el tiempo."""
    cutoff = str(date.today() - timedelta(days=days))
    conn = db.get_connection(db_path)
    conn.execute(
        "DELETE FROM boxscore_batting WHERE game_pk IN (SELECT game_pk FROM games WHERE game_date < ?)",
        (cutoff,))
    conn.execute(
        "DELETE FROM boxscore_pitching WHERE game_pk IN (SELECT game_pk FROM games WHERE game_date < ?)",
        (cutoff,))
    conn.execute(
        "DELETE FROM probable_pitchers WHERE game_pk IN (SELECT game_pk FROM games WHERE game_date < ?)",
        (cutoff,))
    conn.execute("DELETE FROM statcast_batted_balls WHERE game_date < ?", (cutoff,))
    n_games = conn.execute("DELETE FROM games WHERE game_date < ?", (cutoff,)).rowcount
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    print(f"[prune] {n_games} partidos viejos (antes de {cutoff}) eliminados de {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta o poda una ventana reciente de la base")
    parser.add_argument("--source", default="data/mlb.db")
    parser.add_argument("--dest", default="data/mlb_recent.db")
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--prune-only", action="store_true",
                         help="solo podar --dest, sin re-exportar desde --source")
    args = parser.parse_args()

    if args.prune_only:
        prune(args.dest, args.days)
    else:
        run(args.source, args.dest, args.days)
