"""
build_training_dataset.py
Corre build_features_for_date() sobre TODO el historial (o un rango),
guarda cada fila en la tabla `game_features` de la base local (resumible
por game_pk), y al final exporta todo a un .parquet listo para la Fase 4.

Solo incluye partidos con status='Final' (se necesita el resultado real
como variable objetivo para entrenar). No requiere red -- todo esto es
consultas locales a SQLite, así que corre mucho más rápido que el
backfill original.

Uso:
    python src/build_training_dataset.py
    python src/build_training_dataset.py --start-date 2023-01-01
"""

from __future__ import annotations

import argparse
import time
from datetime import date

import db
import features

# Columnas que son texto de verdad; todo lo demás se guarda como NUMERIC
# (SQLite intenta guardarlo como INTEGER/REAL y solo cae a texto si el
# valor no es numérico -- evita el problema de que una columna quede
# mal tipada por culpa de un None en la primera fila que se procese).
TEXT_COLUMNS = {"game_date", "status", "home_abridor_throws", "away_abridor_throws",
                "weather_condition", "weather_wind"}


def distinct_game_dates(conn, start_date: str, end_date: str) -> list[str]:
    rows = conn.execute(
        """SELECT DISTINCT game_date FROM games
           WHERE game_type = 'R' AND status = 'Final'
                 AND game_date >= ? AND game_date < ?
           ORDER BY game_date""",
        (start_date, end_date),
    ).fetchall()
    return [r[0] for r in rows]


def _table_exists(conn, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def ensure_game_features_table(conn, sample_row: dict) -> None:
    cols = []
    for k in sample_row.keys():
        if k == "game_pk":
            continue
        col_type = "TEXT" if k in TEXT_COLUMNS else "NUMERIC"
        cols.append(f'"{k}" {col_type}')
    ddl = f'CREATE TABLE IF NOT EXISTS game_features (game_pk INTEGER PRIMARY KEY, {", ".join(cols)})'
    conn.execute(ddl)
    conn.commit()


def already_processed_game_pks(conn) -> set[int]:
    if not _table_exists(conn, "game_features"):
        return set()
    return {r[0] for r in conn.execute("SELECT game_pk FROM game_features")}


def upsert_game_features(conn, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    col_names = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    conn.executemany(
        f'INSERT OR REPLACE INTO game_features ({col_names}) VALUES ({placeholders})',
        rows,
    )
    conn.commit()


def _compute_date_features(db_path: str, d: str) -> list[dict]:
    import sqlite3
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        rows = features.build_features_for_date(conn, d)
        return [r for r in rows if r.get("status") == "Final"]
    finally:
        conn.close()


def run(db_path: str = "data/mlb.db", start_date: str = "2015-01-01",
        end_date: str | None = None, export_path: str = "data/training_dataset.parquet",
        progress_every: int = 100, max_workers: int = 6) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    conn = db.get_connection(db_path)
    db.init_db(conn)
    end_date = end_date or str(date.today())

    dates = distinct_game_dates(conn, start_date, end_date)
    print(f"{len(dates)} fechas con partidos Final entre {start_date} y {end_date}")

    table_ready = _table_exists(conn, "game_features")
    processed_pks = already_processed_game_pks(conn)
    if processed_pks:
        print(f"{len(processed_pks)} partidos ya estaban en game_features (se saltan)")

    # Filtrar fechas que ya tengan todos sus partidos procesados
    pending_dates = []
    for d in dates:
        day_pks = {
            r[0] for r in conn.execute(
                "SELECT game_pk FROM games WHERE game_date=? AND game_type='R' AND status='Final'",
                (d,),
            )
        }
        if not (day_pks and day_pks.issubset(processed_pks)):
            pending_dates.append(d)

    print(f"{len(pending_dates)} fechas pendientes por procesar con {max_workers} hilos paralelos...")

    total_nuevas = 0
    t0 = time.time()
    done_dates = 0

    if pending_dates:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_date = {pool.submit(_compute_date_features, db_path, d): d for d in pending_dates}

            batch_rows = []
            for future in as_completed(future_to_date):
                d = future_to_date[future]
                done_dates += 1
                try:
                    rows = future.result()
                    if rows:
                        batch_rows.extend(rows)
                except Exception as err:
                    print(f"  [fallo] {d}: {err}")

                if len(batch_rows) >= 200 or done_dates == len(pending_dates):
                    if batch_rows:
                        if not table_ready:
                            ensure_game_features_table(conn, batch_rows[0])
                            table_ready = True
                        upsert_game_features(conn, batch_rows)
                        total_nuevas += len(batch_rows)
                        batch_rows = []

                if done_dates % progress_every == 0 or done_dates == len(pending_dates):
                    elapsed = time.time() - t0
                    rate = done_dates / elapsed if elapsed > 0 else 0
                    rem_min = (len(pending_dates) - done_dates) / rate / 60 if rate > 0 else 0
                    print(f"  ... {done_dates}/{len(pending_dates)} fechas ({total_nuevas} filas nuevas, {elapsed:.0f}s, ~{rem_min:.1f}min restantes)")

    print(f"\n{total_nuevas} filas nuevas agregadas a game_features.")

    if table_ready:
        import pandas as pd
        query = """
            SELECT g.*, 
                   f5.home_score_f5, f5.away_score_f5, f5.total_runs_f5,
                   CASE WHEN f5.home_score_f5 > f5.away_score_f5 THEN 1 ELSE 0 END AS home_win_f5,
                   l1.nrfi
            FROM game_features g
            LEFT JOIN (
                SELECT game_pk,
                       SUM(home_runs) AS home_score_f5,
                       SUM(away_runs) AS away_score_f5,
                       SUM(home_runs + away_runs) AS total_runs_f5
                FROM game_linescore
                WHERE inning <= 5
                GROUP BY game_pk
            ) f5 ON f5.game_pk = g.game_pk
            LEFT JOIN (
                SELECT game_pk,
                       CASE WHEN SUM(home_runs + away_runs) = 0 THEN 1 ELSE 0 END AS nrfi
                FROM game_linescore
                WHERE inning = 1
                GROUP BY game_pk
            ) l1 ON l1.game_pk = g.game_pk
        """
        df = pd.read_sql_query(query, conn)
        for col in df.columns:
            if col not in TEXT_COLUMNS and col != "game_pk":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.to_parquet(export_path, index=False)
        print(f"Exportado: {export_path} ({len(df)} filas, {len(df.columns)} columnas)")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arma el dataset de entrenamiento histórico")
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--export-path", default="data/training_dataset.parquet")
    args = parser.parse_args()

    run(args.db_path, args.start_date, args.end_date, args.export_path)
