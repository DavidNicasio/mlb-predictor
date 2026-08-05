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
TEXT_COLUMNS = {"game_date", "status", "home_abridor_throws", "away_abridor_throws"}


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


def run(db_path: str = "data/mlb.db", start_date: str = "2015-01-01",
        end_date: str | None = None, export_path: str = "data/training_dataset.parquet",
        progress_every: int = 50) -> None:
    conn = db.get_connection(db_path)
    db.init_db(conn)  # crea índices nuevos si faltan, incluso en una base ya existente
    end_date = end_date or str(date.today())

    dates = distinct_game_dates(conn, start_date, end_date)
    print(f"{len(dates)} fechas con partidos Final entre {start_date} y {end_date}")

    table_ready = _table_exists(conn, "game_features")
    processed_pks = already_processed_game_pks(conn)
    if processed_pks:
        print(f"{len(processed_pks)} partidos ya estaban en game_features (se saltan)")

    total_nuevas = 0
    t0 = time.time()

    for i, d in enumerate(dates):
        day_pks = {
            r[0] for r in conn.execute(
                "SELECT game_pk FROM games WHERE game_date=? AND game_type='R' AND status='Final'",
                (d,),
            )
        }
        if day_pks and day_pks.issubset(processed_pks):
            continue

        try:
            rows = features.build_features_for_date(conn, d)
        except Exception as err:  # noqa: BLE001
            print(f"  [fallo] {d}: {err}")
            continue

        rows = [r for r in rows if r["status"] == "Final"]
        if not rows:
            continue

        if not table_ready:
            ensure_game_features_table(conn, rows[0])
            table_ready = True

        upsert_game_features(conn, rows)
        processed_pks.update(r["game_pk"] for r in rows)
        total_nuevas += len(rows)

        if i == 0 and total_nuevas > 0:
            eta_min = (time.time() - t0) * len(dates) / 60
            print(f"  (primera fecha: {time.time()-t0:.1f}s -> estimado total ~{eta_min:.1f} min "
                  f"para las {len(dates)} fechas)")

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  ... {i+1}/{len(dates)} fechas ({total_nuevas} filas nuevas, {elapsed:.0f}s)")

    print(f"\n{total_nuevas} filas nuevas agregadas a game_features.")

    if table_ready:
        import pandas as pd
        df = pd.read_sql_query("SELECT * FROM game_features", conn)
        # Forzar numerico explicito en todo lo que no es texto: la
        # inferencia automatica de pandas al leer de SQLite puede dejar
        # una columna como 'object' si le tocaron muchos NULL, y XGBoost
        # necesita dtypes numericos limpios.
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
