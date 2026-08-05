"""
inspect_db.py
Utilidad rápida para revisar data/mlb.db sin necesitar el CLI de sqlite3
(que en Windows no viene instalado por defecto, a diferencia de Mac/Linux).

Uso:
    python src/inspect_db.py                       -> resumen de filas por tabla
    python src/inspect_db.py --table games          -> últimas 5 filas de games
    python src/inspect_db.py --table games --limit 15
"""

from __future__ import annotations

import argparse
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--table", default=None,
                         help="si se omite, muestra el conteo de filas de todas las tablas")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--order-by", default=None, help="columna para ordenar (ej. factor_runs)")
    parser.add_argument("--desc", action="store_true", help="orden descendente")
    parser.add_argument("--sql", default=None,
                         help="query SQL arbitraria (SELECT), para joins/filtros que --table no cubre")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row

    if args.sql:
        for row in conn.execute(args.sql).fetchall():
            print(dict(row))
    elif args.table is None:
        print(f"Resumen de '{args.db_path}':\n")
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")]
        for t in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:<28} {count} filas")
        print("\nUsa --table <nombre> para ver el detalle de una tabla.")
    else:
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({args.table})")]
        order_col = args.order_by or ("game_date" if "game_date" in cols else None)
        query = f"SELECT * FROM {args.table}"
        if order_col:
            # Por compatibilidad: game_date por defecto sigue siendo DESC
            # (más reciente primero); cualquier --order-by explícito es
            # ASC salvo que se pida --desc.
            default_desc = args.order_by is None and order_col == "game_date"
            direction = "DESC" if (args.desc or default_desc) else "ASC"
            query += f" ORDER BY {order_col} {direction}"
        query += f" LIMIT {args.limit}"

        rows = conn.execute(query).fetchall()
        if not rows:
            print(f"'{args.table}' está vacía todavía.")
        for row in rows:
            print(dict(row))

    conn.close()


if __name__ == "__main__":
    main()
