"""
extract_statcast.py
Fuente: Baseball Savant (baseballsavant.mlb.com), usando el export CSV
público del buscador de Statcast (statcast_search/csv). No es HTML a
parsear: es un endpoint de exportación pensado para descarga de datos.

Guardamos solo bolas puestas en juego (batted balls), no cada pitch,
para mantener la base liviana. De ahí sacamos exit velocity, launch
angle y barrels -> insumos de Barrel%, Hard-Hit% y clasificación de
tipo de batazo (GB/LD/FB/PU) que alimentan xFIP en metrics.py.

IMPORTANTE: no pude probar esta función contra la red real desde mi
entorno de trabajo (firewall del sandbox). La URL y las columnas están
construidas según el formato estándar y documentado de Statcast/pybaseball,
pero la primera vez que la corras en tu máquina o en GitHub Actions,
revisa el print de columnas para confirmar que nada cambió del lado de MLB.
"""

from __future__ import annotations

import argparse
import io
from datetime import date, timedelta

import pandas as pd
import requests

SEARCH_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
HEADERS = {"User-Agent": "personal-mlb-model/1.0 (uso no comercial, contacto: tu_email@ejemplo.com)"}
TIMEOUT = 30

# Columnas que nos interesan del CSV de Statcast (puede traer 90+)
COLUMNS_NEEDED = [
    "game_pk", "game_date", "batter", "pitcher", "stand", "p_throws",
    "launch_speed", "launch_angle", "hit_distance_sc", "events",
]


def _build_params(start_date: str, end_date: str) -> dict:
    """Replica los parámetros del buscador de Statcast para bolas en juego,
    tipo 'details' (una fila por evento), en un rango de fechas."""
    return {
        "all": "true",
        "hfGT": "R|",             # temporada regular
        "hfSea": "",              # se filtra por fecha, no por año fijo
        "game_date_gt": start_date,
        "game_date_lt": end_date,
        "type": "details",
        "min_pitches": 0,
        "min_results": 0,
    }


def fetch_statcast_range(start_date: str, end_date: str) -> pd.DataFrame:
    resp = requests.get(SEARCH_URL, params=_build_params(start_date, end_date),
                         headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    return df


def classify_batted_ball(launch_angle: float | None) -> str | None:
    """Clasificación estándar aproximada por ángulo de lanzamiento."""
    if launch_angle is None or pd.isna(launch_angle):
        return None
    if launch_angle < 10:
        return "GB"
    if launch_angle < 25:
        return "LD"
    if launch_angle < 50:
        return "FB"
    return "PU"


def is_barrel(ev: float | None, la: float | None) -> int | None:
    """Aproximación de la definición oficial de Barrel de Statcast:
    EV minimo 98 mph, con una ventana de angulo que se ensancha a medida
    que sube el EV (8-50 grados en 116+ mph). No es la tabla exacta de
    MLB grado a grado, pero es una buena aproximacion para features de
    calidad de contacto."""
    if ev is None or la is None or pd.isna(ev) or pd.isna(la):
        return None
    if ev < 98:
        return 0
    if ev >= 116:
        lo, hi = 8, 50
    else:
        frac = (ev - 98) / (116 - 98)
        lo = 26 - frac * (26 - 8)
        hi = 30 + frac * (50 - 30)
    return 1 if lo <= la <= hi else 0


def parse_statcast_df(df: pd.DataFrame) -> list[dict]:
    """Convierte el DataFrame crudo de Savant a filas listas para la tabla
    statcast_batted_balls. Ignora filas sin evento de bateo (pitches que
    no terminaron en bola puesta en juego)."""
    missing = [c for c in COLUMNS_NEEDED if c not in df.columns]
    if missing:
        print(f"[extract_statcast] AVISO: columnas no encontradas {missing}. "
              f"Columnas disponibles: {list(df.columns)[:20]}...")

    rows: list[dict] = []
    for _, r in df.iterrows():
        events = r.get("events")
        if pd.isna(events) or events in ("", None):
            continue  # no fue una bola puesta en juego con resultado
        ev = r.get("launch_speed")
        la = r.get("launch_angle")
        ev = None if pd.isna(ev) else float(ev)
        la = None if pd.isna(la) else float(la)

        rows.append({
            "game_pk": None if pd.isna(r.get("game_pk")) else int(r.get("game_pk")),
            "game_date": r.get("game_date"),
            "batter_id": int(r.get("batter")),
            "pitcher_id": int(r.get("pitcher")),
            "stand": r.get("stand"),
            "p_throws": r.get("p_throws"),
            "launch_speed": ev,
            "launch_angle": la,
            "hit_distance": None if pd.isna(r.get("hit_distance_sc")) else float(r.get("hit_distance_sc")),
            "events": events,
            "barrel": is_barrel(ev, la),
        })
    return rows


def upsert_batted_balls(conn, rows: list[dict]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO statcast_batted_balls
           (game_pk, game_date, batter_id, pitcher_id, stand, p_throws,
            launch_speed, launch_angle, hit_distance, events, barrel)
           VALUES (:game_pk, :game_date, :batter_id, :pitcher_id, :stand,
                   :p_throws, :launch_speed, :launch_angle, :hit_distance,
                   :events, :barrel)""",
        rows,
    )
    conn.commit()


def run_for_range(conn, start_date: str, end_date: str) -> dict:
    df = fetch_statcast_range(start_date, end_date)
    rows = parse_statcast_df(df)
    upsert_batted_balls(conn, rows)
    return {"batted_balls_cargados": len(rows), "filas_crudas_recibidas": len(df)}


if __name__ == "__main__":
    import db

    parser = argparse.ArgumentParser(description="Extrae batted balls de Statcast para un rango")
    yesterday = str(date.today() - timedelta(days=1))
    parser.add_argument("--start-date", default=yesterday)
    parser.add_argument("--end-date", default=yesterday)
    parser.add_argument("--db-path", default="data/mlb.db")
    args = parser.parse_args()

    conn = db.get_connection(args.db_path)
    db.init_db(conn)
    resumen = run_for_range(conn, args.start_date, args.end_date)
    print(f"[{args.start_date} a {args.end_date}] {resumen}")
    conn.close()
