"""
backfill_weather.py
Completa los datos historicos de clima (weather_temp, weather_condition, weather_wind)
en la tabla games llamando a la API oficial de MLB con hydrate=weather para 2015-2024.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
import requests

import db

BASE_URL = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "personal-mlb-model/1.0 (uso no comercial)"}


def fetch_weather_range(start_date: str, end_date: str) -> dict:
    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "weather",
    }
    resp = requests.get(f"{BASE_URL}/schedule", params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_backfill_weather(db_path: str = "data/mlb.db", start_year: int = 2015, end_year: int = 2024) -> None:
    conn = db.get_connection(db_path)
    db.init_db(conn)

    print(f"Iniciando backfill de clima histórico desde {start_year} hasta {end_year}...")
    total_updated = 0

    for year in range(start_year, end_year + 1):
        start_dt = date(year, 3, 20)
        end_dt = date(year, 11, 10)
        curr = start_dt

        print(f"\n--- Procesando temporada {year} ---")
        while curr <= end_dt:
            chunk_end = min(curr + timedelta(days=20), end_dt)
            s_str = str(curr)
            e_str = str(chunk_end)

            try:
                data = fetch_weather_range(s_str, e_str)
                updates = []
                for day in data.get("dates", []):
                    for g in day.get("games", []):
                        gpk = g.get("gamePk")
                        w = g.get("weather", {})
                        temp = int(w.get("temp")) if w.get("temp") and str(w.get("temp")).isdigit() else None
                        cond = w.get("condition")
                        wind = w.get("wind")

                        if gpk and (temp is not None or wind):
                            updates.append((cond, temp, wind, gpk))

                if updates:
                    conn.executemany(
                        "UPDATE games SET weather_condition=?, weather_temp=?, weather_wind=? WHERE game_pk=?",
                        updates,
                    )
                    conn.commit()
                    total_updated += len(updates)
                    print(f"  {s_str} a {e_str}: {len(updates)} partidos actualizados con datos de clima")

            except Exception as err:
                print(f"  [error {s_str} a {e_str}]: {err}")

            curr = chunk_end + timedelta(days=1)
            time.sleep(0.3)

    conn.close()
    print(f"\nBackfill de clima finalizado: {total_updated} partidos actualizados.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill de clima historico de la MLB")
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()

    run_backfill_weather(args.db_path, args.start_year, args.end_year)
