"""
pipeline.py
Orquestador de la corrida diaria. Esto es lo que ejecuta GitHub Actions
cada mañana. Cada paso está aislado en try/except: si Statcast falla,
igual queremos guardar el calendario del día y viceversa.

Pasos:
  1. Calendario + boxscores de AYER (ya deberían estar Final)
  2. Calendario + abridores probables de HOY
  3. Batted balls de Statcast de AYER
  4. Recalcular constantes de liga de la temporada actual
"""

from __future__ import annotations

import sys
import traceback
from datetime import date, timedelta

import db
import extract_players
import extract_schedule
import extract_statcast
import extract_teams
import metrics


def _safe(step_name: str, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        print(f"[OK] {step_name}: {result}")
        return result
    except Exception as err:  # noqa: BLE001 - queremos capturar cualquier falla y seguir
        print(f"[FALLO] {step_name}: {err}")
        traceback.print_exc()
        return None


def run(db_path: str = "data/mlb.db", target_date: str | None = None) -> None:
    conn = db.get_connection(db_path)
    db.init_db(conn)

    if target_date:
        today = date.fromisoformat(target_date)
    else:
        today = date.today()

    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    today_str, yesterday_str, tomorrow_str = str(today), str(yesterday), str(tomorrow)

    # --- MLB (sport_id=1) ---
    _safe(f"MLB schedule+boxscores de {yesterday_str}",
          extract_schedule.run_for_date, conn, yesterday_str, True, 1, None, "MLB")

    _safe(f"MLB schedule+probables de {today_str}",
          extract_schedule.run_for_date, conn, today_str, False, 1, None, "MLB")

    _safe(f"MLB schedule+probables de {tomorrow_str}",
          extract_schedule.run_for_date, conn, tomorrow_str, False, 1, None, "MLB")

    # --- LMB (sport_id=23, league_id=125) ---
    _safe(f"LMB schedule+boxscores de {yesterday_str}",
          extract_schedule.run_for_date, conn, yesterday_str, True, 23, 125, "LMB")

    _safe(f"LMB schedule+probables de {today_str}",
          extract_schedule.run_for_date, conn, today_str, False, 23, 125, "LMB")

    _safe(f"LMB schedule+probables de {tomorrow_str}",
          extract_schedule.run_for_date, conn, tomorrow_str, False, 23, 125, "LMB")

    _safe(f"statcast de {yesterday_str}",
          extract_statcast.run_for_range, conn, yesterday_str, today_str)

    _safe("jugadores nuevos (bats/throws)", extract_players.run, conn)

    _safe("nombres de equipo", extract_teams.run, conn)

    _safe(f"constantes de liga {today.year}",
          metrics.compute_league_constants, conn, today.year)

    conn.close()
    print("Pipeline diario terminado.")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/mlb.db"
    run(db_path)
