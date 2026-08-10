"""
pipeline_lmb.py
Orquestador diario y pipeline de datos exclusivo para la Liga Mexicana de Béisbol (LMB).
Aísla la ingesta en `data/lmb.db`.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import db
import extract_schedule
import extract_schedule_lmb
import extract_teams
import extract_players


def run(db_path: str = "data/lmb.db", target_date: str | None = None, days_back: int = 15) -> None:
    if target_date is None:
        target_date = str(date.today())

    dt = date.fromisoformat(target_date)
    start_date = str(dt - timedelta(days=days_back))
    end_date = str(dt + timedelta(days=3))

    conn = db.get_connection(db_path)
    db.init_db(conn)

    print(f"=== PIPELINE LMB ({start_date} -> {end_date}) ===")

    # 1. Equipos LMB
    teams_json = extract_teams.fetch_teams(league_id=125)
    team_rows = extract_teams.parse_teams(teams_json, league="LMB")
    if team_rows:
        conn.executemany("""
            INSERT OR REPLACE INTO teams (team_id, name, abbreviation, league)
            VALUES (:team_id, :name, :abbreviation, :league)
        """, team_rows)
        conn.commit()
        print(f"Cargados/actualizados {len(team_rows)} equipos de la LMB.")

    # 2. Calendario y Marcadores LMB
    sched_json = extract_schedule_lmb.fetch_lmb_schedule_range(start_date, end_date)
    game_rows, probable_rows = extract_schedule_lmb.parse_lmb_games(sched_json)

    if game_rows:
        conn.executemany("""
            INSERT OR REPLACE INTO games
            (game_pk, game_date, game_date_utc, season, game_type, status,
             home_team_id, away_team_id, home_score, away_score,
             venue_id, venue_name, league)
            VALUES (:game_pk, :game_date, :game_date_utc, :season, :game_type, :status,
                    :home_team_id, :away_team_id, :home_score, :away_score,
                    :venue_id, :venue_name, :league)
        """, game_rows)

        if probable_rows:
            conn.executemany("""
                INSERT OR REPLACE INTO probable_pitchers
                (game_pk, team_id, is_home, pitcher_id, pitcher_name)
                VALUES (:game_pk, :team_id, :is_home, :pitcher_id, :pitcher_name)
            """, probable_rows)

        conn.commit()
        print(f"Cargados/actualizados {len(game_rows)} partidos y {len(probable_rows)} abridores de la LMB.")

    # 3. Boxscores para partidos Finalizados de LMB
    final_pks = [
        r[0] for r in conn.execute(
            "SELECT game_pk FROM games WHERE status='Final' AND league='LMB'"
        ).fetchall()
    ]

    unfetched = [
        pk for pk in final_pks
        if conn.execute("SELECT 1 FROM boxscore_pitching WHERE game_pk=?", (pk,)).fetchone() is None
    ]

    if unfetched:
        print(f"Descargando boxscores para {len(unfetched)} partidos LMB pendientes...")
        for i, pk in enumerate(unfetched, 1):
            try:
                data = extract_schedule.fetch_boxscore(pk)
                b_bat, b_pit = extract_schedule.parse_boxscore(data, pk)
                if b_bat:
                    conn.executemany("""
                        INSERT OR REPLACE INTO boxscore_batting
                        (game_pk, team_id, player_id, player_name, bats, ab, h, doubles, triples, hr, bb, ibb, hbp, sf, so, sb, cs)
                        VALUES (:game_pk, :team_id, :player_id, :player_name, :bats, :ab, :h, :doubles, :triples, :hr, :bb, :ibb, :hbp, :sf, :so, :sb, :cs)
                    """, b_bat)
                if b_pit:
                    conn.executemany("""
                        INSERT OR REPLACE INTO boxscore_pitching
                        (game_pk, team_id, player_id, player_name, throws, is_starter, outs, h, r, er, bb, ibb, hbp, so, hr, pitches_thrown)
                        VALUES (:game_pk, :team_id, :player_id, :player_name, :throws, :is_starter, :outs, :h, :r, :er, :bb, :ibb, :hbp, :so, :hr, :pitches_thrown)
                    """, b_pit)
                conn.commit()
            except Exception:
                pass
            if i % 20 == 0:
                print(f"  ... {i}/{len(unfetched)} boxscores procesados")

    # 4. Bio de Jugadores LMB (Mano bateador/lanzador)
    extract_players.run(conn)

    conn.close()
    print("Pipeline LMB finalizado exitosamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline diario LMB")
    parser.add_argument("--db-path", default="data/lmb.db")
    parser.add_argument("--date", default=None)
    parser.add_argument("--days-back", type=int, default=15)
    args = parser.parse_args()

    run(args.db_path, args.date, args.days_back)
