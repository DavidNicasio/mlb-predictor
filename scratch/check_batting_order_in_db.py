import sqlite3

conn = sqlite3.connect("data/mlb.db")

# Tomar 5 partidos al azar y ver los primeros 9 bateadores insertados por equipo
g_pks = [r[0] for r in conn.execute("SELECT game_pk FROM games WHERE status='Final' AND season=2024 LIMIT 5").fetchall()]

for pk in g_pks:
    print(f"\n--- Game PK: {pk} ---")
    h_team, a_team = conn.execute("SELECT home_team_id, away_team_id FROM games WHERE game_pk=?", (pk,)).fetchone()

    h_batters = conn.execute("SELECT player_id, player_name, ab FROM boxscore_batting WHERE game_pk=? AND team_id=?", (pk, h_team)).fetchall()
    a_batters = conn.execute("SELECT player_id, player_name, ab FROM boxscore_batting WHERE game_pk=? AND team_id=?", (pk, a_team)).fetchall()

    print(f"Home Team ({len(h_batters)} bateadores totales en boxscore):")
    for b in h_batters[:9]:
        print(f"   {b[1]} (AB: {b[2]})")

    print(f"Away Team ({len(a_batters)} bateadores totales en boxscore):")
    for b in a_batters[:9]:
        print(f"   {b[1]} (AB: {b[2]})")

conn.close()
