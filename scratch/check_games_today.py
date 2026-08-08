import sqlite3
from datetime import date

today_str = str(date.today())
conn = sqlite3.connect("data/mlb.db")

print(f"Buscando partidos en 'games' para fecha = {today_str}:")
rows = conn.execute("""
    SELECT g.game_pk, g.game_date, g.game_date_utc, g.status, ht.name, at.name
    FROM games g
    LEFT JOIN teams ht ON ht.team_id = g.home_team_id
    LEFT JOIN teams at ON at.team_id = g.away_team_id
    WHERE g.game_date = ?
""", (today_str,)).fetchall()

print(f"Encontrados {len(rows)} partidos:")
for r in rows:
    print(f"  {r[0]} | {r[1]} {r[2]} | {r[3]} | {r[5]} @ {r[4]}")

conn.close()
