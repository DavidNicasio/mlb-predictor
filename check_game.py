import sqlite3
conn = sqlite3.connect("data/mlb.db")
rows = conn.execute("SELECT game_pk, status, home_score, away_score FROM games WHERE game_date='2026-08-01'").fetchall()
for r in rows:
    print(r)
conn.close()
