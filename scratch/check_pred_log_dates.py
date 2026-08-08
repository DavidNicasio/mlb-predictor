import sqlite3

conn = sqlite3.connect("data/mlb.db")

query = """
    SELECT 
        g.season, 
        MIN(g.game_date) AS min_date, 
        MAX(g.game_date) AS max_date, 
        COUNT(DISTINCT p.game_pk) AS n_games
    FROM predictions_log p 
    JOIN games g ON g.game_pk = p.game_pk 
    GROUP BY g.season 
    ORDER BY g.season
"""

rows = conn.execute(query).fetchall()
print("Desglose de predictions_log por temporada:")
for r in rows:
    print(f"  Temporada {r[0]}: {r[3]} partidos (del {r[1]} al {r[2]})")

t_15_23 = conn.execute("""
    SELECT COUNT(DISTINCT p.game_pk) 
    FROM predictions_log p 
    JOIN games g ON g.game_pk = p.game_pk 
    WHERE g.season BETWEEN 2015 AND 2023
""").fetchone()[0]

print(f"\nTotal partidos de 2015-2023 en predictions_log: {t_15_23}")
conn.close()
