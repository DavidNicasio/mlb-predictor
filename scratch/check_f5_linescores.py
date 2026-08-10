import sqlite3
import pandas as pd

conn = sqlite3.connect("data/mlb.db")

df_f5 = pd.read_sql_query("""
    SELECT g.season,
           COUNT(DISTINCT g.game_pk) AS total_games,
           COUNT(DISTINCT f5.game_pk) AS f5_games
    FROM games g
    LEFT JOIN (
        SELECT game_pk
        FROM game_linescore
        WHERE inning <= 5
        GROUP BY game_pk
        HAVING COUNT(DISTINCT inning) = 5
    ) f5 ON f5.game_pk = g.game_pk
    WHERE g.game_type = 'R' AND g.status = 'Final'
    GROUP BY g.season
    ORDER BY g.season ASC
""", conn)

print("=== COBERTURA DE PRIMERAS 5 ENTRADAS (F5) EN LINESCORE ===")
print(df_f5.to_string(index=False))

conn.close()
