import sqlite3
import pandas as pd

conn = sqlite3.connect("data/mlb.db")

df = pd.read_sql_query("""
    SELECT season,
           COUNT(*) AS total_games,
           SUM(CASE WHEN weather_temp IS NOT NULL THEN 1 ELSE 0 END) AS count_temp,
           SUM(CASE WHEN weather_wind IS NOT NULL AND weather_wind != '' THEN 1 ELSE 0 END) AS count_wind
    FROM games
    WHERE game_type = 'R'
    GROUP BY season
    ORDER BY season ASC
""", conn)

print("=== COBERTURA DE CLIMA POR TEMPORADA EN GAMES ===")
print(df.to_string(index=False))

sample_winds = pd.read_sql_query("""
    SELECT weather_wind, COUNT(*) as cnt
    FROM games
    WHERE weather_wind IS NOT NULL AND weather_wind != ''
    GROUP BY weather_wind
    ORDER BY cnt DESC
    LIMIT 25
""", conn)

print("\n=== MUESTRA DE VALORES DE WEATHER_WIND ===")
print(sample_winds.to_string(index=False))

conn.close()
