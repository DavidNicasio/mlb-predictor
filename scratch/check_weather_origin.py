import sqlite3

conn = sqlite3.connect("data/mlb.db")
total_games = conn.execute("SELECT COUNT(*) FROM games WHERE status='Final'").fetchone()[0]
weather_games = conn.execute("SELECT COUNT(*) FROM games WHERE status='Final' AND weather_temp IS NOT NULL").fetchone()[0]

print(f"Total partidos Finales MLB en DB: {total_games}")
print(f"Partidos con weather_temp cargado: {weather_games}")

# Ver muestra por temporada
rows = conn.execute("""
    SELECT season, COUNT(*) AS total,
           SUM(CASE WHEN weather_temp IS NOT NULL THEN 1 ELSE 0 END) AS with_weather
    FROM games
    WHERE status='Final'
    GROUP BY season
    ORDER BY season ASC
""").fetchall()

print("\nDesglose de clima por temporada:")
for r in rows:
    print(f"  Temporada {r[0]}: {r[1]} partidos | {r[2]} con clima ({r[2]/r[1]*100:.1f}%)")

conn.close()
