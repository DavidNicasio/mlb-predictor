"""Quick inspection of weather data coverage and linescore availability."""
import sqlite3

conn = sqlite3.connect("data/mlb.db")

# Total final regular-season games
t = conn.execute("SELECT COUNT(*) FROM games WHERE game_type='R' AND status='Final'").fetchone()[0]

# Weather temp coverage
r = conn.execute("SELECT COUNT(*) FROM games WHERE weather_temp IS NOT NULL AND game_type='R' AND status='Final'").fetchone()[0]
print(f"Weather temp coverage: {r}/{t} ({r/t*100:.1f}%)")

# Weather wind coverage
w = conn.execute("SELECT COUNT(*) FROM games WHERE weather_wind IS NOT NULL AND weather_wind != '' AND game_type='R' AND status='Final'").fetchone()[0]
print(f"Weather wind coverage: {w}/{t} ({w/t*100:.1f}%)")

# Sample wind values
print("\nSample weather_wind values:")
rows = conn.execute("SELECT DISTINCT weather_wind FROM games WHERE weather_wind IS NOT NULL AND weather_wind != '' LIMIT 30").fetchall()
for x in rows:
    print(f"  '{x[0]}'")

# Sample weather_temp
print("\nSample weather_temp range:")
row = conn.execute("SELECT MIN(weather_temp), MAX(weather_temp), AVG(weather_temp) FROM games WHERE weather_temp IS NOT NULL AND game_type='R'").fetchone()
print(f"  Min: {row[0]}, Max: {row[1]}, Avg: {row[2]:.1f}")

# Check if linescore data is stored
print("\n--- Linescore / inning-by-inning data ---")
# Check game_features table columns
cols = conn.execute("PRAGMA table_info(game_features)").fetchall()
f5_cols = [c[1] for c in cols if 'f5' in c[1].lower() or 'inning' in c[1].lower() or 'linescore' in c[1].lower()]
print(f"F5/inning columns in game_features: {f5_cols}")

# Check if extract_schedule stores linescore
print("\nChecking tables for linescore:")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(f"Tables: {[t[0] for t in tables]}")

# Check boxscore_pitching for starter outs (proxy for F5)
sample = conn.execute("""
    SELECT bp.game_pk, bp.player_id, bp.is_starter, bp.outs, bp.r, bp.er
    FROM boxscore_pitching bp
    WHERE bp.is_starter = 1
    LIMIT 5
""").fetchall()
print(f"\nStarter boxscore samples (game_pk, player_id, is_starter, outs, r, er):")
for s in sample:
    print(f"  {s}")

# Count starters with >= 15 outs (5+ innings)
c5 = conn.execute("""
    SELECT COUNT(*) FROM boxscore_pitching
    WHERE is_starter = 1 AND outs IS NOT NULL
""").fetchone()[0]
print(f"\nStarters with outs data: {c5}")

# Check weather_condition values
print("\nDistinct weather conditions:")
conds = conn.execute("SELECT DISTINCT weather_condition FROM games WHERE weather_condition IS NOT NULL AND weather_condition != '' LIMIT 20").fetchall()
for c in conds:
    print(f"  '{c[0]}'")

# Check game_features columns that include weather
gf_cols = [c[1] for c in cols]
weather_in_gf = [c for c in gf_cols if 'weather' in c.lower() or 'temp' in c.lower() or 'wind' in c.lower()]
print(f"\nWeather columns in game_features: {weather_in_gf}")

# Check statcast for HR data per venue
hr_count = conn.execute("SELECT COUNT(*) FROM statcast_batted_balls WHERE events='home_run'").fetchone()[0]
print(f"\nHome runs in statcast_batted_balls: {hr_count}")

# Seasons coverage
seasons = conn.execute("SELECT season, COUNT(*) FROM games WHERE game_type='R' AND status='Final' GROUP BY season ORDER BY season").fetchall()
print("\nGames per season:")
for s in seasons:
    print(f"  {s[0]}: {s[1]}")

conn.close()
