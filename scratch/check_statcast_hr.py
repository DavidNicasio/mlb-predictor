"""Check statcast data for park factor HR calculations."""
import sqlite3

conn = sqlite3.connect("data/mlb.db")

# HR data by batter hand
rows = conn.execute("""
    SELECT s.stand, COUNT(*) as n,
           SUM(CASE WHEN s.events = 'home_run' THEN 1 ELSE 0 END) as hr
    FROM statcast_batted_balls s
    GROUP BY s.stand
""").fetchall()
print("Statcast HR by batter hand:")
for x in rows:
    print(f"  stand={x[0]}: {x[1]} BBE, {x[2]} HR")

# Can we join to games to get venue?
sample = conn.execute("""
    SELECT s.game_pk, g.venue_id, g.venue_name, s.stand, s.events
    FROM statcast_batted_balls s
    JOIN games g ON g.game_pk = s.game_pk
    WHERE s.events = 'home_run'
    LIMIT 5
""").fetchall()
print("\nSample HR with venue:")
for x in sample:
    print(f"  game_pk={x[0]} venue_id={x[1]} venue={x[2]} stand={x[3]} event={x[4]}")

# Count HR by venue and season
venue_hr = conn.execute("""
    SELECT g.season, g.venue_id, g.venue_name,
           COUNT(*) as total_bbe,
           SUM(CASE WHEN s.events = 'home_run' THEN 1 ELSE 0 END) as hr_count
    FROM statcast_batted_balls s
    JOIN games g ON g.game_pk = s.game_pk
    WHERE g.game_type = 'R' AND g.status = 'Final'
    GROUP BY g.season, g.venue_id
    ORDER BY g.season DESC, hr_count DESC
    LIMIT 20
""").fetchall()
print("\nHR by venue (top 20 most recent):")
for v in venue_hr:
    print(f"  {v[0]} venue_id={v[1]} ({v[2]}): {v[3]} BBE, {v[4]} HR ({v[4]/v[3]*100:.1f}% HR rate)")

# Check if game_pk is populated in statcast
null_gpk = conn.execute("SELECT COUNT(*) FROM statcast_batted_balls WHERE game_pk IS NULL").fetchone()[0]
total_sbb = conn.execute("SELECT COUNT(*) FROM statcast_batted_balls").fetchone()[0]
print(f"\nStatcast records: {total_sbb} total, {null_gpk} with NULL game_pk")

# Seasons coverage in statcast
seasons_sc = conn.execute("""
    SELECT SUBSTR(game_date, 1, 4) as yr, COUNT(*) 
    FROM statcast_batted_balls 
    GROUP BY yr ORDER BY yr
""").fetchall()
print("\nStatcast by year:")
for s in seasons_sc:
    print(f"  {s[0]}: {s[1]} batted balls")

# Check HR by batter hand and venue (for split park factors)
split_hr = conn.execute("""
    SELECT g.venue_id, s.stand,
           COUNT(*) as bbe,
           SUM(CASE WHEN s.events = 'home_run' THEN 1 ELSE 0 END) as hr
    FROM statcast_batted_balls s
    JOIN games g ON g.game_pk = s.game_pk
    WHERE g.game_type = 'R' AND g.season = 2024
    GROUP BY g.venue_id, s.stand
    HAVING bbe >= 100
    ORDER BY hr DESC
    LIMIT 15
""").fetchall()
print("\n2024 HR by venue+hand (top 15):")
for v in split_hr:
    print(f"  venue_id={v[0]} stand={v[1]}: {v[2]} BBE, {v[3]} HR ({v[3]/v[2]*100:.2f}%)")

conn.close()
