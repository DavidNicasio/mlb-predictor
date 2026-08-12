import sqlite3

conn = sqlite3.connect("data/lmb.db")
total_lmb = conn.execute("SELECT COUNT(*) FROM games WHERE status='Final' AND league='LMB'").fetchone()[0]
print(f"Total partidos Finales LMB en data/lmb.db: {total_lmb}")

rows = conn.execute("""
    SELECT season, MIN(game_date), MAX(game_date), COUNT(*)
    FROM games
    WHERE status='Final' AND league='LMB'
    GROUP BY season
    ORDER BY season ASC
""").fetchall()

print("\nDesglose por temporada en LMB:")
for r in rows:
    print(f"  Temporada {r[0]}: {r[3]} partidos (Desde {r[1]} hasta {r[2]})")

conn.close()
