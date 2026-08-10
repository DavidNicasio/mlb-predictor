import requests
import sqlite3

print("=== TAREA 1: INVESTIGACIÓN DE ENDPOINTS DE ALINEACIÓN EN VIVO EN MLB STATS API ===")

# Probar endpoint de calendario con hydrate=lineups
url_sched = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=2024-06-15&endDate=2024-06-15&hydrate=lineups,boxscore"
r_sched = requests.get(url_sched, timeout=10)
if r_sched.status_code == 200:
    data = r_sched.json()
    dates = data.get("dates", [])
    if dates and dates[0].get("games"):
        sample_g = dates[0]["games"][0]
        game_pk = sample_g.get("gamePk")
        print(f"Partido de prueba: gamePk {game_pk} ({sample_g.get('teams', {}).get('away', {}).get('team', {}).get('name')} @ {sample_g.get('teams', {}).get('home', {}).get('team', {}).get('name')})")

        # Checar si hydrate=lineups trajo el bloque lineups
        lineups = sample_g.get("lineups")
        print(f"  ¿Tiene objeto 'lineups' en schedule?: {lineups is not None}")
        if lineups:
            print("  Contenido lineups:", lineups)

        # Checar boxscore live / API boxscore
        url_box = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        r_box = requests.get(url_box, timeout=10)
        if r_box.status_code == 200:
            box_data = r_box.json()
            home_players = box_data.get("teams", {}).get("home", {}).get("players", {})
            sample_batting_orders = []
            for pid, pinfo in list(home_players.items())[:10]:
                border = pinfo.get("battingOrder")
                if border:
                    sample_batting_orders.append((pinfo.get("person", {}).get("fullName"), border))
            print(f"  Ejemplo de battingOrder en boxscore (Home Team): {sample_batting_orders[:5]}")

print("\n=== TAREA 2: INSPECCIÓN DE MUESTRA HISTÓRICA EN data/mlb.db ===")
conn = sqlite3.connect("data/mlb.db")
cols = [r[1] for r in conn.execute("PRAGMA table_info(boxscore_batting)").fetchall()]
print(f"Columnas actuales en boxscore_batting: {cols}")

# Checar si un boxscore guardado en DB tiene datos de jugadores ordenados o si requerimos extraer battingOrder del JSON crudo
sample_pk = conn.execute("SELECT game_pk FROM games WHERE status='Final' AND season=2024 LIMIT 1").fetchone()[0]
print(f"Analizando game_pk {sample_pk} en sqlite...")
rows = conn.execute("""
    SELECT player_name, bats, ab, h, hr, bb
    FROM boxscore_batting
    WHERE game_pk=? AND team_id=(SELECT home_team_id FROM games WHERE game_pk=?)
""", (sample_pk, sample_pk)).fetchall()

print(f"Filas de bateadores registradas para home_team en game_pk {sample_pk} ({len(rows)} bateadores):")
for r in rows[:10]:
    print("  -", r)

conn.close()
