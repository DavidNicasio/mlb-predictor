import requests

# Consultar calendario LMB (leagueId=125, sportId=23)
url = "https://statsapi.mlb.com/api/v1/schedule"
params = {
    "sportId": 23,
    "leagueId": 125,
    "date": "2026-08-08",
    "hydrate": "team,probablePitcher",
}

r = requests.get(url, params=params).json()
print("Calendario LMB para 2026-08-08:")
dates = r.get("dates", [])
if not dates:
    print("  No hay fechas registradas para hoy. Probemos buscando la última fecha jugada...")
    params_range = {
        "sportId": 23,
        "leagueId": 125,
        "startDate": "2026-08-01",
        "endDate": "2026-08-08",
        "hydrate": "team",
    }
    r2 = requests.get(url, params=params_range).json()
    for d in r2.get("dates", []):
        print(f"  Fecha {d.get('date')}: {len(d.get('games', []))} partidos LMB")
        for g in d.get("games", [])[:3]:
            away = g.get("teams", {}).get("away", {}).get("team", {}).get("name")
            home = g.get("teams", {}).get("home", {}).get("team", {}).get("name")
            print(f"    {away} @ {home} (status={g.get('status',{}).get('detailedState')})")
else:
    for d in dates:
        print(f"  {d.get('date')}: {len(d.get('games', []))} partidos")
        for g in d.get("games", []):
            away = g.get("teams", {}).get("away", {}).get("team", {}).get("name")
            home = g.get("teams", {}).get("home", {}).get("team", {}).get("name")
            print(f"    {away} @ {home}")
