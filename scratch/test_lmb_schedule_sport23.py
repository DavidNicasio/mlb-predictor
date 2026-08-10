import requests

url = "https://statsapi.mlb.com/api/v1/schedule"
params = {
    "sportId": 23,
    "startDate": "2024-05-01",
    "endDate": "2024-05-07",
    "hydrate": "linescore,boxscore",
}
r = requests.get(url, params=params, timeout=10)
print("Status:", r.status_code)
if r.status_code == 200:
    data = r.json()
    dates = data.get("dates", [])
    total_g = sum(len(d.get("games", [])) for d in dates)
    print(f"Total partidos en sportId=23 entre 2024-05-01 y 2024-05-07: {total_g}")
    for d in dates:
        print(f"\nFecha: {d.get('date')}")
        for g in d.get("games", []):
            league_id = g.get("teams", {}).get("home", {}).get("team", {}).get("league", {}).get("id")
            away_n = g.get("teams", {}).get("away", {}).get("team", {}).get("name")
            home_n = g.get("teams", {}).get("home", {}).get("team", {}).get("name")
            print(f"  - gamePk {g.get('gamePk')}: {away_n} @ {home_n} (LeagueId: {league_id})")
