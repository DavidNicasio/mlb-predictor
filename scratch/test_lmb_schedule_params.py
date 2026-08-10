import requests

# Probar varias combinaciones de sportId y leagueId para LMB en MLB Stats API
params_list = [
    {"sportId": 11, "startDate": "2024-05-01", "endDate": "2024-05-07"},
    {"sportId": 16, "startDate": "2024-05-01", "endDate": "2024-05-07"},
    {"leagueId": 125, "startDate": "2024-05-01", "endDate": "2024-05-07"},
    {"sportIds": "11", "startDate": "2024-05-01", "endDate": "2024-05-07"},
    {"leagueIds": "125", "startDate": "2024-05-01", "endDate": "2024-05-07"},
]

for p in params_list:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    r = requests.get(url, params=p, timeout=10)
    print(f"Parametros {p}: Status {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        dates = data.get("dates", [])
        total_g = sum(len(d.get("games", [])) for d in dates)
        print(f"   -> Encontrados {total_g} partidos LMB")
        if total_g > 0:
            sample_g = dates[0]["games"][0]
            print(f"      Ejemplo: gamePk {sample_g.get('gamePk')} | {sample_g.get('teams', {}).get('away', {}).get('team', {}).get('name')} @ {sample_g.get('teams', {}).get('home', {}).get('team', {}).get('name')}")
