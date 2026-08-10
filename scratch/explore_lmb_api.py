import requests

print("=== EXPLORANDO OFICIALMENTE LA API DE MLB STATS PARA LA LMB (leagueId=125) ===")

url_teams = "https://statsapi.mlb.com/api/v1/teams?leagueIds=125"
r_teams = requests.get(url_teams, timeout=10)
if r_teams.status_code == 200:
    data = r_teams.json()
    teams = data.get("teams", [])
    print(f"Equipos LMB encontrados en Stats API (leagueId=125): {len(teams)}")
    for t in teams[:5]:
        print(f"  - ID: {t.get('id')} | Name: {t.get('name')} | Abbr: {t.get('abbreviation')}")
else:
    print("Error consultando equipos LMB:", r_teams.status_code)

url_sched = "https://statsapi.mlb.com/api/v1/schedule?leagueId=125&startDate=2024-05-01&endDate=2024-05-07&hydrate=linescore,boxscore"
r_sched = requests.get(url_sched, timeout=10)
if r_sched.status_code == 200:
    data_s = r_sched.json()
    dates = data_s.get("dates", [])
    total_g = sum(len(d.get("games", [])) for d in dates)
    print(f"\nPartidos LMB encontrados en mayo 2024: {total_g}")
    if total_g > 0:
        sample_g = dates[0]["games"][0]
        print(f"  Ejemplo Partido: {sample_g.get('gamePk')} | {sample_g.get('teams', {}).get('away', {}).get('team', {}).get('name')} @ {sample_g.get('teams', {}).get('home', {}).get('team', {}).get('name')}")
        print(f"  Marcador: {sample_g.get('teams', {}).get('away', {}).get('score')} - {sample_g.get('teams', {}).get('home', {}).get('score')}")
        print(f"  Status: {sample_g.get('status', {}).get('detailedState')}")
        box = sample_g.get("boxscore", {})
        has_batting = "teams" in box
        print(f"  ¿Tiene Boxscore completo?: {has_batting}")
else:
    print("Error consultando calendario LMB:", r_sched.status_code)
