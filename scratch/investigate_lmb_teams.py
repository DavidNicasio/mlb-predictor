import requests

url_sport = "https://statsapi.mlb.com/api/v1/teams"
r_sport23 = requests.get(url_sport, params={"sportId": 23}).json()
teams_sport23 = r_sport23.get("teams", [])
print(f"Total equipos con sportId=23: {len(teams_sport23)}")

r_lmb125 = requests.get(url_sport, params={"leagueId": 125}).json()
teams_lmb125 = r_lmb125.get("teams", [])
print(f"\nTotal equipos filtrados con leagueId=125 (LMB): {len(teams_lmb125)}")

print("\n--- Equipos oficiales en leagueId=125 (LMB) ---")
for t in teams_lmb125:
    league_info = t.get("league", {})
    print(f"  ID: {t.get('id'):<5} | Nombre: {t.get('name'):<35} | Abbr: {t.get('abbreviation')}")

print("\n--- Verificación de Charros de Jalisco ---")
charros = [t for t in teams_sport23 if "charro" in t.get("name", "").lower()]
for c in charros:
    print(f"  ID: {c.get('id')} | Nombre: {c.get('name')} | League: {c.get('league', {}).get('name')} (leagueId={c.get('league',{}).get('id')})")
