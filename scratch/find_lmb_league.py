import requests

url = "https://statsapi.mlb.com/api/v1/league"
r = requests.get(url, timeout=10)
if r.status_code == 200:
    data = r.json()
    leagues = data.get("leagues", [])
    print(f"Total ligas registradas en MLB Stats API: {len(leagues)}")
    for l in leagues:
        name = l.get("name", "")
        if "mexic" in name.lower() or "lmb" in name.lower() or "summer" in name.lower() or l.get("id") in [125, 132]:
            print(f"  ID: {l.get('id'):>3} | SportID: {l.get('sport', {}).get('id')} | Code: {l.get('abbreviation'):<5} | Name: {name}")
