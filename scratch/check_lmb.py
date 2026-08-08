import requests

r = requests.get("https://statsapi.mlb.com/api/v1/sports").json()
print("Sports list in Stats API:")
for s in r.get("sports", []):
    print(f"  id={s.get('id')}: {s.get('name')} ({s.get('code')})")

r_l = requests.get("https://statsapi.mlb.com/api/v1/league").json()
print("\nLeagues list in Stats API:")
for l in r_l.get("leagues", []):
    if "mexic" in l.get("name", "").lower() or "lmb" in l.get("name", "").lower():
        print(f"  id={l.get('id')}: {l.get('name')} (sportId={l.get('sport',{}).get('id')})")
