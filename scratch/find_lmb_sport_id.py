import requests

url = "https://statsapi.mlb.com/api/v1/sports"
r = requests.get(url, timeout=10)
if r.status_code == 200:
    data = r.json()
    sports = data.get("sports", [])
    print("=== DEPORTES Y LIGAS DISPONIBLES EN MLB STATS API ===")
    for s in sports:
        print(f"ID: {s.get('id'):>2} | Code: {s.get('code'):<6} | Name: {s.get('name')}")
