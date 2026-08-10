import requests

url = "https://statsapi.mlb.com/api/v1/schedule"
params = {
    "sportId": 1,
    "startDate": "2024-05-01",
    "endDate": "2024-05-03",
    "hydrate": "weather",
}

r = requests.get(url, params=params).json()

print("Probando API oficial de MLB con hydrate=weather para mayo de 2024:")
for date_item in r.get("dates", []):
    print(f"\nFecha: {date_item.get('date')}")
    for g in date_item.get("games", [])[:3]:
        w = g.get("weather", {})
        print(f"  GamePK: {g.get('gamePk')} | Temp: {w.get('temp')} | Wind: {w.get('wind')} | Condition: {w.get('condition')}")
