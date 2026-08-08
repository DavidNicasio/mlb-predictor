"""Check if the MLB API provides inning-by-inning linescore data (for F5 model)."""
import requests
import json

# Use a known completed game
game_pk = 822868
url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"
r = requests.get(url).json()

print("Linescore top-level keys:", list(r.keys()))
innings = r.get("innings", [])
print(f"Innings available: {len(innings)}")

for i in innings[:6]:
    home_runs = i.get("home", {}).get("runs")
    away_runs = i.get("away", {}).get("runs")
    print(f"  Inning {i.get('num')}: home={home_runs}, away={away_runs}")

# Calculate F5 score
if len(innings) >= 5:
    home_f5 = sum(i.get("home", {}).get("runs", 0) or 0 for i in innings[:5])
    away_f5 = sum(i.get("away", {}).get("runs", 0) or 0 for i in innings[:5])
    print(f"\nF5 score: away={away_f5}, home={home_f5}, total_f5={away_f5 + home_f5}")

# Also check if linescore is embedded in schedule hydration
url2 = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=2026-08-07&hydrate=weather,team,linescore,probablePitcher"
r2 = requests.get(url2).json()
dates = r2.get("dates", [])
if dates:
    game0 = dates[0].get("games", [{}])[0]
    ls = game0.get("linescore", {})
    print(f"\nLinescore in schedule hydrate keys: {list(ls.keys())}")
    hy_innings = ls.get("innings", [])
    print(f"Hydrated innings: {len(hy_innings)}")
    for i in hy_innings[:5]:
        print(f"  Inning {i.get('num')}: home={i.get('home',{}).get('runs')}, away={i.get('away',{}).get('runs')}")
