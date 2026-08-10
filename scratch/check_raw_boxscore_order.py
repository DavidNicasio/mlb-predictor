import requests

url = "https://statsapi.mlb.com/api/v1/game/744795/boxscore"
r = requests.get(url, timeout=10)
if r.status_code == 200:
    box = r.json()
    teams = box.get("teams", {})
    for side in ("home", "away"):
        print(f"\n=== SIDE: {side.upper()} ===")
        team_b = teams.get(side, {})
        batters_list = team_b.get("batters", [])
        players = team_b.get("players", {})

        # Filtrar solo bateadores titulares (battingOrder termina en 00: 100, 200, 300...)
        starters = []
        for pid in batters_list:
            pkey = f"ID{pid}"
            pdata = players.get(pkey, {})
            border = pdata.get("battingOrder")
            if border and border.endswith("00"):
                starters.append((border, pdata.get("person", {}).get("fullName"), pdata.get("batSide", {}).get("code")))

        starters.sort(key=lambda x: int(x[0]))
        print(f"Starters confirmados por battingOrder ({len(starters)} jugadores):")
        for s in starters:
            print(f"  Pos {int(s[0])//100}: {s[1]} (Bats: {s[2]})")
