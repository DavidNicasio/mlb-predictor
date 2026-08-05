"""
debug_boxscore_structure.py
Diagnostico puntual (no toca la base de datos): descarga el boxscore
crudo de UN partido real y muestra, para un pitcher y un bateador, tanto
las keys de nivel superior como las de 'person', para confirmar
EXACTAMENTE donde viven pitchHand/batSide en la respuesta real de la API.

Uso:
    python src/debug_boxscore_structure.py 777491
"""

import json
import sys

sys.path.insert(0, "src") if "src" not in sys.path else None
import extract_schedule

game_pk = int(sys.argv[1]) if len(sys.argv) > 1 else 777491
print(f"Descargando boxscore real de game_pk={game_pk}...\n")
box = extract_schedule.fetch_boxscore(game_pk)

home_players = box["teams"]["home"]["players"]

first_pitcher_key = None
first_batter_key = None
for key, pdata in home_players.items():
    stats = pdata.get("stats", {})
    if not first_pitcher_key and stats.get("pitching", {}).get("inningsPitched") not in (None, "0.0"):
        first_pitcher_key = key
    if not first_batter_key and stats.get("batting", {}).get("atBats") is not None:
        first_batter_key = key

for label, key in (("PITCHER", first_pitcher_key), ("BATEADOR", first_batter_key)):
    print(f"=== {label} ({key}) ===")
    pdata = home_players.get(key, {})
    print("Keys de nivel superior del jugador:", list(pdata.keys()))
    person = pdata.get("person", {})
    print("Keys dentro de 'person':", list(person.keys()))
    print("pdata.get('pitchHand') ->", pdata.get("pitchHand"))
    print("pdata.get('batSide')   ->", pdata.get("batSide"))
    print("person.get('pitchHand') ->", person.get("pitchHand"))
    print("person.get('batSide')   ->", person.get("batSide"))
    print()
