"""
fetch_logos.py
Descarga y almacena en caché local los logos en PNG de los 30 equipos de la MLB.
"""

from __future__ import annotations

from pathlib import Path
import requests

import db

# Mapeo especial de abreviación MLB a código de logo de ESPN
ESPN_LOGO_MAP = {
    "AZ": "ari",
    "ARI": "ari",
    "ATH": "oak",
    "OAK": "oak",
    "CWS": "chw",
    "WSH": "was",
    "WAS": "was",
    "SD": "sd",
    "SDP": "sd",
    "SF": "sf",
    "SFG": "sf",
    "KC": "kc",
    "KCR": "kc",
    "TB": "tb",
    "TBR": "tb",
}

LOGOS_DIR = Path("assets/logos")


def download_logos(db_path: str = "data/mlb.db") -> None:
    LOGOS_DIR.mkdir(parents=True, exist_ok=True)
    conn = db.get_connection(db_path)
    teams = conn.execute("SELECT team_id, abbreviation, name FROM teams").fetchall()
    conn.close()

    print(f"Verificando {len(teams)} logos de equipos MLB...")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    for team_id, abbr, name in teams:
        id_png = LOGOS_DIR / f"{team_id}.png"
        abbr_png = LOGOS_DIR / f"{abbr}.png"

        if id_png.exists() and abbr_png.exists():
            continue

        espn_code = ESPN_LOGO_MAP.get(abbr, abbr.lower())
        url = f"https://a.espncdn.com/i/teamlogos/mlb/500/{espn_code}.png"

        try:
            r = session.get(url, timeout=5)
            if r.status_code == 200 and len(r.content) > 1000:
                id_png.write_bytes(r.content)
                abbr_png.write_bytes(r.content)
                print(f"  [OK] Logo guardado para {name} ({abbr})")
            else:
                print(f"  [AVISO] No se pudo obtener logo para {name} ({abbr}) desde {url}")
        except Exception as err:
            print(f"  [ERROR] Fallo descargando logo de {abbr}: {err}")


if __name__ == "__main__":
    download_logos()
