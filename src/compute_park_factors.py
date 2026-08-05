"""
compute_park_factors.py
Factor de parque por carreras, calculado 100% desde nuestros propios
datos de calendario -- sin red, sin FanGraphs.

Método clásico (el mismo que usa la comunidad sabermétrica desde hace
décadas): para cada equipo en una temporada, se compara el promedio de
carreras totales (local + visitante) en SUS partidos de local contra el
promedio en SUS partidos de visitante. Usar al mismo equipo como propio
control aísla el efecto del parque de la calidad ofensiva/defensiva del
equipo. Luego se promedia en una ventana de `window_years` temporadas
(3 por defecto) para reducir el ruido de una sola temporada.

Limitación conocida: esta primera versión solo calcula factor_runs.
factor_hr / factor_hr_vs_l / factor_hr_vs_r (más útiles para el
emparejamiento zurdo/derecho de la Fase 3) requieren atribuir home runs
por parque desde statcast_batted_balls -- se puede sumar después sin
tocar lo que ya hay.

Uso:
    python src/compute_park_factors.py
    python src/compute_park_factors.py --window-years 3
"""

from __future__ import annotations

import argparse

import db as db_module

MIN_GAMES_MUESTRA = 10  # menos que esto (ej. equipo que cambió de parque a mitad de año) se ignora


def _season_extremes(conn) -> tuple[int, int]:
    row = conn.execute(
        "SELECT MIN(season), MAX(season) FROM games WHERE game_type='R' AND status='Final'"
    ).fetchone()
    if not row or row[0] is None:
        raise RuntimeError("No hay temporadas de temporada regular cargadas todavía.")
    return row[0], row[1]


def compute_season_park_factors(conn, season: int) -> dict[int, float]:
    """factor_runs por venue_id para UNA temporada."""
    teams_venues = conn.execute(
        """SELECT DISTINCT home_team_id, venue_id FROM games
           WHERE season=? AND game_type='R' AND status='Final' AND venue_id IS NOT NULL""",
        (season,),
    ).fetchall()

    factors: dict[int, float] = {}
    for team_id, venue_id in teams_venues:
        # OJO: home_avg tiene que ser específico de ESTE venue, no el
        # promedio general de local del equipo. Si no, un equipo que jugó
        # un puñado de partidos "de local" en una sede especial (Serie de
        # Londres, México, Tokio, Field of Dreams, etc.) le asigna a esa
        # sede el promedio de SU ESTADIO NORMAL, y de paso nunca se filtra
        # por muestra chica porque el conteo usado era el total de local
        # del equipo (~81 partidos), no los 1-2 de esa sede puntual.
        home_avg, home_n = conn.execute(
            """SELECT AVG(home_score + away_score), COUNT(*) FROM games
               WHERE season=? AND game_type='R' AND status='Final'
                     AND home_team_id=? AND venue_id=?""",
            (season, team_id, venue_id),
        ).fetchone()
        # El promedio de visitante sí se deja general (across ~15 parques
        # distintos): esa variedad es justo lo que lo hace un buen "neutral"
        # de referencia para la fuerza ofensiva/defensiva real del equipo.
        road_avg, road_n = conn.execute(
            """SELECT AVG(home_score + away_score), COUNT(*) FROM games
               WHERE season=? AND game_type='R' AND status='Final' AND away_team_id=?""",
            (season, team_id),
        ).fetchone()

        if not home_avg or not road_avg or home_n < MIN_GAMES_MUESTRA or road_n < MIN_GAMES_MUESTRA:
            continue

        # Si dos equipos comparten venue en la misma temporada (poco común
        # pero pasa), promediamos sus factores individuales.
        factor = home_avg / road_avg
        if venue_id in factors:
            factors[venue_id] = (factors[venue_id] + factor) / 2
        else:
            factors[venue_id] = factor

    return {v: round(f, 4) for v, f in factors.items()}


def compute_rolling_park_factors(conn, end_season: int, window_years: int = 3) -> dict[int, float]:
    """Promedio de factor_runs en las temporadas [end_season-window_years+1, end_season]
    que sí tengamos cargadas (usa menos años si no hay suficiente historia)."""
    per_season = []
    for s in range(end_season - window_years + 1, end_season + 1):
        try:
            per_season.append(compute_season_park_factors(conn, s))
        except Exception:
            continue

    venues = set()
    for d in per_season:
        venues.update(d.keys())

    rolling: dict[int, float] = {}
    for v in venues:
        vals = [d[v] for d in per_season if v in d]
        if vals:
            rolling[v] = round(sum(vals) / len(vals), 4)
    return rolling


def upsert_park_factors(conn, season: int, factors: dict[int, float], source: str) -> None:
    rows = [
        {"venue_id": v, "season": season, "factor_runs": f,
         "factor_hr": None, "factor_hr_vs_l": None, "factor_hr_vs_r": None, "source": source}
        for v, f in factors.items() if v is not None  # cinturón y tirantes
    ]
    if not rows:
        return
    conn.executemany(
        """INSERT OR REPLACE INTO park_factors
           (venue_id, season, factor_runs, factor_hr, factor_hr_vs_l, factor_hr_vs_r, source)
           VALUES (:venue_id, :season, :factor_runs, :factor_hr, :factor_hr_vs_l,
                   :factor_hr_vs_r, :source)""",
        rows,
    )
    conn.commit()


def run(db_path: str = "data/mlb.db", window_years: int = 3) -> None:
    conn = db_module.get_connection(db_path)
    db_module.init_db(conn)

    min_season, max_season = _season_extremes(conn)
    source = f"computado_localmente_ventana{window_years}a"

    for season in range(min_season, max_season + 1):
        factors = compute_rolling_park_factors(conn, season, window_years=window_years)
        upsert_park_factors(conn, season, factors, source)
        print(f"[{season}] park factors calculados para {len(factors)} venues "
              f"(ventana de {window_years} años)")

    conn.close()
    print("\nPark factors listos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula park factors desde datos propios")
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--window-years", type=int, default=3)
    args = parser.parse_args()
    run(args.db_path, args.window_years)
