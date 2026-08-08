"""
compute_park_factors.py
Factor de parque por carreras Y home runs, calculado 100% desde nuestros
propios datos -- sin red, sin FanGraphs.

Método clásico: para cada equipo en una temporada, se compara el rate
en SUS partidos de local contra el rate en SUS partidos de visitante.
Luego se promedia en una ventana de `window_years` temporadas (3 por
defecto) para reducir el ruido de una sola temporada.

Calcula:
  - factor_runs:    (total runs/game en casa) / (total runs/game de visita)
  - factor_hr:      (HR rate en casa) / (HR rate de visita), usando statcast
  - factor_hr_vs_l: mismo que factor_hr pero solo bateadores zurdos
  - factor_hr_vs_r: mismo que factor_hr pero solo bateadores derechos

Uso:
    python src/compute_park_factors.py
    python src/compute_park_factors.py --window-years 3
"""

from __future__ import annotations

import argparse

import db as db_module

MIN_GAMES_MUESTRA = 10   # para factor_runs
MIN_BBE_MUESTRA = 50     # para factor_hr (batted ball events mínimos)


def _season_extremes(conn) -> tuple[int, int]:
    row = conn.execute(
        "SELECT MIN(season), MAX(season) FROM games WHERE game_type='R' AND status='Final'"
    ).fetchone()
    if not row or row[0] is None:
        raise RuntimeError("No hay temporadas de temporada regular cargadas todavía.")
    return row[0], row[1]


# ---------------------------------------------------------------------------
# factor_runs (método original, sin cambios)
# ---------------------------------------------------------------------------

def compute_season_park_factors(conn, season: int) -> dict[int, float]:
    """factor_runs por venue_id para UNA temporada."""
    teams_venues = conn.execute(
        """SELECT DISTINCT home_team_id, venue_id FROM games
           WHERE season=? AND game_type='R' AND status='Final' AND venue_id IS NOT NULL""",
        (season,),
    ).fetchall()

    factors: dict[int, float] = {}
    for team_id, venue_id in teams_venues:
        home_avg, home_n = conn.execute(
            """SELECT AVG(home_score + away_score), COUNT(*) FROM games
               WHERE season=? AND game_type='R' AND status='Final'
                     AND home_team_id=? AND venue_id=?""",
            (season, team_id, venue_id),
        ).fetchone()
        road_avg, road_n = conn.execute(
            """SELECT AVG(home_score + away_score), COUNT(*) FROM games
               WHERE season=? AND game_type='R' AND status='Final' AND away_team_id=?""",
            (season, team_id),
        ).fetchone()

        if not home_avg or not road_avg or home_n < MIN_GAMES_MUESTRA or road_n < MIN_GAMES_MUESTRA:
            continue

        factor = home_avg / road_avg
        if venue_id in factors:
            factors[venue_id] = (factors[venue_id] + factor) / 2
        else:
            factors[venue_id] = factor

    return {v: round(f, 4) for v, f in factors.items()}


# ---------------------------------------------------------------------------
# factor_hr y splits L/R (NUEVO)
# ---------------------------------------------------------------------------

def _hr_rate_query(season: int, team_id: int, venue_id: int | None,
                   is_home: bool, stand_filter: str | None = None) -> tuple[str, list]:
    """Construye query para HR rate = HR_count / BBE_count desde statcast.

    Si is_home=True, filtra statcast BBE en partidos donde team_id es local
    en el venue_id dado.
    Si is_home=False, filtra statcast BBE en todos los partidos de visita
    del team_id (across all venues).
    """
    if is_home:
        sql = """
            SELECT COUNT(*) as bbe,
                   SUM(CASE WHEN s.events = 'home_run' THEN 1 ELSE 0 END) as hr
            FROM statcast_batted_balls s
            JOIN games g ON g.game_pk = s.game_pk
            WHERE g.season = ? AND g.game_type = 'R' AND g.status = 'Final'
                  AND g.home_team_id = ? AND g.venue_id = ?
        """
        params: list = [season, team_id, venue_id]
    else:
        sql = """
            SELECT COUNT(*) as bbe,
                   SUM(CASE WHEN s.events = 'home_run' THEN 1 ELSE 0 END) as hr
            FROM statcast_batted_balls s
            JOIN games g ON g.game_pk = s.game_pk
            WHERE g.season = ? AND g.game_type = 'R' AND g.status = 'Final'
                  AND g.away_team_id = ?
        """
        params = [season, team_id]

    if stand_filter:
        sql += " AND s.stand = ?"
        params.append(stand_filter)

    return sql, params


def compute_season_hr_park_factors(
    conn, season: int, stand_filter: str | None = None
) -> dict[int, float]:
    """factor_hr (o factor_hr_vs_l / factor_hr_vs_r) por venue_id para UNA temporada.

    stand_filter: None = all, 'L' = lefty batters only, 'R' = righty batters only.
    """
    teams_venues = conn.execute(
        """SELECT DISTINCT home_team_id, venue_id FROM games
           WHERE season=? AND game_type='R' AND status='Final' AND venue_id IS NOT NULL""",
        (season,),
    ).fetchall()

    factors: dict[int, float] = {}
    for team_id, venue_id in teams_venues:
        # HR rate at home (specific venue)
        sql_h, params_h = _hr_rate_query(season, team_id, venue_id, is_home=True, stand_filter=stand_filter)
        home_row = conn.execute(sql_h, params_h).fetchone()
        home_bbe, home_hr = (home_row[0] or 0), (home_row[1] or 0)

        # HR rate on the road (all away venues)
        sql_r, params_r = _hr_rate_query(season, team_id, None, is_home=False, stand_filter=stand_filter)
        road_row = conn.execute(sql_r, params_r).fetchone()
        road_bbe, road_hr = (road_row[0] or 0), (road_row[1] or 0)

        if home_bbe < MIN_BBE_MUESTRA or road_bbe < MIN_BBE_MUESTRA:
            continue
        if home_hr == 0 and road_hr == 0:
            continue

        home_rate = home_hr / home_bbe
        road_rate = road_hr / road_bbe

        if road_rate == 0:
            continue  # evitar div/0

        factor = home_rate / road_rate

        if venue_id in factors:
            factors[venue_id] = (factors[venue_id] + factor) / 2
        else:
            factors[venue_id] = factor

    return {v: round(f, 4) for v, f in factors.items()}


# ---------------------------------------------------------------------------
# Rolling (ventana multi-año)
# ---------------------------------------------------------------------------

def _rolling_average(per_season_list: list[dict[int, float]]) -> dict[int, float]:
    """Promedia factores de múltiples temporadas por venue_id."""
    venues = set()
    for d in per_season_list:
        venues.update(d.keys())

    rolling: dict[int, float] = {}
    for v in venues:
        vals = [d[v] for d in per_season_list if v in d]
        if vals:
            rolling[v] = round(sum(vals) / len(vals), 4)
    return rolling


def compute_rolling_park_factors(conn, end_season: int, window_years: int = 3) -> dict[int, float]:
    """Promedio de factor_runs en ventana rolling."""
    per_season = []
    for s in range(end_season - window_years + 1, end_season + 1):
        try:
            per_season.append(compute_season_park_factors(conn, s))
        except Exception:
            continue
    return _rolling_average(per_season)


def compute_rolling_hr_factors(
    conn, end_season: int, window_years: int = 3, stand_filter: str | None = None
) -> dict[int, float]:
    """Promedio de factor_hr (o split L/R) en ventana rolling."""
    per_season = []
    for s in range(end_season - window_years + 1, end_season + 1):
        try:
            per_season.append(compute_season_hr_park_factors(conn, s, stand_filter=stand_filter))
        except Exception:
            continue
    return _rolling_average(per_season)


# ---------------------------------------------------------------------------
# Upsert y orquestación
# ---------------------------------------------------------------------------

def upsert_park_factors(conn, season: int, runs_factors: dict[int, float],
                        hr_factors: dict[int, float],
                        hr_vs_l: dict[int, float],
                        hr_vs_r: dict[int, float],
                        source: str) -> None:
    """Inserta o reemplaza park factors para la temporada dada."""
    all_venues = set(runs_factors) | set(hr_factors) | set(hr_vs_l) | set(hr_vs_r)
    rows = [
        {
            "venue_id": v,
            "season": season,
            "factor_runs": runs_factors.get(v),
            "factor_hr": hr_factors.get(v),
            "factor_hr_vs_l": hr_vs_l.get(v),
            "factor_hr_vs_r": hr_vs_r.get(v),
            "source": source,
        }
        for v in all_venues if v is not None
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
        runs_f = compute_rolling_park_factors(conn, season, window_years=window_years)
        hr_f = compute_rolling_hr_factors(conn, season, window_years=window_years, stand_filter=None)
        hr_l = compute_rolling_hr_factors(conn, season, window_years=window_years, stand_filter="L")
        hr_r = compute_rolling_hr_factors(conn, season, window_years=window_years, stand_filter="R")

        upsert_park_factors(conn, season, runs_f, hr_f, hr_l, hr_r, source)

        n_hr = len(hr_f)
        n_l = len(hr_l)
        n_r = len(hr_r)
        print(f"[{season}] park factors: {len(runs_f)} venues runs, "
              f"{n_hr} HR, {n_l} HR_vs_L, {n_r} HR_vs_R "
              f"(ventana de {window_years} años)")

    conn.close()
    print("\nPark factors listos (runs + HR + splits L/R).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula park factors desde datos propios")
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--window-years", type=int, default=3)
    args = parser.parse_args()
    run(args.db_path, args.window_years)
