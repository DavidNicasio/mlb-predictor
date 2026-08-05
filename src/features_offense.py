"""
features_offense.py
wOBA rolling por equipo, con splits vs lanzador zurdo/derecho.

Fuente: statcast_batted_balls. A pesar del nombre de la tabla, en realidad
guarda el evento terminal de CADA aparición al plato (walk, strikeout,
single, home_run, etc.), no solo bolas puestas en juego -- Statcast marca
el campo `events` al final de cada turno al bate, tenga o no bateo en
juego. Eso es justo lo que necesitamos: cobertura completa de apariciones,
con el lado del bateador (stand) y la mano del lanzador (p_throws) de
cada una, para poder partir el wOBA por mano del pitcher rival.

Para saber a qué EQUIPO perteneció cada bateador en un partido dado
(statcast no trae team_id), se cruza con boxscore_batting por
(game_pk, batter_id = player_id).
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import metrics

# Clasificación de `events` de Statcast a categorías de wOBA.
BB_EVENTS = {"walk"}
IBB_EVENTS = {"intent_walk"}
HBP_EVENTS = {"hit_by_pitch"}
SINGLE_EVENTS = {"single"}
DOUBLE_EVENTS = {"double"}
TRIPLE_EVENTS = {"triple"}
HR_EVENTS = {"home_run"}
SF_EVENTS = {"sac_fly", "sac_fly_double_play"}
# PA que NO cuentan como at-bat (para no inflar el denominador de AB)
NON_AB_EVENTS = (BB_EVENTS | IBB_EVENTS | HBP_EVENTS | SF_EVENTS
                 | {"sac_bunt", "sac_bunt_double_play", "catcher_interf"})

# Punto de estabilización aproximado para wOBA agregado de equipo (PA).
WOBA_STABILIZATION_K = 200


def _woba_components_query(vs_hand: str | None) -> str:
    hand_filter = "AND sb.p_throws = :vs_hand" if vs_hand else ""
    return f"""
        SELECT
            SUM(CASE WHEN sb.events = 'walk' THEN 1 ELSE 0 END) AS bb,
            SUM(CASE WHEN sb.events = 'intent_walk' THEN 1 ELSE 0 END) AS ibb,
            SUM(CASE WHEN sb.events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbp,
            SUM(CASE WHEN sb.events = 'single' THEN 1 ELSE 0 END) AS singles,
            SUM(CASE WHEN sb.events = 'double' THEN 1 ELSE 0 END) AS doubles,
            SUM(CASE WHEN sb.events = 'triple' THEN 1 ELSE 0 END) AS triples,
            SUM(CASE WHEN sb.events = 'home_run' THEN 1 ELSE 0 END) AS hr,
            SUM(CASE WHEN sb.events IN ('sac_fly','sac_fly_double_play') THEN 1 ELSE 0 END) AS sf,
            COUNT(*) AS pa
        FROM statcast_batted_balls sb
        JOIN boxscore_batting bb_link
             ON bb_link.game_pk = sb.game_pk AND bb_link.player_id = sb.batter_id
        WHERE bb_link.team_id = :team_id
          AND sb.game_date >= :start_date AND sb.game_date < :end_date
          {hand_filter}
    """


def team_woba_components(conn, team_id: int, start_date: str, end_date: str,
                          vs_hand: str | None = None) -> dict:
    params = {"team_id": team_id, "start_date": start_date, "end_date": end_date}
    if vs_hand:
        params["vs_hand"] = vs_hand
    row = conn.execute(_woba_components_query(vs_hand), params).fetchone()
    bb, ibb, hbp, singles, doubles, triples, hr, sf, pa = row
    pa = pa or 0
    non_ab = (bb or 0) + (hbp or 0) + (sf or 0)  # ibb ya está incluido en bb (walk cubre intent_walk aparte, ver nota abajo)
    ab = pa - non_ab
    return {
        "bb": bb or 0, "ibb": ibb or 0, "hbp": hbp or 0,
        "singles": singles or 0, "doubles": doubles or 0, "triples": triples or 0,
        "hr": hr or 0, "sf": sf or 0, "ab": max(ab, 0), "pa": pa,
    }


def rolling_team_woba(conn, team_id: int, as_of_date: str, window_days: int,
                       season: int, vs_hand: str | None = None,
                       league_woba: float = 0.320) -> dict:
    """wOBA de un equipo en los `window_days` anteriores a as_of_date
    (sin incluir as_of_date), con shrinkage hacia league_woba si la
    muestra es chica (típico en splits vs una sola mano en pocos días)."""
    end_d = date.fromisoformat(as_of_date)
    start_d = end_d - timedelta(days=window_days)

    comp = team_woba_components(conn, team_id, str(start_d), str(end_d), vs_hand)
    raw_woba = metrics.woba(
        bb=comp["bb"], ibb=comp["ibb"], hbp=comp["hbp"],
        singles=comp["singles"], doubles=comp["doubles"], triples=comp["triples"],
        hr=comp["hr"], ab=comp["ab"], sf=comp["sf"], season=season,
    )
    shrunk = metrics.shrink_rate(raw_woba, comp["pa"], league_woba, WOBA_STABILIZATION_K)
    return {"woba_raw": raw_woba, "woba_shrunk": shrunk, "pa": comp["pa"]}


@lru_cache(maxsize=20000)
def league_woba_for_window(conn, start_date: str, end_date: str, season: int) -> float:
    """wOBA de liga en el mismo rango de fechas, para usar como ancla del
    shrinkage (más correcto que una constante fija de toda la temporada).

    Memoizada: para una misma fecha, TODOS los partidos del día (~15,
    x2 equipos) piden exactamente el mismo rango -- sin cache esto
    recalculaba un SUM/COUNT sobre statcast_batted_balls (2M+ filas)
    ~30 veces por día en vano. Es seguro cachear porque, dentro de una
    misma corrida, el historial ya cargado no cambia."""
    row = conn.execute(
        """SELECT
             SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bb,
             SUM(CASE WHEN events = 'intent_walk' THEN 1 ELSE 0 END) AS ibb,
             SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbp,
             SUM(CASE WHEN events = 'single' THEN 1 ELSE 0 END) AS singles,
             SUM(CASE WHEN events = 'double' THEN 1 ELSE 0 END) AS doubles,
             SUM(CASE WHEN events = 'triple' THEN 1 ELSE 0 END) AS triples,
             SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hr,
             SUM(CASE WHEN events IN ('sac_fly','sac_fly_double_play') THEN 1 ELSE 0 END) AS sf,
             COUNT(*) AS pa
           FROM statcast_batted_balls
           WHERE game_date >= ? AND game_date < ?""",
        (start_date, end_date),
    ).fetchone()
    bb, ibb, hbp, singles, doubles, triples, hr, sf, pa = row
    pa = pa or 0
    ab = max(pa - (bb or 0) - (hbp or 0) - (sf or 0), 0)
    result = metrics.woba(bb=bb, ibb=ibb, hbp=hbp, singles=singles, doubles=doubles,
                           triples=triples, hr=hr, ab=ab, sf=sf, season=season)
    return result if result is not None else 0.320  # fallback razonable si no hay datos


def build_offense_features(conn, team_id: int, as_of_date: str, season: int,
                            opponent_starter_throws: str | None = None) -> dict:
    """Arma el bloque de features ofensivos de un equipo para una fecha dada.
    opponent_starter_throws ('L'/'R'): si se da, agrega el wOBA de ese
    equipo específicamente contra esa mano (el split que importa para el
    partido de hoy)."""
    features = {}
    lg_30 = league_woba_for_window(
        conn, str(date.fromisoformat(as_of_date) - timedelta(days=30)), as_of_date, season)

    for window in (7, 15, 30):
        r = rolling_team_woba(conn, team_id, as_of_date, window, season, league_woba=lg_30)
        features[f"woba_{window}d"] = r["woba_shrunk"]
        features[f"woba_{window}d_pa"] = r["pa"]

    if opponent_starter_throws in ("L", "R"):
        r = rolling_team_woba(conn, team_id, as_of_date, 30, season,
                               vs_hand=opponent_starter_throws, league_woba=lg_30)
        features["woba_30d_vs_abridor_hand"] = r["woba_shrunk"]
        features["woba_30d_vs_abridor_hand_pa"] = r["pa"]
    else:
        # Mismas claves siempre presentes (aunque en None) -- importante
        # para que el dataset de entrenamiento tenga columnas consistentes
        # fila a fila, sin importar si se conocía la mano del rival.
        features["woba_30d_vs_abridor_hand"] = None
        features["woba_30d_vs_abridor_hand_pa"] = 0

    return features
