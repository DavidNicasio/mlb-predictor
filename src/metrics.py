"""
metrics.py
Fórmulas sabermétricas calculadas desde datos propios (MLB Stats API +
Statcast), sin depender de scraping de FanGraphs.

- FIP y su constante: se derivan 100% de nuestros propios datos de liga.
- xFIP: usa el HR/FB de liga que calculamos nosotros desde Statcast.
- wOBA: usa una tabla de pesos por temporada. Estos pesos SÍ vienen de
  investigación publicada (no son secretos), pero cambian cada año.
  Actualízalos una vez al año leyendo la página pública de "Guts" de
  FanGraphs (lectura manual puntual, no scraping automatizado). Si el
  año no está en la tabla, se usa el más reciente disponible como
  aproximación razonable.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# wOBA: pesos de referencia por temporada (aproximados, revisar cada enero)
# Fuente para actualizar: fangraphs.com/guts.aspx?type=cn  (lectura manual)
# ---------------------------------------------------------------------------
WOBA_WEIGHTS: dict[int, dict[str, float]] = {
    2024: {"bb": 0.689, "hbp": 0.720, "s1": 0.883, "s2": 1.257, "s3": 1.593, "hr": 2.058},
    2025: {"bb": 0.692, "hbp": 0.723, "s1": 0.888, "s2": 1.271, "s3": 1.616, "hr": 2.101},
}


_WARNED_SEASONS: set[int] = set()


def get_woba_weights(season: int) -> dict[str, float]:
    if season in WOBA_WEIGHTS:
        return WOBA_WEIGHTS[season]
    latest = max(WOBA_WEIGHTS)
    if season not in _WARNED_SEASONS:
        print(f"[metrics] sin pesos de wOBA para {season}, uso los de {latest} como aproximación")
        _WARNED_SEASONS.add(season)
    return WOBA_WEIGHTS[latest]


def woba(bb: int, ibb: int, hbp: int, singles: int, doubles: int, triples: int,
          hr: int, ab: int, sf: int, season: int) -> float | None:
    w = get_woba_weights(season)
    ubb = (bb or 0) - (ibb or 0)  # wOBA usa BB no intencional
    denom = (ab or 0) + ubb + (sf or 0) + (hbp or 0)
    if denom == 0:
        return None
    num = (w["bb"] * ubb + w["hbp"] * (hbp or 0) + w["s1"] * (singles or 0)
           + w["s2"] * (doubles or 0) + w["s3"] * (triples or 0) + w["hr"] * (hr or 0))
    return round(num / denom, 3)


# ---------------------------------------------------------------------------
# FIP / xFIP: derivados enteramente de nuestros propios datos
# ---------------------------------------------------------------------------

def innings_from_outs(outs: int | None) -> float | None:
    if outs is None:
        return None
    return round(outs / 3, 2)


def fip(hr: int, bb: int, hbp: int, k: int, outs: int, fip_constant: float) -> float | None:
    ip = innings_from_outs(outs)
    if not ip:
        return None
    core = (13 * (hr or 0) + 3 * ((bb or 0) + (hbp or 0)) - 2 * (k or 0)) / ip
    return round(core + fip_constant, 2)


def compute_fip_constant(league_era: float, hr: int, bb: int, hbp: int,
                           k: int, outs: int) -> float:
    """FIP_constant = lgERA - ((13*HR + 3*(BB+HBP) - 2*K) / IP), calculado
    con los totales de liga que ya tenemos en nuestra propia base."""
    ip = innings_from_outs(outs)
    core = (13 * hr + 3 * (bb + hbp) - 2 * k) / ip
    return round(league_era - core, 3)


def xfip(fb: int, league_hr_per_fb: float, bb: int, hbp: int, k: int,
          outs: int, fip_constant: float) -> float | None:
    """Como FIP, pero reemplaza HR reales por HR esperados = FB * lgHR/FB.
    Esto reduce el ruido de temporadas cortas en el HR/FB del lanzador."""
    ip = innings_from_outs(outs)
    if not ip:
        return None
    expected_hr = (fb or 0) * league_hr_per_fb
    core = (13 * expected_hr + 3 * ((bb or 0) + (hbp or 0)) - 2 * (k or 0)) / ip
    return round(core + fip_constant, 2)


# ---------------------------------------------------------------------------
# Regresión a la media (shrinkage bayesiano simple), usada por los módulos
# de features (Fase 3) para no tomar muestras chicas al pie de la letra.
# ---------------------------------------------------------------------------

def shrink_rate(observed_rate: float | None, sample_size: int | None,
                 league_rate: float, k: float) -> float:
    """weight = n/(n+k): entre más muestra, más se confía en lo observado.
    k es el 'punto de estabilización' aproximado de la métrica (ver Fase 1:
    K% ~60 TBF, BB% ~150 TBF, HR/FB ~400-500 TBF, wOBA agregado ~150-250 PA)."""
    if observed_rate is None or not sample_size:
        return league_rate
    weight = sample_size / (sample_size + k)
    return round(weight * observed_rate + (1 - weight) * league_rate, 4)


# ---------------------------------------------------------------------------
# Constantes de liga: se recalculan cada noche desde nuestra propia base
# (boxscore_pitching + statcast_batted_balls), sin tocar FanGraphs.
# ---------------------------------------------------------------------------

def compute_league_constants(conn, season: int) -> dict | None:
    row = conn.execute(
        """SELECT SUM(bp.er) AS er, SUM(bp.outs) AS outs, SUM(bp.hr) AS hr,
                  SUM(bp.bb) AS bb, SUM(bp.hbp) AS hbp, SUM(bp.so) AS so
           FROM boxscore_pitching bp
           JOIN games g ON g.game_pk = bp.game_pk
           WHERE g.season = ? AND g.status = 'Final'""",
        (season,),
    ).fetchone()

    if not row or not row[1]:
        print(f"[metrics] sin datos suficientes de pitcheo para {season} todavía")
        return None

    er, outs, hr, bb, hbp, so = row
    ip = innings_from_outs(outs)
    league_era = round((er / ip) * 9, 3) if ip else None
    fip_constant = compute_fip_constant(league_era, hr, bb, hbp, so, outs) if league_era else None

    fb_row = conn.execute(
        """SELECT
             SUM(CASE WHEN launch_angle >= 25 AND launch_angle < 50 THEN 1 ELSE 0 END) AS fb,
             SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hr_bb
           FROM statcast_batted_balls
           WHERE game_date LIKE ? """,
        (f"{season}-%",),
    ).fetchone()

    lg_hr_fb = None
    if fb_row and fb_row[0]:
        fb_count, hr_count = fb_row
        lg_hr_fb = round(hr_count / fb_count, 4) if fb_count else None

    result = {
        "season": season,
        "league_era": league_era,
        "fip_constant": fip_constant,
        "lg_hr_fb": lg_hr_fb,
    }

    if league_era is not None:
        conn.execute(
            """INSERT OR REPLACE INTO league_constants (season, league_era, fip_constant, lg_hr_fb)
               VALUES (:season, :league_era, :fip_constant, :lg_hr_fb)""",
            result,
        )
        conn.commit()

    return result


if __name__ == "__main__":
    # Prueba rápida de las fórmulas con números de mano, sin necesitar BD.
    fc = compute_fip_constant(league_era=4.00, hr=5200, bb=15200, hbp=1800, k=39000, outs=131220)
    print("FIP constant de ejemplo:", fc)
    print("FIP de un abridor de ejemplo:", fip(hr=18, bb=40, hbp=5, k=180, outs=540, fip_constant=fc))
    print("wOBA de ejemplo:", woba(bb=50, ibb=3, hbp=5, singles=90, doubles=25,
                                    triples=2, hr=20, ab=520, sf=4, season=2025))
