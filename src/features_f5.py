"""
features_f5.py
Módulo de proyecciones para las Primeras 5 Entradas (First 5 Innings - F5).
Aísla el duelo de los abridores titulares frente a los bateadores contrarios,
eliminando el impacto de los bullpens en las entradas finales.
"""

from __future__ import annotations

import config


def calculate_f5_projections(
    home_fip: float | None,
    away_fip: float | None,
    home_woba: float | None,
    away_woba: float | None,
    full_game_runs_pred: float,
    home_win_proba: float,
) -> dict:
    """Calcula la probabilidad de victoria F5 y la línea de carreras totales F5."""
    # Factor de proporción base de F5 (aprox 55% de las carreras del partido completo)
    base_f5_ratio = config.F5_BASE_RATIO

    # Ajuste por FIP de abridores (si los FIPs son muy bajos, F5 cae; si son altos, F5 sube)
    h_fip = home_fip if home_fip and home_fip > 0 else 4.20
    a_fip = away_fip if away_fip and away_fip > 0 else 4.20

    avg_fip = (h_fip + a_fip) / 2.0
    fip_factor = avg_fip / 4.20

    f5_runs_pred = max(config.F5_MIN_RUNS_PRED, min(config.F5_MAX_RUNS_PRED, full_game_runs_pred * base_f5_ratio * fip_factor))

    # Ajuste de ventaja en F5 para el abridor de mejor FIP
    fip_diff = a_fip - h_fip  # Positivo si el abridor local es mejor (menor FIP)
    f5_win_proba = max(0.20, min(0.80, home_win_proba + (fip_diff * 0.03)))

    return {
        "f5_total_runs_pred": round(f5_runs_pred, 1),
        "f5_home_win_proba": round(f5_win_proba, 3),
    }
