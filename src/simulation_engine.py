"""
simulation_engine.py
Fase 6: Motor de Simulación Monte Carlo (10,000 corridas por partido, inning por inning).
Simula el desarrollo estocástico del juego usando un autómata de estados de base y outs
basado en las métricas esperadas del matchup (K%, BB%, Single%, Double%, Triple%, HR%).
"""

from __future__ import annotations

import random
import numpy as np


def simulate_plate_appearance(
    p_k: float, p_bb: float, p_single: float, p_double: float, p_triple: float, p_hr: float
) -> str:
    """Simula una sola aparición en el plato (PA)."""
    r = random.random()
    if r < p_k:
        return "K"
    r -= p_k
    if r < p_bb:
        return "BB"
    r -= p_bb
    if r < p_single:
        return "1B"
    r -= p_single
    if r < p_double:
        return "2B"
    r -= p_double
    if r < p_triple:
        return "3B"
    r -= p_triple
    if r < p_hr:
        return "HR"
    return "OUT"


def simulate_half_inning(
    p_k: float, p_bb: float, p_single: float, p_double: float, p_triple: float, p_hr: float
) -> int:
    """Simula media entrada (half-inning) y devuelve el número de carreras anotadas."""
    outs = 0
    b1, b2, b3 = False, False, False
    runs = 0

    while outs < 3:
        event = simulate_plate_appearance(p_k, p_bb, p_single, p_double, p_triple, p_hr)
        if event in ("K", "OUT"):
            outs += 1
        elif event == "BB":
            if b1 and b2 and b3:
                runs += 1
            elif b1 and b2:
                b3 = True
            elif b1:
                b2 = True
            else:
                b1 = True
        elif event == "1B":
            if b3:
                runs += 1
                b3 = False
            if b2:
                b3 = True
                b2 = False
            if b1:
                b2 = True
            b1 = True
        elif event == "2B":
            if b3:
                runs += 1
                b3 = False
            if b2:
                runs += 1
            if b1:
                b3 = True
                b1 = False
            b2 = True
        elif event == "3B":
            runs += (1 if b1 else 0) + (1 if b2 else 0) + (1 if b3 else 0)
            b1, b2, b3 = False, False, True
        elif event == "HR":
            runs += 1 + (1 if b1 else 0) + (1 if b2 else 0) + (1 if b3 else 0)
            b1, b2, b3 = False, False, False

    return runs


def simulate_single_game(
    home_rates: tuple[float, float, float, float, float, float],
    away_rates: tuple[float, float, float, float, float, float]
) -> tuple[int, int]:
    """Simula un juego completo de 9 entradas (con extra innings si hay empate)."""
    home_runs = 0
    away_runs = 0

    for _inning in range(9):
        away_runs += simulate_half_inning(*away_rates)
        home_runs += simulate_half_inning(*home_rates)

    # Extra innings en caso de empate
    extra_inning = 10
    while home_runs == away_runs and extra_inning <= 13:
        away_runs += simulate_half_inning(*away_rates)
        home_runs += simulate_half_inning(*home_rates)
        extra_inning += 1

    # Si persiste empate al inning 13, desempate aleatorio por ventaja de campo
    if home_runs == away_runs:
        if random.random() < 0.54:
            home_runs += 1
        else:
            away_runs += 1

    return home_runs, away_runs


def run_monte_carlo_simulation(
    home_woba: float, away_woba: float,
    home_fip: float | None = None, away_fip: float | None = None,
    n_simulations: int = 10000, seed: int = 42
) -> dict:
    """Ejecuta N simulaciones Monte Carlo por juego y devuelve las métricas estadísticas."""
    random.seed(seed)
    np.random.seed(seed)

    # Tasas base por defecto derivadas de wOBA y FIP
    # (K%, BB%, Single%, Double%, Triple%, HR%)
    base_k = 0.22
    base_bb = 0.08
    base_1b = 0.15
    base_2b = 0.045
    base_3b = 0.005
    base_hr = 0.035

    # Ajuste por wOBA ofensivo y FIP del abridor
    h_woba_adj = (home_woba - 0.315) * 0.4
    a_woba_adj = (away_woba - 0.315) * 0.4

    home_rates = (
        max(0.05, base_k - h_woba_adj),
        max(0.02, base_bb + h_woba_adj),
        max(0.05, base_1b + h_woba_adj),
        max(0.01, base_2b + h_woba_adj * 0.3),
        max(0.001, base_3b),
        max(0.005, base_hr + h_woba_adj * 0.4),
    )

    away_rates = (
        max(0.05, base_k - a_woba_adj),
        max(0.02, base_bb + a_woba_adj),
        max(0.05, base_1b + a_woba_adj),
        max(0.01, base_2b + a_woba_adj * 0.3),
        max(0.001, base_3b),
        max(0.005, base_hr + a_woba_adj * 0.4),
    )

    home_scores = []
    away_scores = []
    home_wins = 0

    for _ in range(n_simulations):
        h_score, a_score = simulate_single_game(home_rates, away_rates)
        home_scores.append(h_score)
        away_scores.append(a_score)
        if h_score > a_score:
            home_wins += 1

    home_scores_arr = np.array(home_scores)
    away_scores_arr = np.array(away_scores)
    totals_arr = home_scores_arr + away_scores_arr
    diffs_arr = home_scores_arr - away_scores_arr

    return {
        "n_simulations": n_simulations,
        "home_win_proba_sim": round(home_wins / n_simulations, 4),
        "home_runs_mean": round(float(np.mean(home_scores_arr)), 2),
        "away_runs_mean": round(float(np.mean(away_scores_arr)), 2),
        "total_runs_mean": round(float(np.mean(totals_arr)), 2),
        "over_8_5_proba": round(float(np.mean(totals_arr > 8.5)), 4),
        "under_8_5_proba": round(float(np.mean(totals_arr < 8.5)), 4),
        "run_line_minus_1_5_proba": round(float(np.mean(diffs_arr >= 2)), 4),
    }
