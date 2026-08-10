import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import features_f5


def test_calculate_f5_projections_bounds_and_nones():
    # Caso 1: Valores normales
    res = features_f5.calculate_f5_projections(
        home_fip=3.10,
        away_fip=3.80,
        home_woba=0.340,
        away_woba=0.310,
        full_game_runs_pred=8.5,
        home_win_proba=0.58,
    )
    assert "f5_total_runs_pred" in res
    assert "f5_home_win_proba" in res
    assert 2.5 <= res["f5_total_runs_pred"] <= 8.0

    # Caso 2: FIP/wOBA en None (debe manejar fallback a 4.20 FIP sin fallar)
    res_none = features_f5.calculate_f5_projections(
        home_fip=None,
        away_fip=None,
        home_woba=None,
        away_woba=None,
        full_game_runs_pred=9.0,
        home_win_proba=0.52,
    )
    assert res_none["f5_total_runs_pred"] == 5.0  # 9.0 * 0.55 * (4.2/4.2) = 4.95 -> 5.0
    assert 2.5 <= res_none["f5_total_runs_pred"] <= 8.0

    # Caso 3: Proyección extrema baja de juego completo (ej. 3.0 carreras) -> debe acotarse a 2.5
    res_low = features_f5.calculate_f5_projections(
        home_fip=2.0,
        away_fip=2.0,
        home_woba=0.250,
        away_woba=0.250,
        full_game_runs_pred=3.0,
        home_win_proba=0.50,
    )
    assert res_low["f5_total_runs_pred"] == 2.5
