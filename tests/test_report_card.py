import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import report_card


def test_compute_grades_final_and_pending():
    sample_data = [
        {
            "game_pk": 101,
            "status": "Final",
            "home_score": 5,
            "away_score": 3,
            "home_win_proba": 0.65,
            "total_runs_pred": 9.2,
            "home_name": "New York Yankees",
            "away_name": "Boston Red Sox",
            "home_abbr": "NYY",
            "away_abbr": "BOS",
        },
        {
            "game_pk": 102,
            "status": "Scheduled",
            "home_score": None,
            "away_score": None,
            "home_win_proba": 0.45,
            "total_runs_pred": 7.8,
            "home_name": "Los Angeles Dodgers",
            "away_name": "San Francisco Giants",
            "home_abbr": "LAD",
            "away_abbr": "SF",
        },
    ]
    df_raw = pd.DataFrame(sample_data)
    df_graded = report_card.compute_grades(df_raw)

    assert len(df_graded) == 2

    # Partido Finalizado (101): Marcador 5-3 (8 carreras totales)
    # Ganador real: Home (1). Proba local: 0.65 (Favorito Home). win_hit debe ser True.
    # Proyeccion carreras: 9.2 -> actual_total (8) - total_runs_pred (9.2) = -1.2 -> ou_label debe ser "BAJO".
    r_final = df_graded[df_graded["game_pk"] == 101].iloc[0]
    assert bool(r_final["is_final"]) is True
    assert bool(r_final["win_hit"]) is True
    assert r_final["ou_label"] == "BAJO"

    # Partido Pendiente (102): status="Scheduled"
    r_pending = df_graded[df_graded["game_pk"] == 102].iloc[0]
    assert bool(r_pending["is_final"]) is False
    assert r_pending["win_hit"] is None
