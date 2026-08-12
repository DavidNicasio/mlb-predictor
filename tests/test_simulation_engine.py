import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import simulation_engine


def test_monte_carlo_convergence():
    res = simulation_engine.run_monte_carlo_simulation(
        home_woba=0.340, away_woba=0.310, n_simulations=1000, seed=42
    )
    assert res["n_simulations"] == 1000
    assert 0.40 <= res["home_win_proba_sim"] <= 0.70
    assert res["total_runs_mean"] > 0
    assert 0.0 <= res["over_8_5_proba"] <= 1.0
    assert 0.0 <= res["run_line_minus_1_5_proba"] <= 1.0


def test_fatigue_and_statcast():
    import features_bullpen_rest
    import features_statcast_matchup
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE boxscore_pitching (game_pk INT, team_id INT, player_id INT, is_starter INT, pitches_thrown INT, outs INT)")
    conn.execute("CREATE TABLE games (game_pk INT, game_date TEXT, status TEXT)")
    conn.execute("CREATE TABLE statcast_batted_balls (pitcher_id INT, game_date TEXT, pitch_type TEXT)")
    conn.commit()

    fatigue = features_bullpen_rest.compute_bullpen_fatigue(conn, 108, "2024-06-15")
    assert fatigue["fatigue_index"] == 0.0

    mix = features_statcast_matchup.get_pitcher_arsenal_mix(conn, 12345, "2024-06-15")
    assert mix["FF"] == 0.45

    conn.close()
