import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from test_pipeline_and_models import test_config_constants, test_init_db_schema, test_parse_teams, test_weather_parser
from test_metrics import test_fip_formula, test_woba_formula, test_shrink_rate
from test_report_card import test_compute_grades_final_and_pending
from test_features_f5 import test_calculate_f5_projections_bounds_and_nones
from test_lmb_pipeline import test_parse_lmb_games, test_lmb_models_training
from test_features_lineup import test_lineup_woba_fallback_and_shrinkage
from test_simulation_engine import test_monte_carlo_convergence, test_fatigue_and_statcast

class TestMLB(unittest.TestCase):
    def test_config(self):
        test_config_constants()

    def test_db(self):
        test_init_db_schema()

    def test_teams(self):
        test_parse_teams()

    def test_weather(self):
        test_weather_parser()

    def test_sabermetric_metrics(self):
        test_fip_formula()
        test_woba_formula()
        test_shrink_rate()

    def test_report_card_grades(self):
        test_compute_grades_final_and_pending()

    def test_f5_projections(self):
        test_calculate_f5_projections_bounds_and_nones()

    def test_lmb_module(self):
        test_parse_lmb_games()
        test_lmb_models_training(Path("scratch"))

    def test_lineup_module(self):
        test_lineup_woba_fallback_and_shrinkage(Path("scratch"))

    def test_simulation_and_monte_carlo(self):
        test_monte_carlo_convergence()
        test_fatigue_and_statcast()

if __name__ == "__main__":
    unittest.main()
