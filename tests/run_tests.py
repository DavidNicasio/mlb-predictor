import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from test_pipeline_and_models import test_config_constants, test_init_db_schema, test_parse_teams

class TestMLB(unittest.TestCase):
    def test_config(self):
        test_config_constants()

    def test_db(self):
        test_init_db_schema()

    def test_teams(self):
        test_parse_teams()

if __name__ == "__main__":
    unittest.main()
