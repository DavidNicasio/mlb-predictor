import sqlite3
import sys
from pathlib import Path

# Añadir src/ al sys.path para importar los módulos del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config
import db
import extract_teams


def test_config_constants():
    assert config.PROB_FAVORITE_THRESHOLD == 0.60
    assert config.FIP_EXCELLENT_THRESHOLD == 3.20
    assert config.MIN_PROP_SAMPLE_SIZE == 30


def test_init_db_schema():
    conn = sqlite3.connect(":memory:")
    db.init_db(conn)
    cursor = conn.cursor()

    # Verificar existencia de tablas clave
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    assert "games" in tables
    assert "teams" in tables
    assert "predictions_log" in tables

    # Verificar columna league en games y teams
    cols_games = [d[1] for d in cursor.execute("PRAGMA table_info(games)").fetchall()]
    assert "league" in cols_games

    conn.close()


def test_parse_teams():
    sample_json = {
        "teams": [
            {"id": 532, "name": "Diablos Rojos del Mexico", "abbreviation": "MEX"}
        ]
    }
    rows = extract_teams.parse_teams(sample_json, league="LMB")
    assert len(rows) == 1
    assert rows[0]["team_id"] == 532
    assert rows[0]["league"] == "LMB"
