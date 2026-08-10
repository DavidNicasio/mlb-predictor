import sys
from pathlib import Path
import sqlite3

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import extract_schedule_lmb
import train_lmb_models
import db


def test_parse_lmb_games():
    sample_sched_json = {
        "dates": [
            {
                "date": "2024-05-01",
                "games": [
                    {
                        "gamePk": 770237,
                        "gameType": "R",
                        "season": "2024",
                        "status": {"detailedState": "Final"},
                        "teams": {
                            "away": {"score": 3, "team": {"id": 500, "name": "El Aguila"}},
                            "home": {"score": 8, "team": {"id": 501, "name": "Diablos Rojos"}},
                        },
                        "venue": {"id": 10, "name": "Estadio Alfredo Harp Helu"},
                    }
                ],
            }
        ]
    }

    games, probables = extract_schedule_lmb.parse_lmb_games(sample_sched_json)
    assert len(games) == 1
    assert games[0]["game_pk"] == 770237
    assert games[0]["league"] == "LMB"
    assert games[0]["home_score"] == 8


def test_lmb_models_training(tmp_path):
    db_file = tmp_path / "test_lmb.db"
    conn = db.get_connection(str(db_file))
    db.init_db(conn)

    # Insertar partido de prueba
    conn.execute("""
        INSERT OR REPLACE INTO games (game_pk, game_date, season, status, home_team_id, away_team_id, home_score, away_score, league)
        VALUES (770237, '2024-05-01', 2024, 'Final', 501, 500, 8, 3, 'LMB')
    """)
    conn.commit()

    df_feats = train_lmb_models.build_lmb_features(conn)
    conn.close()

    assert not df_feats.empty
    assert len(df_feats) == 1
    assert df_feats.iloc[0]["home_won"] == 1
    assert df_feats.iloc[0]["total_runs"] == 11
