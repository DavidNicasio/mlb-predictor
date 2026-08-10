import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import db
import features_lineup


def test_lineup_woba_fallback_and_shrinkage(tmp_path):
    db_file = tmp_path / "test_lineup.db"
    conn = db.get_connection(str(db_file))
    db.init_db(conn)

    # 1. Alineacion vacia debe retornar wOBA de liga (0.315)
    res_empty = features_lineup.get_lineup_projected_woba(conn, [], "2024-06-15")
    assert res_empty == 0.315

    # 2. Alineacion de 9 jugadores sin historial (shrinkage completo a 0.315)
    pids = [101, 102, 103, 104, 105, 106, 107, 108, 109]
    res_new = features_lineup.get_lineup_projected_woba(conn, pids, "2024-06-15")
    assert res_new == 0.315

    conn.close()
