"""
db.py
Esquema y helpers de conexión para la base de datos local (SQLite).

Diseño en capas (mini medallion architecture):
  - bronze: games, probable_pitchers, boxscore_batting, boxscore_pitching,
            statcast_batted_balls  -> datos crudos, casi 1:1 con la fuente
  - reference: league_constants, park_factors -> tablas pequeñas de apoyo
  - (silver/gold, con rolling windows y features, se construyen en Fase 3)

Uso:
    from db import get_connection, init_db
    conn = get_connection("data/mlb.db")
    init_db(conn)
"""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_pk         INTEGER PRIMARY KEY,
    game_date       TEXT NOT NULL,      -- YYYY-MM-DD
    game_date_utc   TEXT,               -- Timestamp UTC de inicio del partido
    season          INTEGER NOT NULL,
    game_type       TEXT,               -- R, F, D, L, W, S...
    status          TEXT,               -- Scheduled, Final, In Progress...
    home_team_id    INTEGER NOT NULL,
    away_team_id    INTEGER NOT NULL,
    home_score      INTEGER,
    away_score      INTEGER,
    venue_id        INTEGER,
    venue_name      TEXT,
    weather_condition TEXT,
    weather_temp    INTEGER,
    weather_wind    TEXT,
    league          TEXT DEFAULT 'MLB',
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS probable_pitchers (
    game_pk         INTEGER NOT NULL,
    team_id         INTEGER NOT NULL,
    is_home         INTEGER NOT NULL,   -- 1 = local, 0 = visitante
    pitcher_id      INTEGER,
    pitcher_name    TEXT,
    PRIMARY KEY (game_pk, team_id),
    FOREIGN KEY (game_pk) REFERENCES games(game_pk)
);

-- Una fila por bateador por partido (boxscore, no play-by-play)
CREATE TABLE IF NOT EXISTS boxscore_batting (
    game_pk         INTEGER NOT NULL,
    team_id         INTEGER NOT NULL,
    player_id       INTEGER NOT NULL,
    player_name     TEXT,
    bats            TEXT,               -- L, R, S
    ab              INTEGER, h INTEGER, doubles INTEGER, triples INTEGER,
    hr              INTEGER, bb INTEGER, ibb INTEGER, hbp INTEGER,
    sf INTEGER, so INTEGER, sb INTEGER, cs INTEGER,
    PRIMARY KEY (game_pk, player_id)
);

-- Una fila por lanzador por partido (boxscore, no pitch-by-pitch)
CREATE TABLE IF NOT EXISTS boxscore_pitching (
    game_pk         INTEGER NOT NULL,
    team_id         INTEGER NOT NULL,
    player_id       INTEGER NOT NULL,
    player_name     TEXT,
    throws          TEXT,               -- L, R
    is_starter      INTEGER NOT NULL DEFAULT 0,
    outs            INTEGER,            -- IP en outs (para evitar el .1/.2 de IP)
    h INTEGER, r INTEGER, er INTEGER, bb INTEGER, ibb INTEGER,
    hbp INTEGER, so INTEGER, hr INTEGER, pitches_thrown INTEGER,
    PRIMARY KEY (game_pk, player_id)
);

-- Eventos de bateo en juego (batted balls) desde Baseball Savant/Statcast.
-- Guardamos solo bolas puestas en juego, no cada pitch, para mantener
-- la base liviana (uso personal) sin perder Barrel%/Hard-Hit%/EV.
CREATE TABLE IF NOT EXISTS statcast_batted_balls (
    game_pk         INTEGER,
    game_date       TEXT NOT NULL,
    batter_id       INTEGER NOT NULL,
    pitcher_id      INTEGER NOT NULL,
    stand           TEXT,               -- lado del bateador L/R
    p_throws        TEXT,               -- mano del lanzador L/R
    launch_speed    REAL,
    launch_angle    REAL,
    hit_distance    REAL,
    events          TEXT,               -- single, double, home_run, field_out...
    barrel          INTEGER,            -- 1/0, ya viene calculado en Statcast
    PRIMARY KEY (game_date, batter_id, pitcher_id, launch_speed, launch_angle)
);

-- Constantes de liga por temporada, derivadas de nuestros propios datos
-- (no dependemos de FanGraphs para esto: se recalculan cada noche con
-- las temporadas cargadas hasta la fecha).
CREATE TABLE IF NOT EXISTS league_constants (
    season          INTEGER PRIMARY KEY,
    league_era      REAL,
    fip_constant    REAL,
    lg_hr_fb        REAL,       -- HR/FB rate de liga, para xFIP
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS park_factors (
    venue_id        INTEGER NOT NULL,
    season          INTEGER NOT NULL,
    factor_runs     REAL,
    factor_hr       REAL,
    factor_hr_vs_l  REAL,
    factor_hr_vs_r  REAL,
    source          TEXT,       -- ej. 'fangraphs_manual_export_2026'
    PRIMARY KEY (venue_id, season)
);

-- Registro de qué rangos de fechas ya se descargaron para una fuente
-- dada. Se usa en el backfill para saber qué ya está cargado sin
-- depender de si hubo o no filas de datos (un día sin partidos
-- legítimamente no genera filas, pero sí debe quedar marcado como
-- "ya revisado" para no volver a pedirlo).
CREATE TABLE IF NOT EXISTS ingestion_log (
    source          TEXT NOT NULL,   -- ej. 'statcast'
    range_start     TEXT NOT NULL,
    range_end       TEXT NOT NULL,
    loaded_at       TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (source, range_start, range_end)
);

-- Datos biograficos de jugador (bats/throws) que NO vienen en el
-- boxscore -- confirmado con la API real que el 'person' embebido ahi
-- es solo {id, fullName, link, boxscoreName}. Hay que pedirlos aparte
-- al endpoint /people, que sí trae batSide/pitchHand.
CREATE TABLE IF NOT EXISTS players (
    player_id       INTEGER PRIMARY KEY,
    full_name       TEXT,
    bats            TEXT,   -- L, R, S
    throws          TEXT,   -- L, R
    updated_at      TEXT DEFAULT (datetime('now'))
);

-- Nombres de equipo (la API de partidos/boxscores solo trae team_id).
CREATE TABLE IF NOT EXISTS teams (
    team_id         INTEGER PRIMARY KEY,
    name            TEXT,
    abbreviation    TEXT,
    league          TEXT DEFAULT 'MLB',
    updated_at      TEXT DEFAULT (datetime('now'))
);

-- Historial de lo que el modelo predijo cada día, para poder comparar
-- despues contra el resultado real (games.home_score/away_score) una
-- vez que el partido ya se jugó, y monitorear si la calibración se
-- mantiene con el tiempo.
CREATE TABLE IF NOT EXISTS predictions_log (
    game_pk             INTEGER NOT NULL,
    predicted_at        TEXT NOT NULL,
    home_win_proba      REAL,
    total_runs_pred      REAL,
    win_model_type      TEXT,
    runs_model_type     TEXT,
    weather_temp        INTEGER,
    weather_wind        TEXT,
    PRIMARY KEY (game_pk, predicted_at)
);

-- Linescore: carreras por entrada (para modelo F5 y backtesting NRFI).
-- Se backfillea desde /api/v1/game/{game_pk}/linescore.
CREATE TABLE IF NOT EXISTS game_linescore (
    game_pk     INTEGER NOT NULL,
    inning      INTEGER NOT NULL,
    home_runs   INTEGER,
    away_runs   INTEGER,
    PRIMARY KEY (game_pk, inning),
    FOREIGN KEY (game_pk) REFERENCES games(game_pk)
);

CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_home_team ON games(home_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_games_away_team ON games(away_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_pitch_team_starter ON boxscore_pitching(team_id, is_starter);
CREATE INDEX IF NOT EXISTS idx_bat_player ON boxscore_batting(player_id);
CREATE INDEX IF NOT EXISTS idx_pitch_player ON boxscore_pitching(player_id);
CREATE INDEX IF NOT EXISTS idx_statcast_date ON statcast_batted_balls(game_date);
CREATE INDEX IF NOT EXISTS idx_statcast_pitcher ON statcast_batted_balls(pitcher_id, game_pk);
CREATE INDEX IF NOT EXISTS idx_linescore_game ON game_linescore(game_pk);
"""


def get_connection(db_path: str = "data/mlb.db") -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.OperationalError:
        pass
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    for col, ctype in [("game_date_utc", "TEXT"), ("weather_condition", "TEXT"), ("weather_temp", "INTEGER"), ("weather_wind", "TEXT"), ("league", "TEXT DEFAULT 'MLB'")]:
        try:
            conn.execute(f"ALTER TABLE games ADD COLUMN {col} {ctype};")
        except sqlite3.OperationalError:
            pass
    try:
        conn.execute("ALTER TABLE teams ADD COLUMN league TEXT DEFAULT 'MLB';")
    except sqlite3.OperationalError:
        pass
    for col, ctype in [("weather_temp", "INTEGER"), ("weather_wind", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE predictions_log ADD COLUMN {col} {ctype};")
        except sqlite3.OperationalError:
            pass
    conn.commit()


if __name__ == "__main__":
    conn = get_connection("data/mlb.db")
    init_db(conn)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()
    print("Tablas creadas:", [t[0] for t in tables])
    conn.close()
