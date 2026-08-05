"""
model_data.py
Carga data/training_dataset.parquet, define las variables objetivo,
separa en train/val/test POR TEMPORADA (nunca al azar -- mezclar fechas
al azar filtraría información del futuro hacia el pasado, algo crítico
en datos deportivos secuenciales) y prepara las features quitando las
columnas que son fuga de información obvia.

Split temporal:
    train = 2015-2023   (9 temporadas para aprender)
    val   = 2024        (para afinar hiperparámetros)
    test  = 2025-2026   (se toca UNA sola vez, al final, nunca antes)
"""

from __future__ import annotations

import pandas as pd

TARGET_WIN = "home_win"
TARGET_RUNS = "total_runs"

TRAIN_SEASONS = set(range(2015, 2024))   # 2015-2023
VAL_SEASONS = {2024}
TEST_SEASONS = {2025, 2026}

# Columnas que son fuga de información directa (resultado o identificadores
# de partido) o que decidimos no usar todavía en esta primera versión.
DROP_COLS = {
    "game_pk", "game_date", "status", "home_score", "away_score",
    "season",  # se usa para el split, no como feature (evita atarse al año exacto)
    "venue_id",  # ya viene resumido en park_factor_runs
    "home_team_id", "away_team_id",  # identidad de equipo != talento actual;
                                       # el modelo debe aprender de las stats,
                                       # no memorizar IDs (se puede revisar
                                       # más adelante con encoding apropiado)
    "home_abridor_id", "away_abridor_id",  # igual que team_id: identidad, no talento
    "home_abridor_throws", "away_abridor_throws",  # se reemplazan por la versión binaria
    TARGET_WIN, TARGET_RUNS,
}


def load_dataset(path: str = "data/training_dataset.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df[TARGET_WIN] = (df["home_score"] > df["away_score"]).astype(int)
    df[TARGET_RUNS] = df["home_score"] + df["away_score"]
    # Mano del abridor a binario (1 = zurdo). Los ~0.07% con dato faltante
    # quedan como 0 (equivalente a "no zurdo") -- una simplificación
    # aceptable dado lo chico del porcentaje; se puede refinar con un
    # flag de "mano desconocida" más adelante si se justifica.
    df["home_abridor_zurdo"] = (df["home_abridor_throws"] == "L").astype(int)
    df["away_abridor_zurdo"] = (df["away_abridor_throws"] == "L").astype(int)
    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val = df[df["season"].isin(VAL_SEASONS)].copy()
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    return train, val, test


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def prepare_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    feats = feature_columns(df)
    return df[feats], df[target]


if __name__ == "__main__":
    df = load_dataset()
    train, val, test = temporal_split(df)
    print(f"Total: {len(df)} | Train (2015-2023): {len(train)} | "
          f"Val (2024): {len(val)} | Test (2025-2026): {len(test)}")
    print(f"\nFeatures ({len(feature_columns(df))}):")
    for c in feature_columns(df):
        print(f"  {c}")
