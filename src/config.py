"""
config.py
Centralización de rutas por defecto y umbrales de negocio (números mágicos)
del sistema de predicciones de MLB / LMB.
"""

from __future__ import annotations

# Rutas de base de datos y modelos por defecto
DEFAULT_DB_PATH: str = "data/mlb.db"
DEFAULT_WIN_MODEL_PATH: str = "data/model_win.joblib"
DEFAULT_RUNS_MODEL_PATH: str = "data/model_runs.joblib"
DEFAULT_DATASET_PATH: str = "data/training_dataset.parquet"

# Umbrales para clasificar probabilidades de victoria (ML)
PROB_FAVORITE_THRESHOLD: float = 0.60
PROB_UNDERDOG_THRESHOLD: float = 0.40

# Umbrales de FIP de abridores para recomendaciones de apuestas F5 / NRFI
FIP_EXCELLENT_THRESHOLD: float = 3.20
FIP_GOOD_THRESHOLD: float = 3.65

# Tamaño mínimo de muestra para dar confianza a una regla en el backtest
MIN_PROP_SAMPLE_SIZE: int = 30
