"""
weather_features.py
Codificación numérica del clima para ajuste post-predicción.

Como la API de MLB no devuelve clima retroactivo (solo ~0.2% de los
juegos tienen el dato), NO se usa como feature de entrenamiento.
En cambio, se aplica como ajuste heurístico post-modelo en la
inferencia diaria cuando el dato SÍ está disponible.

Codificación del viento:
  - Velocidad (mph): numérico
  - Dirección: "Out" (favorece HR/carreras), "In" (las suprime),
    "Cross" (L to R / R to L, efecto neutral), "None" (calma o domo)
"""

from __future__ import annotations

import re


def parse_wind(wind_str: str | None) -> tuple[int, str]:
    """Parsea '7 mph, Out To RF' → (7, 'Out').

    Categorías de dirección:
      - 'Out': Out To CF / Out To RF / Out To LF  → viento que sale del campo
      - 'In':  In From CF / In From RF / In From LF → viento que entra
      - 'Cross': L To R / R To L → viento cruzado
      - 'None': 0 mph, None / Varies / no data

    Returns (speed_mph, direction_category).
    """
    if not wind_str or not isinstance(wind_str, str):
        return (0, "None")

    # Extraer velocidad
    speed_match = re.search(r"(\d+)\s*mph", wind_str, re.IGNORECASE)
    speed = int(speed_match.group(1)) if speed_match else 0

    # Extraer dirección
    lower = wind_str.lower()
    if "out to" in lower:
        direction = "Out"
    elif "in from" in lower:
        direction = "In"
    elif "l to r" in lower or "r to l" in lower:
        direction = "Cross"
    else:
        direction = "None"

    return (speed, direction)


def is_dome(weather_condition: str | None) -> bool:
    """Retorna True si el partido es en domo / techo cerrado."""
    if not weather_condition:
        return False
    lower = weather_condition.lower()
    return "roof closed" in lower or "dome" in lower


def weather_runs_adjustment(
    temp: int | None,
    wind_speed: int,
    wind_dir: str,
    weather_condition: str | None,
) -> float:
    """Calcula un ajuste aditivo de carreras totales basado en clima.

    Basado en investigación sabermétrica publicada:
    - Temperatura: cada 10°F sobre 72°F → ~+0.15 runs (calor expande el aire,
      la pelota viaja más)
    - Viento Out >10mph: +0.3 runs (ayuda fly balls a salir)
    - Viento In >10mph: -0.25 runs (suprime carreras)
    - Domo/Roof Closed: 0 (clima controlado, sin ajuste)

    Retorna 0.0 si no hay datos suficientes. El ajuste se aplica SOLO a
    total_runs_pred en la inferencia diaria, no al entrenamiento.
    """
    if is_dome(weather_condition):
        return 0.0

    adj = 0.0

    # Ajuste por temperatura
    if temp is not None and temp > 0:
        temp_delta = (temp - 72) / 10.0
        adj += temp_delta * 0.15

    # Ajuste por viento
    if wind_speed >= 8:
        if wind_dir == "Out":
            adj += min(0.6, wind_speed * 0.03)    # ~0.3 a 10mph, ~0.45 a 15mph
        elif wind_dir == "In":
            adj -= min(0.5, wind_speed * 0.025)   # ~-0.25 a 10mph
        # Cross y None: sin ajuste significativo

    # Limitar para no distorsionar demasiado
    return max(-0.8, min(0.8, round(adj, 2)))
