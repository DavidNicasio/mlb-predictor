"""
weather_parser.py
Codificación numérica de temperatura y viento para el modelo de predicciones de MLB.
"""

from __future__ import annotations

import re


def parse_weather_features(weather_temp: float | int | None, weather_wind: str | None) -> dict:
    """Transforma la temperatura y el texto de viento en variables numéricas razonables:
    - weather_temp: Temperatura continua en Fahrenheit (por defecto 72.0)
    - wind_speed_mph: Velocidad del viento en mph (continuo)
    - wind_dir_dummy: Categoría direccional (+1.0 Out, -1.0 In, 0.0 Cross/Calm/Domos)
    - wind_effect_runs: Efecto vectorial neto = wind_speed_mph * wind_dir_dummy
    """
    temp_val = float(weather_temp) if weather_temp is not None else 72.0

    wind_speed = 0.0
    wind_dir_dummy = 0.0

    if weather_wind:
        match = re.search(r"(\d+)\s*mph", str(weather_wind), re.IGNORECASE)
        if match:
            wind_speed = float(match.group(1))

        wind_str = str(weather_wind).upper()
        if "OUT TO" in wind_str:
            wind_dir_dummy = 1.0
        elif "IN FROM" in wind_str:
            wind_dir_dummy = -1.0
        else:
            wind_dir_dummy = 0.0

    wind_effect = wind_speed * wind_dir_dummy

    return {
        "weather_temp": temp_val,
        "wind_speed_mph": wind_speed,
        "wind_dir_dummy": wind_dir_dummy,
        "wind_effect_runs": wind_effect,
    }
