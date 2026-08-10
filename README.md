# MLB & LMB Predictor — Sistema de Predicciones y Análisis de Apuestas

Sistema integral de extracción de datos sabermétricos, ingeniería de características (features), modelos de Machine Learning (XGBoost / LightGBM) y aplicación de escritorio interactiva (`app.py`) para la predicción de partidos de béisbol de Grandes Ligas (**MLB**) y Liga Mexicana de Béisbol (**LMB**).

---

## 🏛️ Filosofía del Proyecto y Decisiones Arquitectónicas

1. **Sin scraping no autorizado**:
   - Toda la información proviene de endpoints públicos oficiales (`statsapi.mlb.com` y Baseball Savant/Statcast). No se realiza scraping sobre FanGraphs o Baseball-Reference.
   - Las constantes de liga (ERA, FIP Constant, wOBA Weights, HR/FB Rate) y Park Factors se calculan de manera local y determinista desde la base de datos histórica.

2. **Split Temporal Estricto en Machine Learning**:
   - **Train**: Temporadas 2015 – 2023.
   - **Validation**: Temporada 2024 (usada para calibración de modelos y selección de hiperparámetros).
   - **Test Set**: Temporadas 2025 – 2026. **El conjunto de test se evalúa bajo demanda estricta y se registra en `TEST_SET_LOG.md` para evitar sobreajuste.**

3. **Cero Contaminación de Datos y Regresión Bayesiana**:
   - Todas las características proyectadas para una fecha dada utilizan estrictamente información registrada antes de la hora del partido.
   - Se utiliza **regresión bayesiana simple (shrinkage)** hacia promedios de la liga para reducir el ruido en muestras pequeñas (ej. abridores con pocas aperturas o bullpens con pocas entradas).

---

## 📂 Estructura del Repositorio

```
mlb_predictor/
├── .github/workflows/
│   └── daily_pipeline.yml         # Automatización diaria en GitHub Actions
├── app.py                         # GUI interactiva en CustomTkinter con Hilos + PDF Viewer
├── src/
│   ├── config.py                  # Centralización de umbrales y constantes del sistema
│   ├── db.py                      # Esquema SQLite (WAL mode, índices y foreign keys)
│   ├── extract_schedule.py        # Extractor de calendarios, abridores y marcadores
│   ├── extract_statcast.py        # Extractor de Statcast Batted Balls
│   ├── weather_parser.py          # Extractor y parser vectorial de clima/viento
│   ├── metrics.py                 # Fórmulas sabermétricas (FIP, xFIP, wOBA, Shrinkage)
│   ├── features.py                # Compilador maestro de características por partido
│   ├── features_offense.py        # wOBA rolling por equipo (7/15/30 días)
│   ├── features_pitching.py       # FIP/xFIP del abridor titular en últimas aperturas
│   ├── features_bullpen.py        # Fatiga e índice FIP ponderado del bullpen
│   ├── features_f5.py             # Proyecciones para primeras 5 entradas (F5)
│   ├── feedback_loop.py           # Calibración móvil (75 días) por bandas de probabilidad
│   ├── pipeline.py                # Orquestador de actualización diaria de datos
│   ├── build_training_dataset.py  # Generador multihilo de Parquet (training_dataset.parquet)
│   ├── model_data.py              # Carga y divisiones temporales del dataset
│   ├── train_win_model.py         # Entrenamiento y afinación del modelo de victoria (XGBoost)
│   ├── train_runs_model.py        # Entrenamiento del modelo Over/Under de carreras totales
│   ├── train_f5_model.py          # Modelo dedicado F5 (Primeras 5 entradas)
│   ├── compute_park_factors.py    # Park Factors locales (runs, HR, splits L/R)
│   ├── backtest_props.py          # Evaluación histórica deduplicada de reglas de apuestas
│   ├── predict_today.py           # Generador diario de predicciones y exportador PDF
│   ├── report_card.py             # Evaluador de aciertos reales vs proyecciones
│   └── pdf_generator.py           # Motor de diseño PDF (ReportLab) con escudos HD
├── tests/
│   ├── run_tests.py               # Ejecutor de la suite de pruebas unitarias
│   ├── test_metrics.py            # Pruebas para formulas de FIP/wOBA/Shrinkage
│   ├── test_report_card.py       # Pruebas para evaluación de partidos (Final vs Scheduled)
│   ├── test_features_f5.py        # Pruebas para límites e intervalos F5
│   └── test_pipeline_and_models.py# Pruebas de esquema de BD y extractores
├── assets/logos/                  # Logos PNG HD de equipos de MLB y LMB
├── data/                          # Almacenamiento local de mlb.db y datasets
├── reports/                       # Reportes PDF generados (predictions_mlb_*.pdf, report_*.pdf)
└── TEST_SET_LOG.md                # Bitácora estricta de evaluaciones en el Test Set
```

---

## 🚀 Uso Rápido y Comandos

### 1. Entorno e Instalación

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Aplicación de Escritorio Interactiva (GUI Dashboard)

```powershell
python app.py
```
Ofrece:
- **Vista dual**: Tabla Grid Estilo PDF o Tarjetas Interactivas.
- **Hilos asíncronos**: Ejecución no bloqueante con animación `CTkProgressBar`.
- **Estadísticas acumuladas**: Resumen inmediato de aciertos en el encabezado.
- **Contadores de riesgo**: Visualización diaria de partidos por categoría (Bajo, Medio, Alto).
- **Persistencia**: Preferencias guardadas en `.app_state.json`.

---

## 🛠️ Ejecución de Scripts Técnicos

### Actualización Manual de Datos y Predicciones del Día
```powershell
python src/pipeline.py data/mlb.db
python src/predict_today.py --date 2026-08-09 --league MLB
python src/report_card.py --date 2026-08-09
```

### Backtesting de Reglas de Apuestas (Deduplicado por Partido)
```powershell
python src/backtest_props.py
```

### Recálculo de Park Factors Locales
```powershell
python src/compute_park_factors.py --window-years 3
```

### Reconstrucción del Dataset de Entrenamiento y Reentrenamiento de Modelos
```powershell
python src/build_training_dataset.py
python src/train_win_model.py --n-iter 20
python src/train_runs_model.py --n-iter 20
python src/train_f5_model.py --n-iter 20
```

---

## 🧪 Pruebas Unitarias

El proyecto cuenta con una suite automatizada sin dependencias externas complejas:

```powershell
python tests/run_tests.py
```

---

## 🤖 Automatización Continua (GitHub Actions)

El archivo `.github/workflows/daily_pipeline.yml` se ejecuta automáticamente todos los días a las 13:00 UTC (9:00 AM ET):
1. Descarga abridores confirmados y partidos recientes.
2. Califica las predicciones de partidos finalizados (`report_card.py`).
3. Genera las proyecciones del día (`predict_today.py`).
4. Realiza commit y push automático de la base de datos `data/mlb_recent.db` y los PDF en `reports/`.
