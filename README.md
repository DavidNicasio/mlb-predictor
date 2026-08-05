# Pipeline de datos MLB — Fase 2

Extracción y limpieza diaria de datos para el modelo de predicción de partidos.
Fuentes: MLB Stats API (calendario, boxscores, abridores probables) y
Baseball Savant/Statcast (batted balls). No depende de scraping de
FanGraphs ni Baseball-Reference: las métricas avanzadas (FIP, xFIP, wOBA)
se calculan localmente desde los datos crudos (ver `src/metrics.py`).

## Estructura

```
mlb_predictor/
├── .github/workflows/daily_pipeline.yml   # corre el pipeline cada mañana
├── src/
│   ├── db.py                 # esquema SQLite
│   ├── extract_schedule.py   # MLB Stats API: calendario, probables, boxscores
│   ├── extract_statcast.py   # Baseball Savant: batted balls
│   ├── metrics.py            # fórmulas de FIP, xFIP, wOBA
│   └── pipeline.py           # orquestador diario
├── data/mlb.db                # se crea solo, se commitea automáticamente
└── requirements.txt
```

## Uso local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/pipeline.py data/mlb.db
```

## Backfill histórico (una sola vez, corre en tu máquina, NO en Actions)

```bash
python src/backfill.py                       # 2015 -> temporada actual, ~1-2 horas
python src/backfill.py --season 2021         # una sola temporada
python src/backfill.py --workers 10          # más paralelismo si tu conexión aguanta
```

Es resumible: si lo interrumpes, vuelve a correr el mismo comando y salta
todo lo que ya esté cargado (usa la tabla `ingestion_log` para saber qué
rangos de Statcast ya se revisaron, y los `game_pk` ya presentes en
`boxscore_pitching` para no re-descargar boxscores).

## Inspeccionar la base

```bash
python src/inspect_db.py                     # resumen de filas por tabla
python src/inspect_db.py --table games --limit 10
```
(Alternativa a instalar el CLI de `sqlite3`, que en Windows no viene por defecto.)

## Park factors (calculados localmente, sin FanGraphs)

```bash
python src/compute_park_factors.py           # ventana de 3 años por defecto
```

Calcula `factor_runs` por venue y temporada comparando, para cada equipo,
sus carreras anotadas de local contra las mismas de visitante (método
clásico, sin depender de scraping). `factor_hr` y los splits por lado del
bateador quedan pendientes para una siguiente iteración (Fase 3).

## Automatización

El workflow `.github/workflows/daily_pipeline.yml` corre todos los días a
las 13:00 UTC (9am ET) vía GitHub Actions, y hace commit de `data/mlb.db`
actualizado al propio repositorio. También se puede lanzar manualmente
desde la pestaña "Actions" del repo (botón "Run workflow").

## Fix importante: bats/throws de jugadores

El boxscore de la MLB Stats API NO trae la mano del bateador/lanzador
(confirmado contra la API real: el `person` embebido ahí es solo
`{id, fullName, link, boxscoreName}`). Por eso `home_abridor_throws`
salía siempre `None`. Se corrigió con un extractor aparte:

```bash
python src/extract_players.py
```

Consulta el endpoint `/people` en bloques de 100 IDs (no reconstruye
boxscores, solo pide bio de los jugadores que ya aparecen en tu base) —
del orden de 20-30 llamadas para toda la historia, no ~29,000. Corre
esto UNA VEZ después del backfill, y ya queda integrado en
`pipeline.py`/`backfill.py` para jugadores nuevos que vayan apareciendo.

## Fase 3: features (ya implementada)

- `features_offense.py` — wOBA rolling 7/15/30 días por equipo, con split
  vs mano del abridor rival, y shrinkage hacia el wOBA de liga del mismo
  período si la muestra es chica.
- `features_pitching.py` — FIP/xFIP del abridor en sus últimas N
  aperturas (no por rango de fechas), días de descanso y pitch count de
  su salida anterior.
- `features_bullpen.py` — FIP del bullpen ponderado por qué tan reciente
  fue cada apertura de relevo (decae exponencialmente), con shrinkage
  hacia el ERA de liga si hay pocas entradas, más un índice de fatiga
  (outs lanzados en los últimos 1/2/3 días). No identifica "el cerrador"
  como rol individual (no rastreamos saves todavía) — usa fatiga general
  del bullpen como proxy.
- `features_rest.py` — días de descanso del equipo, densidad de
  calendario (partidos en los últimos 7 días) y cambio de sede (proxy
  simple de viaje).
- `features.py` — compilador: `build_features_for_date(conn, fecha)`
  arma una fila por partido programado ese día, combinando los 4
  módulos anteriores con prefijo `home_`/`away_`, lista para Fase 4.

Probar contra una fecha histórica real (para ver números de verdad, no
solo partidos de hoy sin abridores confirmados):

```bash
python src/features.py --date 2025-06-15
```

## Fase 4 (arrancando): dataset de entrenamiento

```bash
python src/build_training_dataset.py
```

Corre `build_features_for_date()` sobre todo el historial cargado
(2015-presente), guarda cada fila en la tabla `game_features` (resumible
por `game_pk` -- si lo interrumpes, retoma donde se quedó) y exporta todo
a `data/training_dataset.parquet`, listo para cargar con pandas y
entrenar XGBoost/LightGBM. Solo incluye partidos `Final` (se necesita el
resultado real como variable objetivo). No usa red -- son puras consultas
locales a SQLite, así que corre bastante más rápido que el backfill.

Para un rango específico: `python src/build_training_dataset.py --start-date 2023-01-01`

## Fase 4: modelo

- `model_data.py` — carga el parquet, define `home_win`/`total_runs` como
  variables objetivo, separa train (2015-2023) / val (2024) / test
  (2025-2026) **por temporada** (nunca al azar), y arma el set de
  features quitando fuga de información (resultado, IDs de partido/equipo).
- `train_baseline.py` — baselines antes de XGBoost/LightGBM: "el local
  siempre gana" + regresión logística (clasificación), promedio
  histórico + regresión lineal (Over/Under). Corre esto primero:

```bash
python src/train_baseline.py
```

- `model_search.py` — búsqueda aleatoria de hiperparámetros (evaluada
  en validación temporal, nunca k-fold al azar).
- `train_win_model.py` — entrena y afina XGBoost y LightGBM para
  probabilidad de victoria (`home_win`), compara ambos, guarda el mejor
  en `data/model_win.joblib`. A diferencia del baseline, **no imputa
  NaN** — los árboles usan la ausencia del dato como señal.

```bash
python src/train_win_model.py
```

- `train_runs_model.py` — mismo patrón que `train_win_model.py`, pero
  para carreras totales (`total_runs`, Over/Under): XGBoost + LightGBM,
  búsqueda de hiperparámetros, calibración por rango de predicción,
  guarda el mejor en `data/model_runs.joblib`.

```bash
python src/train_runs_model.py
```

Ninguno de los dos toca el set de test (2025-2026) todavía -- eso se
hace una sola vez, al final, cuando ya se eligieron ambos modelos.

- `evaluate_test_set.py` — el último paso: evalúa los dos modelos ya
  elegidos contra el set de test (2025-2026), la primera y única vez
  que se toca en todo el proceso.

```bash
python src/evaluate_test_set.py
```

## Fase 5: inferencia diaria

- `extract_teams.py` — nombres de equipo (una sola llamada a la API,
  ~30 equipos), para que los reportes digan "Yankees" en vez de "147".
  Ya integrado en `pipeline.py`.
- `predict_today.py` — toma los partidos de una fecha (por defecto,
  hoy), arma sus features y saca probabilidad de victoria + proyección
  de Over/Under con los modelos ya entrenados. Cada predicción queda
  registrada en `predictions_log` para poder comparar después contra
  el resultado real.

```bash
python src/predict_today.py                    # hoy
python src/predict_today.py --date 2026-08-05   # cualquier fecha
```

El workflow de GitHub Actions ya corre esto automáticamente cada
mañana después del pipeline de datos, y guarda el reporte en
`predictions/YYYY-MM-DD.txt` dentro del propio repo.

**Importante**: `data/model_win.joblib` y `data/model_runs.joblib`
tienen que estar commiteados al repo para que Actions los encuentre —
no se generan solos ahí, se entrenan localmente (`train_win_model.py` /
`train_runs_model.py`) y se suben una vez.

## Base completa (local) vs. base recortada (GitHub)

`data/mlb.db` con toda la historia (2015-presente, Statcast incluido)
pesa cientos de MB — por encima del límite práctico de GitHub para un
archivo normal de git. La automatización diaria no necesita esa
historia completa (los rolling de Fase 3 miran para atrás 30-60 días
como mucho), así que:

- `data/mlb.db` (completa) se queda **solo en tu PC** (ya en
  `.gitignore`). Úsala para el backfill, para reentrenar modelos, para
  construir el dataset de entrenamiento.
- `data/mlb_recent.db` (los últimos ~100 días) es la que sí se sube a
  GitHub y usa la automatización diaria.

**Antes de subir el repo por primera vez**, genera la versión recortada:

```bash
python src/export_recent_window.py
```

Esto crea `data/mlb_recent.db`. Súbela junto con los `.joblib` de los
modelos:

```bash
git add data/mlb_recent.db data/model_win.joblib data/model_runs.joblib
git commit -m "Base recortada + modelos entrenados"
git push
```

El workflow de Actions ya se encarga de mantenerla al día: corre
`pipeline.py` sobre `mlb_recent.db`, la poda de vuelta a ~100 días
(`export_recent_window.py --prune-only`) para que no crezca sin
límite, y genera las predicciones -- todo commiteado automáticamente
cada día.
