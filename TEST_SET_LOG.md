# Registro de Evaluaciones en el Set de TEST (2025-2026)

Este archivo registra formal y automáticamente cada corrida del script `src/evaluate_test_set.py`.
Regla estricta: El set de test solo se toca cuando una fase de modelado está terminada.

---

## Historial de Corridas

### Corrida 1 (Baseline Inicial - Fase 4)
- **Fecha**: Previa a las 5 tareas de optimización (2026-08-05)
- **Motivo / Qué cambió**: Evaluación baseline de los modelos iniciales XGBoost/LightGBM.
- **Resultados**:
  - **Victoria (Win)**: Accuracy = `0.5460` | Log-loss = `0.6873` | Brier = `0.2471`
  - **Carreras (Runs)**: MAE = `3.603` | RMSE = `4.531`

---

### Corrida 2 (Post 5 Tareas de Calidad)
- **Fecha**: 2026-08-08
- **Motivo / Qué cambió**: Incorporación de park factors HR, feedback loop por ventana/bandas y modelos F5 dedicados.
- **Resultados**:
  - **Victoria (Win)**: Accuracy = `0.5453` | Log-loss = `0.6867` | Brier = `0.2468`
  - **Carreras (Runs)**: MAE = `3.598` | RMSE = `4.522`

---

### Corrida (2026-08-09 01:52:18)
- **Fecha**: 2026-08-09
- **Motivo / Qué cambió**: Evaluación en test tras revertir park_factor_hr del modelo de entrenamiento (cumpliendo regla de validación 2024)
- **Resultados**:
  - **Victoria (Win)**: Accuracy = `0.5485` | Log-loss = `0.6861` | Brier = `0.2465`
  - **Carreras (Runs)**: MAE = `3.601` | RMSE = `4.527`

---

### Corrida (2026-08-12 00:59:51)
- **Fecha**: 2026-08-12
- **Motivo / Qué cambió**: Primera y única evaluación del modelo de alineación vespertino (Fase 5). El modelo matutino permanece sin cambios.
- **Resultados**:
  - **Victoria (Win)**: Accuracy = `0.5620` | Log-loss = `0.6842` | Brier = `0.2455`
  - **Carreras (Runs)**: MAE = `3.564` | RMSE = `4.520`
