"""
model_search.py
Búsqueda aleatoria de hiperparámetros, genérica para cualquier modelo
sklearn-compatible (XGBoost, LightGBM). Se evalúa contra el set de
VALIDACIÓN temporal (2024) -- nunca k-fold aleatorio, porque mezclar
fechas al azar filtraría información del futuro hacia el pasado.

No es una búsqueda exhaustiva (grid completo): para gradient boosting,
una búsqueda aleatoria de 20-30 combinaciones suele acercarse bastante
al óptimo con una fracción del costo computacional.
"""

from __future__ import annotations

import random


def random_search(model_class, param_space: dict, fixed_params: dict,
                   X_train, y_train, X_val, y_val, score_fn,
                   higher_is_better: bool, n_iter: int = 25,
                   random_state: int = 42, label: str = "") -> tuple[dict, list[dict]]:
    """Devuelve (best, history). best = {'score','params','model'}."""
    rng = random.Random(random_state)
    best = {"score": None, "params": None, "model": None}
    history = []

    for i in range(n_iter):
        params = {k: rng.choice(v) for k, v in param_space.items()}
        model = model_class(**params, **fixed_params)
        model.fit(X_train, y_train)
        score = score_fn(model, X_val, y_val)
        history.append({**params, "score": score})

        is_better = (
            best["score"] is None
            or (higher_is_better and score > best["score"])
            or (not higher_is_better and score < best["score"])
        )
        if is_better:
            best.update(score=score, params=params, model=model)

        marker = " <- mejor hasta ahora" if is_better else ""
        print(f"  [{label} {i+1}/{n_iter}] score={score:.4f}{marker}")

    return best, history
