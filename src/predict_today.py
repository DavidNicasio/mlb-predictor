"""
predict_today.py
Fase 5: inferencia diaria. Toma los partidos de una fecha (por defecto,
hoy), arma sus features con build_features_for_date() (Fase 3), y saca
probabilidad de victoria + proyección de Over/Under con los modelos ya
entrenados (Fase 4). Registra cada predicción en `predictions_log` para
poder comparar después contra el resultado real y genera un reporte en PDF.

Uso:
    python src/predict_today.py                    # hoy
    python src/predict_today.py --date 2026-08-05   # cualquier fecha
    python src/predict_today.py --pdf reports/hoy.pdf
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd

import db
import features
import pdf_generator


def team_info(conn, team_id) -> tuple[str, str]:
    if team_id is None:
        return ("?", "?")
    row = conn.execute("SELECT name, abbreviation FROM teams WHERE team_id=?", (team_id,)).fetchone()
    return (row[0], row[1]) if row else (f"Team {team_id}", "?")


TEXT_COLUMNS = {
    "game_date", "game_date_utc", "status", "home_abridor_throws",
    "away_abridor_throws", "weather_condition", "weather_wind"
}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col not in TEXT_COLUMNS and col != "game_pk":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def predict_games(conn, rows: list[dict], win_saved: dict, runs_saved: dict) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = _coerce_numeric(pd.DataFrame(rows))

    X_win = df.reindex(columns=win_saved["feature_names"])
    X_runs = df.reindex(columns=runs_saved["feature_names"])

    df["home_win_proba"] = win_saved["model"].predict_proba(X_win)[:, 1]
    df["total_runs_pred"] = runs_saved["model"].predict(X_runs)

    import feedback_loop
    df = feedback_loop.apply_feedback_corrections(df, conn)

    # Cargar modelos dedicados F5 si están disponibles
    f5_runs_path = Path("data/model_f5_runs.joblib")
    f5_win_path = Path("data/model_f5_win.joblib")

    if f5_runs_path.exists() and f5_win_path.exists():
        f5_r_saved = joblib.load(f5_runs_path)
        f5_w_saved = joblib.load(f5_win_path)
        X_f5_r = df.reindex(columns=f5_r_saved["feature_names"])
        X_f5_w = df.reindex(columns=f5_w_saved["feature_names"])
        df["f5_total_runs_pred"] = f5_r_saved["model"].predict(X_f5_r)
        df["f5_home_win_proba"] = f5_w_saved["model"].predict_proba(X_f5_w)[:, 1]
    else:
        import features_f5
        f5_res = [
            features_f5.calculate_f5_projections(
                r.get("home_fip"), r.get("away_fip"),
                r.get("home_woba_vs_hand"), r.get("away_woba_vs_hand"),
                float(r["total_runs_pred"]), float(r["home_win_proba"])
            ) for _, r in df.iterrows()
        ]
        df["f5_total_runs_pred"] = [f["f5_total_runs_pred"] for f in f5_res]
        df["f5_home_win_proba"] = [f["f5_home_win_proba"] for f in f5_res]

    df["home_name"] = df["home_team_id"].apply(lambda t: team_info(conn, t)[0])
    df["home_abbr"] = df["home_team_id"].apply(lambda t: team_info(conn, t)[1])
    df["away_name"] = df["away_team_id"].apply(lambda t: team_info(conn, t)[0])
    df["away_abbr"] = df["away_team_id"].apply(lambda t: team_info(conn, t)[1])

    return df


def print_report(df: pd.DataFrame, target_date: str) -> None:
    if df.empty:
        print(f"No hay partidos programados para {target_date}.")
        return

    print(f"\n{'=' * 78}")
    print(f"  Predicciones para {target_date} ({len(df)} partido(s))")
    print(f"{'=' * 78}\n")

    df = df.copy()
    df["_confianza"] = (df["home_win_proba"] - 0.5).abs()
    for _, r in df.sort_values("_confianza", ascending=False).iterrows():
        proba = r["home_win_proba"]
        favorito = r["home_name"] if proba >= 0.5 else r["away_name"]
        proba_favorito = proba if proba >= 0.5 else 1 - proba

        print(f"  {r['away_name']}  @  {r['home_name']}")
        print(f"    Probabilidad de victoria local: {proba:.1%}   "
              f"(favorito: {favorito} a {proba_favorito:.1%})")
        print(f"    Proyección Over/Under (carreras totales): {r['total_runs_pred']:.1f}")

        if pd.notna(r.get("home_abridor_id")):
            print(f"    Abridor local ({r.get('home_abridor_throws') or '?'}): "
                  f"FIP {r.get('home_fip')}, {int(r.get('home_n_starts') or 0)} aperturas recientes")
        else:
            print("    Abridor local: sin confirmar todavía")

        if pd.notna(r.get("away_abridor_id")):
            print(f"    Abridor visitante ({r.get('away_abridor_throws') or '?'}): "
                  f"FIP {r.get('away_fip')}, {int(r.get('away_n_starts') or 0)} aperturas recientes")
        else:
            print("    Abridor visitante: sin confirmar todavía")
        print()


def log_predictions(conn, df: pd.DataFrame, win_saved: dict, runs_saved: dict) -> None:
    if df.empty:
        return
    now = datetime.now().isoformat(timespec="seconds")
    rows = [{
        "game_pk": int(r["game_pk"]),
        "predicted_at": now,
        "home_win_proba": float(r["home_win_proba"]),
        "total_runs_pred": float(r["total_runs_pred"]),
        "win_model_type": win_saved["model_type"],
        "runs_model_type": runs_saved["model_type"],
        "weather_temp": int(r["weather_temp"]) if pd.notna(r.get("weather_temp")) else None,
        "weather_wind": str(r["weather_wind"]) if pd.notna(r.get("weather_wind")) else None,
    } for _, r in df.iterrows()]
    conn.executemany(
        """INSERT OR REPLACE INTO predictions_log
           (game_pk, predicted_at, home_win_proba, total_runs_pred,
            win_model_type, runs_model_type, weather_temp, weather_wind)
           VALUES (:game_pk, :predicted_at, :home_win_proba, :total_runs_pred,
                   :win_model_type, :runs_model_type, :weather_temp, :weather_wind)""",
        rows,
    )
    conn.commit()
    print(f"({len(rows)} predicción(es) guardadas en predictions_log)")


def run(target_date: str | None = None, db_path: str = "data/mlb.db",
        win_model_path: str = "data/model_win.joblib",
        runs_model_path: str = "data/model_runs.joblib",
        pdf_output: str | None = None) -> pd.DataFrame:
    target_date = target_date or str(date.today())
    pdf_output = pdf_output or f"reports/predictions_{target_date}.pdf"

    conn = db.get_connection(db_path)
    db.init_db(conn)

    rows = features.build_features_for_date(conn, target_date)
    win_saved = joblib.load(win_model_path)
    runs_saved = joblib.load(runs_model_path)

    df = predict_games(conn, rows, win_saved, runs_saved)
    print_report(df, target_date)
    log_predictions(conn, df, win_saved, runs_saved)

    if not df.empty:
        pdf_path = pdf_generator.build_daily_predictions_pdf(pdf_output, target_date, df)
        print(f"Reporte PDF de predicciones generado en: {pdf_path}")

    conn.close()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicciones diarias de victoria y Over/Under")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD, por defecto hoy")
    parser.add_argument("--db-path", default="data/mlb.db")
    parser.add_argument("--win-model", default="data/model_win.joblib")
    parser.add_argument("--runs-model", default="data/model_runs.joblib")
    parser.add_argument("--pdf", default=None, help="Ruta del PDF de predicciones de salida")
    args = parser.parse_args()
    run(args.date, args.db_path, args.win_model, args.runs_model, args.pdf)
