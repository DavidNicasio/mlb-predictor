"""
backtest_props.py
Evaluación estadística de las reglas de recomendación de apuestas (_best_prop_recommendation).
Analiza todas las predicciones registradas en predictions_log + games + game_linescore.

Mide para cada regla:
  - n: cantidad de veces que la regla se activó
  - hit_rate: porcentaje de acierto de la regla
  - baseline: tasa base histórica sin regla
  - edge: diferencial de acierto sobre el baseline
  - status: "CONFIABLE", "DUDOSA" o "MUESTRA_INSUFICIENTE" (<30 casos)
"""

from __future__ import annotations

import argparse
import pandas as pd

import db


MIN_SAMPLE_SIZE = 30


def run_backtest(db_path: str = "data/mlb.db") -> pd.DataFrame:
    conn = db.get_connection(db_path)
    db.init_db(conn)

    # 1. Cargar predicciones DEDUPLICADAS de predictions_log (última predicción por partido)
    query_pred = """
        SELECT
            g.game_pk,
            p.home_win_proba,
            p.total_runs_pred,
            g.game_date,
            g.home_score,
            g.away_score,
            (g.home_score + g.away_score) AS actual_total,
            CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS home_won
        FROM games g
        JOIN (
            SELECT game_pk, MAX(predicted_at) AS max_pred
            FROM predictions_log
            GROUP BY game_pk
        ) latest ON latest.game_pk = g.game_pk
        JOIN predictions_log p ON p.game_pk = latest.game_pk AND p.predicted_at = latest.max_pred
        WHERE g.status = 'Final' AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
    """
    pred_df = pd.read_sql_query(query_pred, conn)

    # 2. Cargar datos históricos de games, linescore y FIPs de abridores
    query_hist = """
        SELECT
            g.game_pk,
            g.game_date,
            g.home_score,
            g.away_score,
            gf.home_fip,
            gf.away_fip,
            (g.home_score + g.away_score) AS actual_total,
            CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS home_won,
            l1.home_runs AS inn1_home,
            l1.away_runs AS inn1_away,
            f5.home_f5,
            f5.away_f5
        FROM games g
        JOIN game_features gf ON gf.game_pk = g.game_pk
        LEFT JOIN game_linescore l1 ON l1.game_pk = g.game_pk AND l1.inning = 1
        LEFT JOIN (
            SELECT game_pk,
                   SUM(home_runs) AS home_f5,
                   SUM(away_runs) AS away_f5
            FROM game_linescore
            WHERE inning <= 5
            GROUP BY game_pk
        ) f5 ON f5.game_pk = g.game_pk
        WHERE g.status = 'Final' AND g.game_type = 'R' AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
    """
    hist_df = pd.read_sql_query(query_hist, conn)
    conn.close()

    if pred_df.empty and hist_df.empty:
        print("No hay partidos para evaluar.")
        return pd.DataFrame()

    total_pred_games = len(pred_df)
    total_hist_games = len(hist_df)
    print(f"\n========================================================")
    print(f"  BACKTESTING DE REGLAS DE RECOMENDACIÓN")
    print(f"  - Partidos con predicción previa única (ML/Total Runs): {total_pred_games}")
    print(f"  - Partidos con datos históricos FIP/Linescore: {total_hist_games}")
    print(f"========================================================\n")

    results = []

    # ----------------------------------------------------
    # Regla 1: Victoria Directa Local (Proba >= 60%)
    # ----------------------------------------------------
    base_home_win = float((pred_df["home_won"] == 1).mean()) if len(pred_df) > 0 else 0.50

    r1_df = pred_df[pred_df["home_win_proba"] >= 0.60]
    n1 = len(r1_df)
    if n1 > 0:
        hits1 = int((r1_df["home_won"] == 1).sum())
        hr1 = hits1 / n1
        edge1 = hr1 - base_home_win
        status1 = "CONFIABLE" if n1 >= MIN_SAMPLE_SIZE and edge1 > 0.03 else ("DUDOSA" if n1 >= MIN_SAMPLE_SIZE else "MUESTRA_INSUFICIENTE")
        results.append({
            "regla": "ML Favorito Local (Proba >= 60%)",
            "n": n1,
            "hits": hits1,
            "hit_rate": hr1,
            "baseline": base_home_win,
            "edge": edge1,
            "status": status1,
        })

    # ----------------------------------------------------
    # Regla 2: Victoria Directa Visitante (Proba <= 40%)
    # ----------------------------------------------------
    r2_df = pred_df[pred_df["home_win_proba"] <= 0.40]
    n2 = len(r2_df)
    if n2 > 0:
        hits2 = int((r2_df["home_won"] == 0).sum())
        hr2 = hits2 / n2
        base2 = 1.0 - base_home_win
        edge2 = hr2 - base2
        status2 = "CONFIABLE" if n2 >= MIN_SAMPLE_SIZE and edge2 > 0.03 else ("DUDOSA" if n2 >= MIN_SAMPLE_SIZE else "MUESTRA_INSUFICIENTE")
        results.append({
            "regla": "ML Favorito Visitante (Proba <= 40%)",
            "n": n2,
            "hits": hits2,
            "hit_rate": hr2,
            "baseline": base2,
            "edge": edge2,
            "status": status2,
        })

    # ----------------------------------------------------
    # Regla 3: Over 8.5 carreras (Proy >= 8.5)
    # ----------------------------------------------------
    r3_df = pred_df[pred_df["total_runs_pred"] >= 8.5]
    n3 = len(r3_df)
    if n3 > 0:
        hits3 = int((r3_df["actual_total"] > 8.5).sum())
        hr3 = hits3 / n3
        base3 = float((pred_df["actual_total"] > 8.5).mean())
        edge3 = hr3 - base3
        status3 = "CONFIABLE" if n3 >= MIN_SAMPLE_SIZE and edge3 > 0.03 else ("DUDOSA" if n3 >= MIN_SAMPLE_SIZE else "MUESTRA_INSUFICIENTE")
        results.append({
            "regla": "Over 8.5 Carreras Totales",
            "n": n3,
            "hits": hits3,
            "hit_rate": hr3,
            "baseline": base3,
            "edge": edge3,
            "status": status3,
        })

    # ----------------------------------------------------
    # Regla 4: Under 8.5 carreras (Proy < 8.5)
    # ----------------------------------------------------
    r4_df = pred_df[pred_df["total_runs_pred"] < 8.5]
    n4 = len(r4_df)
    if n4 > 0:
        hits4 = int((r4_df["actual_total"] < 8.5).sum())
        hr4 = hits4 / n4
        base4 = float((pred_df["actual_total"] < 8.5).mean())
        edge4 = hr4 - base4
        status4 = "CONFIABLE" if n4 >= MIN_SAMPLE_SIZE and edge4 > 0.03 else ("DUDOSA" if n4 >= MIN_SAMPLE_SIZE else "MUESTRA_INSUFICIENTE")
        results.append({
            "regla": "Under 8.5 Carreras Totales",
            "n": n4,
            "hits": hits4,
            "hit_rate": hr4,
            "baseline": base4,
            "edge": edge4,
            "status": status4,
        })

    # ----------------------------------------------------
    # Regla 5: NRFI (Ambos FIP <= 3.65)
    # ----------------------------------------------------
    linescore_df = hist_df[hist_df["inn1_home"].notna() & hist_df["inn1_away"].notna()]
    n_linescore = len(linescore_df)
    if n_linescore > 0:
        base_nrfi = float(((linescore_df["inn1_home"] == 0) & (linescore_df["inn1_away"] == 0)).mean())

        r_nrfi = linescore_df[(linescore_df["home_fip"] <= 3.65) & (linescore_df["away_fip"] <= 3.65)]
        n_nrfi = len(r_nrfi)
        if n_nrfi > 0:
            hits_nrfi = int(((r_nrfi["inn1_home"] == 0) & (r_nrfi["inn1_away"] == 0)).sum())
            hr_nrfi = hits_nrfi / n_nrfi
            edge_nrfi = hr_nrfi - base_nrfi
            status_nrfi = "CONFIABLE" if n_nrfi >= MIN_SAMPLE_SIZE and edge_nrfi > 0.03 else ("DUDOSA" if n_nrfi >= MIN_SAMPLE_SIZE else "MUESTRA_INSUFICIENTE")
            results.append({
                "regla": "NRFI (Ambos FIP <= 3.65)",
                "n": n_nrfi,
                "hits": hits_nrfi,
                "hit_rate": hr_nrfi,
                "baseline": base_nrfi,
                "edge": edge_nrfi,
                "status": status_nrfi,
            })

    # ----------------------------------------------------
    # Regla 6: F5 Under 1.5 (Abridor Rival FIP <= 3.20)
    # ----------------------------------------------------
    f5_df = hist_df[hist_df["away_f5"].notna() & hist_df["home_f5"].notna()]
    if len(f5_df) > 0:
        base_f5_away = float((f5_df["away_f5"] <= 1).mean())
        base_f5_home = float((f5_df["home_f5"] <= 1).mean())

        r_f5_away = f5_df[f5_df["home_fip"] <= 3.20]
        n_f5_a = len(r_f5_away)
        if n_f5_a > 0:
            hits_f5_a = int((r_f5_away["away_f5"] <= 1).sum())
            hr_f5_a = hits_f5_a / n_f5_a
            edge_f5_a = hr_f5_a - base_f5_away
            status_f5_a = "CONFIABLE" if n_f5_a >= MIN_SAMPLE_SIZE and edge_f5_a > 0.03 else ("DUDOSA" if n_f5_a >= MIN_SAMPLE_SIZE else "MUESTRA_INSUFICIENTE")
            results.append({
                "regla": "F5 Vis. Under 1.5 (Abridor Local FIP <= 3.20)",
                "n": n_f5_a,
                "hits": hits_f5_a,
                "hit_rate": hr_f5_a,
                "baseline": base_f5_away,
                "edge": edge_f5_a,
                "status": status_f5_a,
            })

        r_f5_home = f5_df[f5_df["away_fip"] <= 3.20]
        n_f5_h = len(r_f5_home)
        if n_f5_h > 0:
            hits_f5_h = int((r_f5_home["home_f5"] <= 1).sum())
            hr_f5_h = hits_f5_h / n_f5_h
            edge_f5_h = hr_f5_h - base_f5_home
            status_f5_h = "CONFIABLE" if n_f5_h >= MIN_SAMPLE_SIZE and edge_f5_h > 0.03 else ("DUDOSA" if n_f5_h >= MIN_SAMPLE_SIZE else "MUESTRA_INSUFICIENTE")
            results.append({
                "regla": "F5 Loc. Under 1.5 (Abridor Vis. FIP <= 3.20)",
                "n": n_f5_h,
                "hits": hits_f5_h,
                "hit_rate": hr_f5_h,
                "baseline": base_f5_home,
                "edge": edge_f5_h,
                "status": status_f5_h,
            })

    res_df = pd.DataFrame(results)

    print(f"{'REGLA DE APUESTA':<38} {'N':>6} {'HITS':>6} {'HIT RATE':>10} {'BASELINE':>10} {'EDGE':>8} {'ESTADO':>22}")
    print("-" * 105)
    for _, r in res_df.iterrows():
        print(f"{r['regla']:<38} {int(r['n']):>6} {int(r['hits']):>6} {r['hit_rate']:>9.1%} {r['baseline']:>9.1%} {r['edge']:>+7.1%} {r['status']:>22}")
    print("\n")

    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtesting real de reglas de recomendación")
    parser.add_argument("--db-path", default="data/mlb.db")
    args = parser.parse_args()
    run_backtest(args.db_path)
