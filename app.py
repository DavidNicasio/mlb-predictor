"""
app.py
Aplicación de Escritorio Interactiva (GUI Dashboard) para MLB Predictor.
Permite gestionar la actualización de datos (pipeline), generar predicciones diarias,
calificar partidos y abrir reportes en PDF con un solo clic.
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import date, timedelta
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image

# Asegurar import de los módulos locales en src/
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import db
import pipeline
import predict_today
import report_card

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

LOGOS_DIR = Path(__file__).parent / "assets" / "logos"


class MLBPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MLB Predictor Dashboard")
        self.geometry("1100 x 750")
        self.minsize(950, 650)

        self.target_date = str(date.today())

        self._create_layout()
        self._load_day_summary()

    def _create_layout(self):
        # Header Frame
        self.header_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#0f172a")
        self.header_frame.pack(fill="x", padx=15, pady=(15, 10))

        title_label = ctk.CTkLabel(
            self.header_frame,
            text="⚾ MLB Predictor Dashboard",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#f8fafc",
        )
        title_label.pack(side="left", padx=20, pady=15)

        self.status_label = ctk.CTkLabel(
            self.header_frame,
            text="Listo | Sistema preparado",
            font=ctk.CTkFont(size=12),
            text_color="#38bdf8",
        )
        self.status_label.pack(side="right", padx=20, pady=15)

        # Control Panel (Fechas y Acciones)
        self.control_frame = ctk.CTkFrame(self, corner_radius=10)
        self.control_frame.pack(fill="x", padx=15, pady=5)

        # Controles de Fecha
        date_subframe = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        date_subframe.pack(side="left", padx=15, pady=12)

        lbl_date = ctk.CTkLabel(date_subframe, text="Fecha:", font=ctk.CTkFont(size=13, weight="bold"))
        lbl_date.pack(side="left", padx=(0, 8))

        self.btn_yesterday = ctk.CTkButton(date_subframe, text="Ayer", width=60, command=self._set_yesterday)
        self.btn_yesterday.pack(side="left", padx=3)

        self.btn_today = ctk.CTkButton(date_subframe, text="Hoy", width=60, command=self._set_today)
        self.btn_today.pack(side="left", padx=3)

        self.btn_tomorrow = ctk.CTkButton(date_subframe, text="Mañana", width=65, command=self._set_tomorrow)
        self.btn_tomorrow.pack(side="left", padx=3)

        self.entry_date = ctk.CTkEntry(date_subframe, width=110, font=ctk.CTkFont(size=13))
        self.entry_date.insert(0, self.target_date)
        self.entry_date.pack(side="left", padx=(8, 3))

        btn_go_date = ctk.CTkButton(date_subframe, text="📅 Ir", width=50, command=self._on_custom_date)
        btn_go_date.pack(side="left", padx=3)

        # Action Buttons Frame
        action_subframe = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        action_subframe.pack(side="right", padx=15, pady=12)

        self.btn_pipeline = ctk.CTkButton(
            action_subframe,
            text="🔄 Actualizar Datos MLB",
            fg_color="#0284c7",
            hover_color="#0369a1",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._run_pipeline_thread,
        )
        self.btn_pipeline.pack(side="left", padx=4)

        self.btn_predict = ctk.CTkButton(
            action_subframe,
            text="📊 Predicciones (PDF)",
            fg_color="#059669",
            hover_color="#047857",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._run_predict,
        )
        self.btn_predict.pack(side="left", padx=4)

        self.btn_report = ctk.CTkButton(
            action_subframe,
            text="🏆 Report Card (PDF)",
            fg_color="#d97706",
            hover_color="#b45309",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._run_report,
        )
        self.btn_report.pack(side="left", padx=4)

        # Quick Stats & Open PDF bar
        self.stats_bar = ctk.CTkFrame(self, corner_radius=10, fg_color="#1e293b")
        self.stats_bar.pack(fill="x", padx=15, pady=5)

        self.lbl_stats = ctk.CTkLabel(
            self.stats_bar,
            text="Resumen del día cargado",
            font=ctk.CTkFont(size=13),
            text_color="#f1f5f9",
        )
        self.lbl_stats.pack(side="left", padx=15, pady=10)

        pdf_open_subframe = ctk.CTkFrame(self.stats_bar, fg_color="transparent")
        pdf_open_subframe.pack(side="right", padx=15, pady=5)

        self.btn_open_pred_pdf = ctk.CTkButton(
            pdf_open_subframe,
            text="👁️ Abrir PDF Predicciones",
            width=150,
            fg_color="#334155",
            hover_color="#475569",
            command=self._open_predictions_pdf,
        )
        self.btn_open_pred_pdf.pack(side="left", padx=4)

        self.btn_open_rep_pdf = ctk.CTkButton(
            pdf_open_subframe,
            text="👁️ Abrir PDF Report Card",
            width=150,
            fg_color="#334155",
            hover_color="#475569",
            command=self._open_report_pdf,
        )
        self.btn_open_rep_pdf.pack(side="left", padx=4)

        # Main Scrollable Games List Area
        self.scroll_games = ctk.CTkScrollableFrame(self, label_text="Partidos y Predicciones de la Fecha")
        self.scroll_games.pack(fill="both", expand=True, padx=15, pady=(5, 15))

    # --- Date Handlers ---
    def _set_date(self, new_date: str):
        self.target_date = new_date
        self.entry_date.delete(0, tk.END)
        self.entry_date.insert(0, new_date)
        self._load_day_summary()

    def _set_yesterday(self):
        self._set_date(str(date.today() - timedelta(days=1)))

    def _set_today(self):
        self._set_date(str(date.today()))

    def _set_tomorrow(self):
        self._set_date(str(date.today() + timedelta(days=1)))

    def _on_custom_date(self):
        val = self.entry_date.get().strip()
        if val:
            self._set_date(val)

    # --- Worker Threads & Logic ---
    def _update_status(self, msg: str):
        self.status_label.configure(text=msg)

    def _run_pipeline_thread(self):
        self.btn_pipeline.configure(state="disabled")
        self._update_status("⏳ Actualizando datos MLB desde la API...")

        def task():
            try:
                conn = db.get_connection("data/mlb.db")
                db.init_db(conn)
                pipeline.run("data/mlb.db")
                conn.close()
                self.after(0, lambda: self._on_pipeline_success())
            except Exception as err:
                self.after(0, lambda: self._on_pipeline_error(str(err)))

        threading.Thread(target=task, daemon=True).start()

    def _on_pipeline_success(self):
        self.btn_pipeline.configure(state="normal")
        self._update_status("✅ Datos MLB actualizados correctamente")
        self._load_day_summary()

    def _on_pipeline_error(self, err_msg: str):
        self.btn_pipeline.configure(state="normal")
        self._update_status("❌ Error actualizando datos")
        messagebox.showerror("Error Pipeline", f"No se pudieron actualizar los datos:\n{err_msg}")

    def _run_predict(self):
        try:
            self._update_status("⏳ Generando predicciones y PDF...")
            predict_today.run(target_date=self.target_date, db_path="data/mlb.db")
            self._update_status("✅ Predicciones y PDF generados")
            self._load_day_summary()
            self._open_predictions_pdf()
        except Exception as err:
            self._update_status("❌ Error en predicciones")
            messagebox.showerror("Error Predicciones", str(err))

    def _run_report(self):
        try:
            self._update_status("⏳ Calificando resultados y generando PDF...")
            report_card.run(target_date=self.target_date, db_path="data/mlb.db")
            self._update_status("✅ Report Card generado")
            self._load_day_summary()
            self._open_report_pdf()
        except Exception as err:
            self._update_status("❌ Error en Report Card")
            messagebox.showerror("Error Report Card", str(err))

    def _open_pdf_file(self, pdf_path: str):
        p = Path(pdf_path)
        if not p.exists():
            messagebox.showwarning("PDF no encontrado", f"El archivo {p.name} aún no ha sido generado para {self.target_date}.")
            return
        try:
            os.startfile(str(p.resolve()))
        except Exception as err:
            messagebox.showerror("Error al abrir PDF", f"No se pudo abrir {pdf_path}:\n{err}")

    def _open_predictions_pdf(self):
        self._open_pdf_file(f"reports/predictions_{self.target_date}.pdf")

    def _open_report_pdf(self):
        self._open_pdf_file(f"reports/report_{self.target_date}.pdf")

    # --- UI Rendering for Day Summary ---
    def _load_day_summary(self):
        for widget in self.scroll_games.winfo_children():
            widget.destroy()

        conn = db.get_connection("data/mlb.db")
        db.init_db(conn)

        try:
            raw_df = report_card.fetch_predictions_with_results(conn, start_date=self.target_date, end_date=self.target_date, include_pending=True)
            df = report_card.compute_grades(raw_df)
        except Exception:
            df = pd.DataFrame()

        conn.close()

        if df.empty:
            lbl_empty = ctk.CTkLabel(
                self.scroll_games,
                text=f"No hay registros o predicciones previas para {self.target_date}.\nHaz clic en '📊 Predicciones (PDF)' para predecir todos los partidos.",
                font=ctk.CTkFont(size=14),
                text_color="#94a3b8",
            )
            lbl_empty.pack(pady=40)
            self.lbl_stats.configure(text=f"Fecha: {self.target_date} | Sin predicciones registradas aún")
            return

        final_df = df[df["is_final"] == True] if "is_final" in df.columns else df
        stats = report_card.summarize(final_df)

        if stats.get("n_games", 0) > 0:
            win_pct = stats['win_pct']
            stats_text = f"Fecha: <b>{self.target_date}</b> &nbsp;|&nbsp; Evaluados: {stats['n_games']} &nbsp;|&nbsp; Aciertos Ganador: <font color='#38bdf8'><b>{stats['win_hits']}/{stats['n_games']} ({win_pct:.1f}%)</b></font> &nbsp;|&nbsp; MAE O/U: {stats['mae_runs']:.2f}"
        else:
            stats_text = f"Fecha: {self.target_date} | {len(df)} partidos programados (Pendientes por calificar)"

        self.lbl_stats.configure(text=f"Fecha: {self.target_date} | Partidos encontrados: {len(df)}")

        # Renderizar cada tarjeta de partido
        for _, r in df.iterrows():
            card = ctk.CTkFrame(self.scroll_games, corner_radius=8, fg_color="#1e293b")
            card.pack(fill="x", padx=5, pady=4)

            # Team away vs home
            away_abbr = r.get("away_abbr") or "AWAY"
            home_abbr = r.get("home_abbr") or "HOME"

            title_text = f"{r.get('away_name', away_abbr)} @ {r.get('home_name', home_abbr)}"
            lbl_matchup = ctk.CTkLabel(card, text=title_text, font=ctk.CTkFont(size=14, weight="bold"), text_color="#f8fafc")
            lbl_matchup.pack(side="left", padx=15, pady=10)

            # Proyección Over/Under
            runs_pred = float(r["total_runs_pred"])
            ou_str = "OVER 8.5" if runs_pred >= 8.5 else "UNDER 8.5"
            lbl_ou = ctk.CTkLabel(card, text=f"O/U: {ou_str} ({runs_pred:.1f})", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38bdf8")
            lbl_ou.pack(side="left", padx=15)

            # Favorito
            fav = r.get("predicted_winner") or r.get("predicted_winner_abbr") or "?"
            proba = float(r.get("favorito_proba", 0.5))
            lbl_fav = ctk.CTkLabel(card, text=f"Favorito: {fav} ({proba:.0%})", font=ctk.CTkFont(size=12), text_color="#cbd5e1")
            lbl_fav.pack(side="left", padx=15)

            # Marcador / Estado
            is_final = bool(r.get("is_final", False))
            if is_final:
                score_str = f"Marcador: {int(r['away_score'])}-{int(r['home_score'])}"
                hit = bool(r.get("win_hit", False))
                result_text = "SÍ Acierto" if hit else "NO Acierto"
                color = "#22c55e" if hit else "#ef4444"
                lbl_res = ctk.CTkLabel(card, text=f"{score_str}  [{result_text}]", font=ctk.CTkFont(size=13, weight="bold"), text_color=color)
                lbl_res.pack(side="right", padx=15)
            else:
                lbl_res = ctk.CTkLabel(card, text="PENDIENTE / EN JUEGO", font=ctk.CTkFont(size=12, weight="bold"), text_color="#f59e0b")
                lbl_res.pack(side="right", padx=15)


def main():
    app = MLBPredictorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
