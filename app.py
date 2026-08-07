"""
app.py
Aplicación de Escritorio Interactiva (GUI Dashboard Premium) para MLB & LMB Predictor.
Permite gestionar la actualización de datos (pipeline), generar predicciones diarias,
calificar partidos y visualizar tarjetas de juego en Dark Mode con escudos PNG,
barras de probabilidad, filtros de riesgo para apuestas y pestañas de Liga (MLB / LMB).
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
import pdf_generator
import pipeline
import predict_today
import report_card

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

LOGOS_DIR = Path(__file__).parent / "assets" / "logos"


class MLBPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Baseball Predictor Dashboard - MLB & LMB")
        self.geometry("1180 x 820")
        self.minsize(1000, 700)

        self.target_date = str(date.today())
        self.selected_league = "MLB"
        self.risk_filter = "TODOS"
        self.search_query = ""

        # Caché de imágenes de escudos
        self._logo_images: dict[str, ctk.CTkImage] = {}

        self._create_layout()
        self._load_day_summary()

    def _get_team_logo(self, abbr: str | None, team_id: int | None = None) -> ctk.CTkImage | None:
        """Carga y aplica caché a las imágenes PNG de los equipos."""
        key = abbr or str(team_id) or "default"
        if key in self._logo_images:
            return self._logo_images[key]

        img_path = None
        if abbr and (LOGOS_DIR / f"{abbr}.png").exists():
            img_path = LOGOS_DIR / f"{abbr}.png"
        elif team_id and (LOGOS_DIR / f"{team_id}.png").exists():
            img_path = LOGOS_DIR / f"{team_id}.png"

        if img_path:
            try:
                pil_img = Image.open(img_path)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(32, 32))
                self._logo_images[key] = ctk_img
                return ctk_img
            except Exception:
                pass
        return None

    def _create_layout(self):
        # 1. Header Frame
        self.header_frame = ctk.CTkFrame(self, corner_radius=12, fg_color="#0f172a")
        self.header_frame.pack(fill="x", padx=15, pady=(15, 8))

        title_subframe = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_subframe.pack(side="left", padx=20, pady=12)

        lbl_main = ctk.CTkLabel(
            title_subframe,
            text="⚾ Baseball Predictor Dashboard",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#f8fafc",
        )
        lbl_main.pack(anchor="w")

        lbl_sub = ctk.CTkLabel(
            title_subframe,
            text="Sistema de Predicciones y Análisis de Apuestas de Béisbol",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8",
        )
        lbl_sub.pack(anchor="w")

        self.status_label = ctk.CTkLabel(
            self.header_frame,
            text="🟢 Listo | Sistema preparado",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#38bdf8",
        )
        self.status_label.pack(side="right", padx=20, pady=15)

        # 2. Pestañas de Selección de Liga
        self.tab_view = ctk.CTkTabview(self, height=45, command=self._on_league_change)
        self.tab_view.pack(fill="x", padx=15, pady=2)
        self.tab_view.add("⚾ MLB (Grandes Ligas)")
        self.tab_view.add("🇲🇽 LMB (Liga Mexicana)")

        # 3. Control Panel (Fechas y Acciones)
        self.control_frame = ctk.CTkFrame(self, corner_radius=10)
        self.control_frame.pack(fill="x", padx=15, pady=5)

        # Subframe Fechas
        date_subframe = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        date_subframe.pack(side="left", padx=15, pady=10)

        lbl_date = ctk.CTkLabel(date_subframe, text="Fecha:", font=ctk.CTkFont(size=13, weight="bold"))
        lbl_date.pack(side="left", padx=(0, 6))

        self.btn_yesterday = ctk.CTkButton(date_subframe, text="Ayer", width=55, command=self._set_yesterday)
        self.btn_yesterday.pack(side="left", padx=2)

        self.btn_today = ctk.CTkButton(date_subframe, text="Hoy", width=55, command=self._set_today)
        self.btn_today.pack(side="left", padx=2)

        self.btn_tomorrow = ctk.CTkButton(date_subframe, text="Mañana", width=65, command=self._set_tomorrow)
        self.btn_tomorrow.pack(side="left", padx=2)

        self.entry_date = ctk.CTkEntry(date_subframe, width=105, font=ctk.CTkFont(size=13))
        self.entry_date.insert(0, self.target_date)
        self.entry_date.pack(side="left", padx=(6, 2))

        btn_go_date = ctk.CTkButton(date_subframe, text="📅 Ir", width=45, command=self._on_custom_date)
        btn_go_date.pack(side="left", padx=2)

        # Subframe Botones Acción
        action_subframe = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        action_subframe.pack(side="right", padx=15, pady=10)

        self.btn_pipeline = ctk.CTkButton(
            action_subframe,
            text="🔄 Actualizar Datos",
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

        # 4. Filter & Stats Bar
        self.filter_bar = ctk.CTkFrame(self, corner_radius=10, fg_color="#1e293b")
        self.filter_bar.pack(fill="x", padx=15, pady=4)

        # Filtro de Riesgo (Segmented Button)
        lbl_risk = ctk.CTkLabel(self.filter_bar, text="Filtrar Riesgo:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#cbd5e1")
        lbl_risk.pack(side="left", padx=(15, 6), pady=8)

        self.segmented_risk = ctk.CTkSegmentedButton(
            self.filter_bar,
            values=["TODOS", "🟢 BAJO", "🔵 MEDIO", "🔴 ALTO"],
            command=self._on_risk_filter_change,
        )
        self.segmented_risk.set("TODOS")
        self.segmented_risk.pack(side="left", padx=4, pady=8)

        # Buscador de Equipos
        self.entry_search = ctk.CTkEntry(self.filter_bar, placeholder_text="🔍 Buscar equipo...", width=140)
        self.entry_search.pack(side="left", padx=(15, 4), pady=8)
        self.entry_search.bind("<KeyRelease>", self._on_search_change)

        # Botones de PDF a la derecha
        pdf_subframe = ctk.CTkFrame(self.filter_bar, fg_color="transparent")
        pdf_subframe.pack(side="right", padx=15, pady=6)

        self.btn_open_pred_pdf = ctk.CTkButton(
            pdf_subframe,
            text="👁️ Abrir PDF Predicciones",
            width=150,
            fg_color="#334155",
            hover_color="#475569",
            command=self._open_predictions_pdf,
        )
        self.btn_open_pred_pdf.pack(side="left", padx=3)

        self.btn_open_rep_pdf = ctk.CTkButton(
            pdf_subframe,
            text="👁️ Abrir PDF Report Card",
            width=150,
            fg_color="#334155",
            hover_color="#475569",
            command=self._open_report_pdf,
        )
        self.btn_open_rep_pdf.pack(side="left", padx=3)

        # 5. Main Scrollable Games List Area
        self.scroll_games = ctk.CTkScrollableFrame(self, label_text="Enfrentamientos y Tarjetas de Análisis")
        self.scroll_games.pack(fill="both", expand=True, padx=15, pady=(4, 15))

    # --- Handlers & Event Listeners ---
    def _on_league_change(self):
        tab = self.tab_view.get()
        if "MLB" in tab:
            self.selected_league = "MLB"
        else:
            self.selected_league = "LMB"
        self._load_day_summary()

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

    def _on_risk_filter_change(self, val: str):
        self.risk_filter = val
        self._load_day_summary()

    def _on_search_change(self, event=None):
        self.search_query = self.entry_search.get().strip().lower()
        self._load_day_summary()

    def _update_status(self, msg: str):
        self.status_label.configure(text=msg)

    # --- Worker Threads ---
    def _run_pipeline_thread(self):
        if self.selected_league == "LMB":
            messagebox.showinfo("LMB Módulo", "Próximamente: Integración de extracción directa de la Liga Mexicana de Béisbol (LMB). Por ahora se encuentra activo el módulo MLB.")
            return

        self.btn_pipeline.configure(state="disabled")
        self._update_status("⏳ Actualizando datos MLB desde la API...")

        def task():
            try:
                conn = db.get_connection("data/mlb.db")
                db.init_db(conn)
                pipeline.run("data/mlb.db", self.target_date)
                conn.close()
                self.after(0, lambda: self._on_pipeline_success())
            except Exception as err:
                self.after(0, lambda: self._on_pipeline_error(str(err)))

        threading.Thread(target=task, daemon=True).start()

    def _on_pipeline_success(self):
        self.btn_pipeline.configure(state="normal")
        self._update_status("🟢 Datos actualizados correctamente")
        self._load_day_summary()

    def _on_pipeline_error(self, err_msg: str):
        self.btn_pipeline.configure(state="normal")
        self._update_status("🔴 Error actualizando datos")
        messagebox.showerror("Error Pipeline", f"No se pudieron actualizar los datos:\n{err_msg}")

    def _run_predict(self):
        if self.selected_league == "LMB":
            messagebox.showinfo("LMB Módulo", "Las predicciones de la LMB estarán disponibles al integrar las fuentes de datos oficiales.")
            return
        try:
            self._update_status("⏳ Generando predicciones y PDF...")
            predict_today.run(target_date=self.target_date, db_path="data/mlb.db")
            self._update_status("🟢 Predicciones y PDF generados")
            self._load_day_summary()
            self._open_predictions_pdf()
        except Exception as err:
            self._update_status("🔴 Error en predicciones")
            messagebox.showerror("Error Predicciones", str(err))

    def _run_report(self):
        if self.selected_league == "LMB":
            messagebox.showinfo("LMB Módulo", "Report Card disponible para MLB.")
            return
        try:
            self._update_status("⏳ Calificando resultados y generando PDF...")
            report_card.run(target_date=self.target_date, db_path="data/mlb.db")
            self._update_status("🟢 Report Card generado")
            self._load_day_summary()
            self._open_report_pdf()
        except Exception as err:
            self._update_status("🔴 Error en Report Card")
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

    # --- UI Rendering for Match Cards ---
    def _load_day_summary(self):
        for widget in self.scroll_games.winfo_children():
            widget.destroy()

        if self.selected_league == "LMB":
            lbl_lmb = ctk.CTkLabel(
                self.scroll_games,
                text="🇲🇽 Liga Mexicana de Béisbol (LMB)\n\nEstructura gráfica preparada para la integración de la LMB.\nPróximamente se sincronizarán los juegos de los Diablos Rojos, Sulseros, Toros y más.",
                font=ctk.CTkFont(size=14),
                text_color="#94a3b8",
            )
            lbl_lmb.pack(pady=60)
            return

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
                text=f"No hay registros o predicciones previas para {self.target_date}.\nHaz clic en '📊 Predicciones (PDF)' para calcular todos los partidos.",
                font=ctk.CTkFont(size=14),
                text_color="#94a3b8",
            )
            lbl_empty.pack(pady=40)
            return

        # Aplicar búsqueda por nombre de equipo
        if self.search_query:
            df = df[
                df["home_name"].str.lower().str.contains(self.search_query)
                | df["away_name"].str.lower().str.contains(self.search_query)
            ]

        # Renderizar cada tarjeta de partido
        for _, r in df.iterrows():
            proba = float(r["home_win_proba"])
            fav_p = proba if proba >= 0.50 else 1.0 - proba

            # Evaluación del nivel de riesgo
            if fav_p >= 0.62:
                risk_tag = "BAJO"
                risk_color = "#22c55e"
            elif fav_p >= 0.55:
                risk_tag = "MEDIO"
                risk_color = "#38bdf8"
            else:
                risk_tag = "ALTO"
                risk_color = "#ef4444"

            # Aplicar filtro de riesgo
            if self.risk_filter != "TODOS":
                if "BAJO" in self.risk_filter and risk_tag != "BAJO":
                    continue
                if "MEDIO" in self.risk_filter and risk_tag != "MEDIO":
                    continue
                if "ALTO" in self.risk_filter and risk_tag != "ALTO":
                    continue

            self._create_match_card(r, proba, fav_p, risk_tag, risk_color)

    def _create_match_card(self, r: dict, proba: float, fav_p: float, risk_tag: str, risk_color: str):
        card = ctk.CTkFrame(self.scroll_games, corner_radius=10, fg_color="#1e293b", border_width=1, border_color="#334155")
        card.pack(fill="x", padx=6, pady=6)

        # Header de la tarjeta (Hora + Badge de Riesgo)
        header_sub = ctk.CTkFrame(card, fg_color="transparent")
        header_sub.pack(fill="x", padx=12, pady=(8, 4))

        time_str = pdf_generator._format_game_time(r.get("game_date_utc")).replace("<font color='#64748b'><b>", "").replace("</b></font><br/>", "")
        lbl_time = ctk.CTkLabel(header_sub, text=time_str or "⏰ Horario pendiente", font=ctk.CTkFont(size=12, weight="bold"), text_color="#94a3b8")
        lbl_time.pack(side="left")

        lbl_risk_badge = ctk.CTkLabel(
            header_sub,
            text=f"Riesgo {risk_tag}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=risk_color,
            fg_color="#0f172a",
            corner_radius=6,
            padx=8,
            pady=2,
        )
        lbl_risk_badge.pack(side="right")

        # Body de la tarjeta (Enfrentamiento con logos)
        body_sub = ctk.CTkFrame(card, fg_color="transparent")
        body_sub.pack(fill="x", padx=12, pady=4)

        away_name = r.get("away_name") or "Visitante"
        away_abbr = r.get("away_abbr")
        away_logo = self._get_team_logo(away_abbr, r.get("away_team_id"))

        home_name = r.get("home_name") or "Local"
        home_abbr = r.get("home_abbr")
        home_logo = self._get_team_logo(home_abbr, r.get("home_team_id"))

        # Visitante
        team_away_sub = ctk.CTkFrame(body_sub, fg_color="transparent")
        team_away_sub.pack(side="left", fill="x", expand=True)

        if away_logo:
            lbl_away_logo = ctk.CTkLabel(team_away_sub, image=away_logo, text="")
            lbl_away_logo.pack(side="left", padx=(0, 6))
        lbl_away_name = ctk.CTkLabel(team_away_sub, text=away_name, font=ctk.CTkFont(size=14, weight="bold"), text_color="#f8fafc")
        lbl_away_name.pack(side="left")

        lbl_vs = ctk.CTkLabel(body_sub, text="@", font=ctk.CTkFont(size=14, weight="bold"), text_color="#64748b")
        lbl_vs.pack(side="left", padx=10)

        # Local
        team_home_sub = ctk.CTkFrame(body_sub, fg_color="transparent")
        team_home_sub.pack(side="left", fill="x", expand=True)

        if home_logo:
            lbl_home_logo = ctk.CTkLabel(team_home_sub, image=home_logo, text="")
            lbl_home_logo.pack(side="left", padx=(0, 6))
        lbl_home_name = ctk.CTkLabel(team_home_sub, text=home_name, font=ctk.CTkFont(size=14, weight="bold"), text_color="#f8fafc")
        lbl_home_name.pack(side="left")

        # Barra de Probabilidad del Favorito
        fav_name = home_name if proba >= 0.50 else away_name
        fav_sub = ctk.CTkFrame(card, fg_color="transparent")
        fav_sub.pack(fill="x", padx=12, pady=4)

        lbl_fav_info = ctk.CTkLabel(
            fav_sub,
            text=f"Favorito: {fav_name} ({fav_p:.1%} proba)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#0284c7",
        )
        lbl_fav_info.pack(side="left")

        pbar = ctk.CTkProgressBar(fav_sub, width=180, height=8, progress_color=risk_color)
        pbar.set(fav_p)
        pbar.pack(side="right", padx=5)

        # Líneas y Proyecciones
        runs_pred = float(r["total_runs_pred"])
        ou_dir = "OVER 8.5" if runs_pred >= 8.5 else "UNDER 8.5"
        f5_runs = float(r.get("f5_total_runs_pred", runs_pred * 0.55))

        lines_sub = ctk.CTkFrame(card, fg_color="transparent")
        lines_sub.pack(fill="x", padx=12, pady=4)

        lbl_ou = ctk.CTkLabel(lines_sub, text=f"Juego Completo: {ou_dir} ({runs_pred:.1f})", font=ctk.CTkFont(size=12, weight="bold"), text_color="#22c55e" if "OVER" in ou_dir else "#ef4444")
        lbl_ou.pack(side="left", padx=(0, 15))

        lbl_f5 = ctk.CTkLabel(lines_sub, text=f"F5: {f5_runs:.1f} carreras", font=ctk.CTkFont(size=12), text_color="#cbd5e1")
        lbl_f5.pack(side="left")

        # Marcador Real o Apuesta Recomendada
        is_final = bool(r.get("is_final", False))
        if is_final:
            score_str = f"Marcador Real: {int(r['away_score'])}-{int(r['home_score'])}"
            hit = bool(r.get("win_hit", False))
            result_text = "✅ SÍ ACIERTO" if hit else "❌ NO ACIERTO"
            color = "#22c55e" if hit else "#ef4444"
            lbl_res = ctk.CTkLabel(lines_sub, text=f"{score_str}  [{result_text}]", font=ctk.CTkFont(size=12, weight="bold"), text_color=color)
            lbl_res.pack(side="right")
        else:
            best_prop = pdf_generator._best_prop_recommendation(r).replace("<br/>", " | ").replace("<font color='#16a34a'><b>", "").replace("<font color='#dc2626'><b>", "").replace("<font color='#475569'>", "").replace("</b></font>", "").replace("</font>", "")
            lbl_prop = ctk.CTkLabel(lines_sub, text=f"Apuesta: {best_prop}", font=ctk.CTkFont(size=12, weight="bold"), text_color="#f59e0b")
            lbl_prop.pack(side="right")


def main():
    app = MLBPredictorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
