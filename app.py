"""
app.py
Aplicación de Escritorio Interactiva (GUI Dashboard Premium) para MLB & LMB Predictor.
Ofrece vista de Tabla Estilo PDF con disposición perfecta en cuadrícula (grid),
selector de Modo Claro / Modo Oscuro, escudos PNG HD, barras de probabilidad,
filtros de riesgo para apuestas y pestañas de Liga (MLB / LMB).
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import date, timedelta
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import pandas as pd
from PIL import Image

# Asegurar import de los módulos locales en src/
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import db
import pdf_generator
import pipeline
import pipeline_lmb
import predict_today
import report_card

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

LOGOS_DIR = Path(__file__).parent / "assets" / "logos"
APP_STATE_PATH = Path(__file__).parent / ".app_state.json"


class MLBPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Baseball Predictor Dashboard - MLB & LMB")
        self.geometry("1240 x 840")
        self.minsize(1050, 720)

        self.target_date = str(date.today())
        self.selected_league = "MLB"
        self.search_query = ""

        # Cargar preferencias guardadas (.app_state.json)
        self._load_app_state()

        # Aplicar el tema configurado
        self._apply_theme_setting(self.theme_pref)

        # Caché de imágenes de escudos
        self._logo_images: dict[str, ctk.CTkImage] = {}

        self._create_layout()
        self.after(50, self._update_header_stats)
        self.after(50, self._load_day_summary)

    def _load_app_state(self):
        self.theme_pref = "🌙 Oscuro"
        self.view_mode = "TABLA"
        self.risk_filter = "TODOS"
        if APP_STATE_PATH.exists():
            try:
                with open(APP_STATE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.theme_pref = data.get("theme", "🌙 Oscuro")
                    self.view_mode = data.get("view_mode", "TABLA")
                    self.risk_filter = data.get("risk_filter", "TODOS")
            except Exception:
                pass

    def _save_app_state(self):
        try:
            data = {
                "theme": self.theme_pref,
                "view_mode": self.view_mode,
                "risk_filter": self.risk_filter,
            }
            with open(APP_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _apply_theme_setting(self, val: str):
        if "Claro" in val:
            ctk.set_appearance_mode("Light")
        else:
            ctk.set_appearance_mode("Dark")

    def _get_team_logo(self, abbr: str | None, team_id: int | None = None, size: tuple[int, int] = (22, 22)) -> ctk.CTkImage | None:
        """Carga y aplica caché a las imágenes PNG de los equipos."""
        key = f"{abbr or team_id or 'default'}_{size[0]}"
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
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=size)
                self._logo_images[key] = ctk_img
                return ctk_img
            except Exception:
                pass
        return None

    def _update_header_stats(self):
        """Muestra el número de aciertos acumulados reales en el header."""
        def task():
            try:
                conn = db.get_connection("data/mlb.db")
                db.init_db(conn)
                raw_df = report_card.fetch_predictions_with_results(conn, include_pending=False)
                conn.close()
                summary = report_card.summarize(raw_df)
                win_hits = summary.get("win_hits", 0)
                win_total = summary.get("n_games", 0)
                win_pct = summary.get("win_pct", 0.0)
                if win_total > 0:
                    txt = f"🏆 Aciertos MLB: {win_hits}/{win_total} · {win_pct:.1f}%"
                else:
                    txt = "🏆 Aciertos MLB: Sin historial evaluado"
            except Exception:
                txt = "🏆 Aciertos MLB: --"

            def safe_update():
                try:
                    if hasattr(self, "lbl_stats_summary") and self.lbl_stats_summary:
                        self.lbl_stats_summary.configure(text=txt)
                except Exception:
                    pass

            try:
                self.after(0, safe_update)
            except RuntimeError:
                pass

        threading.Thread(target=task, daemon=True).start()

    def _create_layout(self):
        # 1. Header Frame
        self.header_frame = ctk.CTkFrame(self, corner_radius=12, fg_color=("#e2e8f0", "#0f172a"))
        self.header_frame.pack(fill="x", padx=15, pady=(15, 6))

        title_subframe = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_subframe.pack(side="left", padx=20, pady=12)

        lbl_main = ctk.CTkLabel(
            title_subframe,
            text="⚾ Baseball Predictor Dashboard",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=("1e293b", "#f8fafc"),
        )
        lbl_main.pack(anchor="w")

        lbl_sub = ctk.CTkLabel(
            title_subframe,
            text="Sistema de Predicciones y Análisis de Apuestas de Béisbol",
            font=ctk.CTkFont(size=12),
            text_color=("#64748b", "#94a3b8"),
        )
        lbl_sub.pack(anchor="w")

        self.lbl_stats_summary = ctk.CTkLabel(
            title_subframe,
            text="🏆 Aciertos MLB: Cargando...",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#d97706",
        )
        self.lbl_stats_summary.pack(anchor="w", pady=(2, 0))

        # Control de Tema (Modo Claro / Modo Oscuro), Barra de Progreso y Estado
        right_header = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        right_header.pack(side="right", padx=15, pady=10)

        self.seg_theme = ctk.CTkSegmentedButton(
            right_header,
            values=["🌙 Oscuro", "☀️ Claro"],
            command=self._on_theme_change,
            width=130,
        )
        self.seg_theme.set(self.theme_pref)
        self.seg_theme.pack(anchor="e", pady=(0, 4))

        self.status_label = ctk.CTkLabel(
            right_header,
            text="🟢 Listo | Sistema preparado",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#0284c7",
        )
        self.status_label.pack(anchor="e")

        self.progress_bar = ctk.CTkProgressBar(right_header, width=180, height=6, mode="indeterminate", progress_color="#0284c7")
        self.progress_bar.pack(anchor="e", pady=(4, 0))
        self.progress_bar.pack_forget()

        # 2. Pestañas de Selección de Liga
        self.tab_view = ctk.CTkTabview(self, height=42, command=self._on_league_change)
        self.tab_view.pack(fill="x", padx=15, pady=2)
        self.tab_view.add("⚾ MLB (Grandes Ligas)")
        self.tab_view.add("🇲🇽 LMB (Liga Mexicana)")

        # 3. Control Panel (Fechas, Selector de Vista y Acciones)
        self.control_frame = ctk.CTkFrame(self, corner_radius=10)
        self.control_frame.pack(fill="x", padx=15, pady=4)

        # Subframe Fechas
        date_subframe = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        date_subframe.pack(side="left", padx=12, pady=8)

        lbl_date = ctk.CTkLabel(date_subframe, text="Fecha:", font=ctk.CTkFont(size=13, weight="bold"))
        lbl_date.pack(side="left", padx=(0, 4))

        self.btn_yesterday = ctk.CTkButton(date_subframe, text="Ayer", width=50, command=self._set_yesterday)
        self.btn_yesterday.pack(side="left", padx=2)

        self.btn_today = ctk.CTkButton(date_subframe, text="Hoy", width=50, command=self._set_today)
        self.btn_today.pack(side="left", padx=2)

        self.btn_tomorrow = ctk.CTkButton(date_subframe, text="Mañana", width=60, command=self._set_tomorrow)
        self.btn_tomorrow.pack(side="left", padx=2)

        self.entry_date = ctk.CTkEntry(date_subframe, width=100, font=ctk.CTkFont(size=13))
        self.entry_date.insert(0, self.target_date)
        self.entry_date.pack(side="left", padx=(4, 2))

        btn_go_date = ctk.CTkButton(date_subframe, text="📅 Ir", width=42, command=self._on_custom_date)
        btn_go_date.pack(side="left", padx=2)

        # Selector de Modo de Vista (Tabla PDF vs Tarjetas)
        view_subframe = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        view_subframe.pack(side="left", padx=10, pady=8)

        lbl_view = ctk.CTkLabel(view_subframe, text="Vista:", font=ctk.CTkFont(size=12, weight="bold"))
        lbl_view.pack(side="left", padx=(0, 4))

        self.segmented_view = ctk.CTkSegmentedButton(
            view_subframe,
            values=["📋 Tabla PDF", "📱 Tarjetas"],
            command=self._on_view_mode_change,
        )
        self.segmented_view.set("📋 Tabla PDF")
        self.segmented_view.pack(side="left")

        # Subframe Botones Acción
        action_subframe = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        action_subframe.pack(side="right", padx=12, pady=8)

        self.btn_pipeline = ctk.CTkButton(
            action_subframe,
            text="🔄 Actualizar Datos",
            fg_color="#0284c7",
            hover_color="#0369a1",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._run_pipeline_thread,
        )
        self.btn_pipeline.pack(side="left", padx=3)

        self.btn_predict = ctk.CTkButton(
            action_subframe,
            text="📊 Predicciones (PDF)",
            fg_color="#059669",
            hover_color="#047857",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._run_predict,
        )
        self.btn_predict.pack(side="left", padx=3)

        self.btn_report = ctk.CTkButton(
            action_subframe,
            text="🏆 Report Card (PDF)",
            fg_color="#d97706",
            hover_color="#b45309",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._run_report,
        )
        self.btn_report.pack(side="left", padx=3)

        # 4. Filter & Stats Bar
        self.filter_bar = ctk.CTkFrame(self, corner_radius=10, fg_color=("#cbd5e1", "#1e293b"))
        self.filter_bar.pack(fill="x", padx=15, pady=4)

        lbl_risk = ctk.CTkLabel(self.filter_bar, text="Filtrar Riesgo:", font=ctk.CTkFont(size=12, weight="bold"))
        lbl_risk.pack(side="left", padx=(12, 4), pady=6)

        self.segmented_risk = ctk.CTkSegmentedButton(
            self.filter_bar,
            values=["TODOS", "🟢 BAJO", "🔵 MEDIO", "🔴 ALTO"],
            command=self._on_risk_filter_change,
        )
        self.segmented_risk.set("TODOS")
        self.segmented_risk.pack(side="left", padx=4, pady=6)

        self.entry_search = ctk.CTkEntry(self.filter_bar, placeholder_text="🔍 Buscar equipo...", width=140)
        self.entry_search.pack(side="left", padx=(12, 4), pady=6)
        self.entry_search.bind("<KeyRelease>", self._on_search_change)

        pdf_subframe = ctk.CTkFrame(self.filter_bar, fg_color="transparent")
        pdf_subframe.pack(side="right", padx=12, pady=6)

        self.btn_open_pred_pdf = ctk.CTkButton(
            pdf_subframe,
            text="👁️ Abrir PDF Predicciones",
            width=145,
            fg_color=("#475569", "#334155"),
            hover_color=("#334155", "#475569"),
            command=self._open_predictions_pdf,
        )
        self.btn_open_pred_pdf.pack(side="left", padx=3)

        self.btn_open_rep_pdf = ctk.CTkButton(
            pdf_subframe,
            text="👁️ Abrir PDF Report Card",
            width=145,
            fg_color=("#475569", "#334155"),
            hover_color=("#334155", "#475569"),
            command=self._open_report_pdf,
        )
        self.btn_open_rep_pdf.pack(side="left", padx=3)

        # 5. Main Scrollable Games Area
        self.scroll_games = ctk.CTkScrollableFrame(self, label_text="Tabla de Análisis y Predicciones Diarias")
        self.scroll_games.pack(fill="both", expand=True, padx=15, pady=(4, 15))

    # --- Handlers & Theme ---
    def _on_theme_change(self, val: str):
        self.theme_pref = val
        self._apply_theme_setting(val)
        self._save_app_state()

    def _on_league_change(self):
        tab = self.tab_view.get()
        if "MLB" in tab:
            self.selected_league = "MLB"
        else:
            self.selected_league = "LMB"
        self._load_day_summary()

    def _on_view_mode_change(self, val: str):
        if "Tabla" in val:
            self.view_mode = "TABLA"
        else:
            self.view_mode = "TARJETAS"
        self._save_app_state()
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
        if "BAJO" in val:
            self.risk_filter = "BAJO"
        elif "MEDIO" in val:
            self.risk_filter = "MEDIO"
        elif "ALTO" in val:
            self.risk_filter = "ALTO"
        else:
            self.risk_filter = "TODOS"
        self._save_app_state()
        self._load_day_summary()

    def _on_search_change(self, event=None):
        self.search_query = self.entry_search.get().strip().lower()
        self._load_day_summary()

    def _update_status(self, msg: str):
        self.status_label.configure(text=msg)

    def _start_busy(self, status_msg: str):
        self.btn_pipeline.configure(state="disabled")
        self.btn_predict.configure(state="disabled")
        self.btn_report.configure(state="disabled")
        self.status_label.configure(text=status_msg)
        self.progress_bar.pack(anchor="e", pady=(4, 0))
        self.progress_bar.start()

    def _stop_busy(self, status_msg: str):
        self.btn_pipeline.configure(state="normal")
        self.btn_predict.configure(state="normal")
        self.btn_report.configure(state="normal")
        self.status_label.configure(text=status_msg)
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

    def _format_error_message(self, err_msg: str) -> tuple[str, str]:
        err_lower = err_msg.lower()
        if "model_win.joblib" in err_lower or "model_runs.joblib" in err_lower or ("file not found" in err_lower and "model" in err_lower):
            title = "Modelo no Encontrado"
            desc = "No se encontraron los archivos de modelos entrenados (ej. data/model_win.joblib).\n\nPor favor ejecuta el entrenamiento de los modelos antes de realizar predicciones."
        elif "no hay partidos" in err_lower or "empty schedule" in err_lower or "no games" in err_lower:
            title = "Sin Partidos Programados"
        elif "mlb.db" in err_lower or "operationalerror" in err_lower or "database is locked" in err_lower:
            title = "Error de Base de Datos"
            desc = "No se pudo acceder a la base de datos local (data/mlb.db).\nAsegúrate de que ningún otro proceso esté utilizando el archivo."
        else:
            title = "Error de Ejecución"
            desc = f"Ocurrió un detalle durante la ejecución:\n\n{err_msg}"
        return title, desc

    # --- Worker Threads ---
    def _run_pipeline_thread(self):
        self._start_busy(f"⏳ Actualizando datos {self.selected_league} y calculando predicciones...")

        def task():
            try:
                db_path = "data/lmb.db" if self.selected_league == "LMB" else "data/mlb.db"
                conn = db.get_connection(db_path)
                db.init_db(conn)
                conn.close()

                if self.selected_league == "LMB":
                    pipeline_lmb.run(db_path, self.target_date)
                else:
                    pipeline.run(db_path, self.target_date)

                # Generar predicciones automáticamente para la fecha y liga actualizada
                predict_today.run(target_date=self.target_date, db_path=db_path, league=self.selected_league)

                self.after(0, lambda: self._on_pipeline_success())
            except Exception as err:
                self.after(0, lambda: self._on_pipeline_error(str(err)))

        threading.Thread(target=task, daemon=True).start()

    def _on_pipeline_success(self):
        self._stop_busy("🟢 Datos y predicciones actualizados correctamente")
        self._update_header_stats()
        self._load_day_summary()

    def _on_pipeline_error(self, err_msg: str):
        self._stop_busy("🔴 Error actualizando datos")
        title, desc = self._format_error_message(err_msg)
        messagebox.showerror(title, desc)

    def _run_predict(self):
        self._start_busy(f"⏳ Generando predicciones {self.selected_league} y PDF...")

        def task():
            try:
                db_path = "data/lmb.db" if self.selected_league == "LMB" else "data/mlb.db"
                predict_today.run(target_date=self.target_date, db_path=db_path, league=self.selected_league)
                self.after(0, lambda: self._on_predict_success())
            except Exception as err:
                self.after(0, lambda: self._on_predict_error(str(err)))

        threading.Thread(target=task, daemon=True).start()

    def _on_predict_success(self):
        self._stop_busy("🟢 Predicciones y PDF generados")
        self._load_day_summary()
        self._open_predictions_pdf()

    def _on_predict_error(self, err_msg: str):
        self._stop_busy("🔴 Error en predicciones")
        title, desc = self._format_error_message(err_msg)
        messagebox.showerror(title, desc)

    def _run_report(self):
        self._start_busy("⏳ Calificando resultados y generando PDF...")

        def task():
            try:
                db_path = "data/lmb.db" if self.selected_league == "LMB" else "data/mlb.db"
                report_card.run(target_date=self.target_date, db_path=db_path, league=self.selected_league)
                self.after(0, lambda: self._on_report_success())
            except Exception as err:
                self.after(0, lambda: self._on_report_error(str(err)))

        threading.Thread(target=task, daemon=True).start()

    def _on_report_success(self):
        self._stop_busy("🟢 Report Card generado")
        self._update_header_stats()
        self._load_day_summary()
        self._open_report_pdf()

    def _on_report_error(self, err_msg: str):
        self._stop_busy("🔴 Error en Report Card")
        title, desc = self._format_error_message(err_msg)
        messagebox.showerror(title, desc)

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
        prefix = f"predictions_{self.selected_league.lower()}_"
        self._open_pdf_file(f"reports/{prefix}{self.target_date}.pdf")

    def _open_report_pdf(self):
        prefix = f"report_{self.selected_league.lower()}_" if self.selected_league != "MLB" else "report_"
        self._open_pdf_file(f"reports/{prefix}{self.target_date}.pdf")

    # --- UI Rendering ---
    def _load_day_summary(self):
        for widget in self.scroll_games.winfo_children():
            widget.destroy()

        league = self.selected_league
        db_path = "data/lmb.db" if league == "LMB" else "data/mlb.db"
        conn = db.get_connection(db_path)
        db.init_db(conn)

        try:
            raw_df = report_card.fetch_predictions_with_results(
                conn, start_date=self.target_date, end_date=self.target_date,
                include_pending=True, league=league
            )
            df = report_card.compute_grades(raw_df)

            # Si no hay predicciones guardadas pero sí hay partidos en games para la liga seleccionada, calcular en automático
            if df.empty:
                n_games = conn.execute(
                    "SELECT COUNT(*) FROM games WHERE game_date=? AND COALESCE(league, 'MLB')=?",
                    (self.target_date, league)
                ).fetchone()[0]

                if n_games > 0:
                    conn.close()
                    predict_today.run(target_date=self.target_date, db_path=db_path, league=league)
                    conn = db.get_connection(db_path)
                    raw_df = report_card.fetch_predictions_with_results(
                        conn, start_date=self.target_date, end_date=self.target_date,
                        include_pending=True, league=league
                    )
                    df = report_card.compute_grades(raw_df)
        except Exception:
            df = pd.DataFrame()

        conn.close()

        if df.empty:
            val_todos = "TODOS (0)"
            val_bajo = "🟢 BAJO (0)"
            val_medio = "🔵 MEDIO (0)"
            val_alto = "🔴 ALTO (0)"
            self.segmented_risk.configure(values=[val_todos, val_bajo, val_medio, val_alto])
            self.segmented_risk.set(val_todos)
            msg = (
                f"🇲🇽 Liga Mexicana de Béisbol (LMB)\n\nExtracción de datos y partidos en vivo activada para la LMB.\n"
                f"Las predicciones probabilísticas están deshabilitadas hasta contar con un modelo entrenado formalmente con historial de la LMB."
                if league == "LMB" else
                f"No hay registros o predicciones previas de {league} para {self.target_date}.\nHaz clic en '🔄 Actualizar Datos' o '📊 Predicciones (PDF)' para sincronizar los partidos."
            )
            lbl_empty = ctk.CTkLabel(
                self.scroll_games,
                text=msg,
                font=ctk.CTkFont(size=14),
                text_color=("#64748b", "#94a3b8"),
            )
            lbl_empty.pack(pady=40)
            return

        # Búsqueda por equipo
        if self.search_query:
            df = df[
                df["home_name"].str.lower().str.contains(self.search_query)
                | df["away_name"].str.lower().str.contains(self.search_query)
            ]

        # Calcular contadores por nivel de riesgo para la botonera
        cnt_todos = len(df)
        cnt_bajo = 0
        cnt_medio = 0
        cnt_alto = 0

        filtered_rows = []
        for _, r in df.iterrows():
            proba = float(r["home_win_proba"])
            fav_p = proba if proba >= 0.50 else 1.0 - proba

            if fav_p >= 0.62:
                risk_tag = "BAJO"
                risk_color = "#22c55e"
                cnt_bajo += 1
            elif fav_p >= 0.55:
                risk_tag = "MEDIO"
                risk_color = "#0284c7"
                cnt_medio += 1
            else:
                risk_tag = "ALTO"
                risk_color = "#ef4444"
                cnt_alto += 1

            if self.risk_filter != "TODOS":
                if self.risk_filter == "BAJO" and risk_tag != "BAJO":
                    continue
                if self.risk_filter == "MEDIO" and risk_tag != "MEDIO":
                    continue
                if self.risk_filter == "ALTO" and risk_tag != "ALTO":
                    continue

            filtered_rows.append((r, proba, fav_p, risk_tag, risk_color))

        val_todos = f"TODOS ({cnt_todos})"
        val_bajo = f"🟢 BAJO ({cnt_bajo})"
        val_medio = f"🔵 MEDIO ({cnt_medio})"
        val_alto = f"🔴 ALTO ({cnt_alto})"
        self.segmented_risk.configure(values=[val_todos, val_bajo, val_medio, val_alto])

        if self.risk_filter == "BAJO":
            self.segmented_risk.set(val_bajo)
        elif self.risk_filter == "MEDIO":
            self.segmented_risk.set(val_medio)
        elif self.risk_filter == "ALTO":
            self.segmented_risk.set(val_alto)
        else:
            self.segmented_risk.set(val_todos)

        if not filtered_rows:
            lbl_empty_filter = ctk.CTkLabel(
                self.scroll_games,
                text=f"No hay partidos que coincidan con el filtro de riesgo '{self.risk_filter}' o la búsqueda.",
                font=ctk.CTkFont(size=13),
                text_color=("#64748b", "#94a3b8"),
            )
            lbl_empty_filter.pack(pady=30)
            return

        if self.view_mode == "TABLA":
            self._render_pdf_table_view(filtered_rows)
        else:
            for item in filtered_rows:
                self._create_match_card(*item)

    def _render_pdf_table_view(self, rows_data: list):
        """Renderiza una tabla en cuadrícula perfecta (grid) idéntica al documento PDF."""
        table_container = ctk.CTkFrame(self.scroll_games, corner_radius=8, fg_color="transparent")
        table_container.pack(fill="x", expand=True, padx=2, pady=2)

        # Configurar 7 columnas proporcionales en el Grid
        col_specs = [
            ("Hora y Enfrentamiento", 0, 2),
            ("Predicción Favorito", 1, 2),
            ("Over / Under (8.5)", 2, 1),
            ("Primeras 5 (F5 4.5)", 3, 1),
            ("Apuesta Recomendada", 4, 2),
            ("Clima / Viento", 5, 1),
            ("Riesgo", 6, 1),
        ]

        for _, col_idx, weight in col_specs:
            table_container.grid_columnconfigure(col_idx, weight=weight, uniform="col_group")

        # 1. Encabezado de Tabla (Estilo PDF #0f172a)
        table_header = ctk.CTkFrame(table_container, corner_radius=6, fg_color="#0f172a")
        table_header.grid(row=0, column=0, columnspan=7, sticky="ew", padx=2, pady=(2, 4))
        for col_idx, weight in [(c[1], c[2]) for c in col_specs]:
            table_header.grid_columnconfigure(col_idx, weight=weight, uniform="col_group")

        for text, col_idx, _ in col_specs:
            lbl = ctk.CTkLabel(
                table_header,
                text=text,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#f8fafc",
            )
            lbl.grid(row=0, column=col_idx, sticky="ew", padx=6, pady=8)

        # 2. Filas de la Tabla en Grid
        for i, (r, proba, fav_p, risk_tag, risk_color) in enumerate(rows_data, start=1):
            bg_color = ("#f1f5f9", "#1e293b") if i % 2 == 1 else ("#e2e8f0", "#0f172a")
            row_frame = ctk.CTkFrame(table_container, corner_radius=6, fg_color=bg_color, border_width=1, border_color=("#cbd5e1", "#334155"))
            row_frame.grid(row=i, column=0, columnspan=7, sticky="ew", padx=2, pady=2)

            for col_idx, weight in [(c[1], c[2]) for c in col_specs]:
                row_frame.grid_columnconfigure(col_idx, weight=weight, uniform="col_group")

            # Columna 0: Hora y Enfrentamiento con logos
            cell0 = ctk.CTkFrame(row_frame, fg_color="transparent")
            cell0.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

            time_str = pdf_generator._format_game_time(r.get("game_date_utc")).replace("<font color='#64748b'><b>", "").replace("</b></font><br/>", "")
            stage_txt = "⚡ Lineup Confirmado" if r.get("prediction_stage") == "lineup_confirmed" else "📅 Matutina"
            stage_clr = "#22c55e" if r.get("prediction_stage") == "lineup_confirmed" else "#64748b"
            ctk.CTkLabel(cell0, text=f"{time_str or '⏰ Sin hora'}  •  {stage_txt}", font=ctk.CTkFont(size=11, weight="bold"), text_color=stage_clr).pack(anchor="w")

            away_sub = ctk.CTkFrame(cell0, fg_color="transparent")
            away_sub.pack(anchor="w")
            away_logo = self._get_team_logo(r.get("away_abbr"), r.get("away_team_id"), size=(18, 18))
            if away_logo:
                ctk.CTkLabel(away_sub, image=away_logo, text="").pack(side="left", padx=(0, 4))
            ctk.CTkLabel(away_sub, text=r.get("away_name", "Away"), font=ctk.CTkFont(size=12, weight="bold"), text_color=("1e293b", "#f8fafc")).pack(side="left")

            vs_sub = ctk.CTkFrame(cell0, fg_color="transparent")
            vs_sub.pack(anchor="w")
            ctk.CTkLabel(vs_sub, text="@", font=ctk.CTkFont(size=11, weight="bold"), text_color="#64748b").pack(side="left", padx=(0, 4))
            home_logo = self._get_team_logo(r.get("home_abbr"), r.get("home_team_id"), size=(18, 18))
            if home_logo:
                ctk.CTkLabel(vs_sub, image=home_logo, text="").pack(side="left", padx=(0, 4))
            ctk.CTkLabel(vs_sub, text=r.get("home_name", "Home"), font=ctk.CTkFont(size=12, weight="bold"), text_color=("1e293b", "#f8fafc")).pack(side="left")

            # Columna 1: Predicción Favorito
            cell1 = ctk.CTkFrame(row_frame, fg_color="transparent")
            cell1.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

            fav_name = r["home_name"] if proba >= 0.50 else r["away_name"]
            fav_abbr = r["home_abbr"] if proba >= 0.50 else r["away_abbr"]
            fav_id = r.get("home_team_id") if proba >= 0.50 else r.get("away_team_id")
            fav_logo = self._get_team_logo(fav_abbr, fav_id, size=(20, 20))

            fav_sub = ctk.CTkFrame(cell1, fg_color="transparent")
            fav_sub.pack(anchor="w")
            if fav_logo:
                ctk.CTkLabel(fav_sub, image=fav_logo, text="").pack(side="left", padx=(0, 4))
            ctk.CTkLabel(fav_sub, text=fav_name, font=ctk.CTkFont(size=12, weight="bold"), text_color=("1e293b", "#f8fafc")).pack(side="left")

            ctk.CTkLabel(cell1, text=f"{fav_p:.1%} victoria", font=ctk.CTkFont(size=11, weight="bold"), text_color="#0284c7").pack(anchor="w")

            # Columna 2: Over / Under (8.5)
            cell2 = ctk.CTkFrame(row_frame, fg_color="transparent")
            cell2.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)

            runs_pred = float(r["total_runs_pred"])
            ou_dir = "OVER 8.5" if runs_pred >= 8.5 else "UNDER 8.5"
            ou_color = "#22c55e" if "OVER" in ou_dir else "#ef4444"

            ctk.CTkLabel(cell2, text=ou_dir, font=ctk.CTkFont(size=12, weight="bold"), text_color=ou_color).pack(anchor="w")
            ctk.CTkLabel(cell2, text=f"{runs_pred:.1f} proy.", font=ctk.CTkFont(size=11), text_color=("#475569", "#cbd5e1")).pack(anchor="w")

            # Columna 3: Primeras 5 Entradas (F5)
            cell3 = ctk.CTkFrame(row_frame, fg_color="transparent")
            cell3.grid(row=0, column=3, sticky="nsew", padx=6, pady=6)

            f5_runs = float(r.get("f5_total_runs_pred", runs_pred * 0.55))
            f5_dir = "OVER 4.5" if f5_runs >= 4.5 else "UNDER 4.5"
            f5_color = "#22c55e" if "OVER" in f5_dir else "#ef4444"

            ctk.CTkLabel(cell3, text=f"{f5_dir} ({f5_runs:.1f})", font=ctk.CTkFont(size=12, weight="bold"), text_color=f5_color).pack(anchor="w")
            ctk.CTkLabel(cell3, text=f"F5: {fav_name}", font=ctk.CTkFont(size=11), text_color=("#475569", "#cbd5e1")).pack(anchor="w")

            # Columna 4: Apuesta Recomendada
            cell4 = ctk.CTkFrame(row_frame, fg_color="transparent")
            cell4.grid(row=0, column=4, sticky="nsew", padx=6, pady=6)

            is_final = bool(r.get("is_final", False))
            if is_final:
                hit = bool(r.get("win_hit", False))
                result_text = "✅ SÍ ACIERTO" if hit else "❌ NO ACIERTO"
                hit_color = "#22c55e" if hit else "#ef4444"
                ctk.CTkLabel(cell4, text=f"Marcador: {int(r['away_score'])}-{int(r['home_score'])}", font=ctk.CTkFont(size=11, weight="bold"), text_color=("1e293b", "#f8fafc")).pack(anchor="w")
                ctk.CTkLabel(cell4, text=result_text, font=ctk.CTkFont(size=11, weight="bold"), text_color=hit_color).pack(anchor="w")
            else:
                best_prop = pdf_generator._best_prop_recommendation(r).replace("<br/>", " ").replace("<font color='#16a34a'><b>", "").replace("<font color='#dc2626'><b>", "").replace("<font color='#475569'>", "").replace("</b></font>", "").replace("</font>", "").replace('<img src="assets/logos/', '').replace('.png" width="15" height="15" valign="middle"/> &nbsp;<b>', ' ').replace('</b>', '')
                ctk.CTkLabel(cell4, text=best_prop, font=ctk.CTkFont(size=11, weight="bold"), text_color="#d97706", wraplength=170).pack(anchor="w")

            # Columna 5: Clima / Viento
            cell5 = ctk.CTkFrame(row_frame, fg_color="transparent")
            cell5.grid(row=0, column=5, sticky="nsew", padx=6, pady=6)

            weather_txt = pdf_generator._weather_html(r.get("weather_temp"), r.get("weather_wind"), r.get("weather_condition")).replace("<b>", "").replace("</b>", "").replace("<br/>", " | ").replace("<font color='#94a3b8'>", "").replace("</font>", "")
            ctk.CTkLabel(cell5, text=weather_txt, font=ctk.CTkFont(size=11), text_color=("#475569", "#cbd5e1"), wraplength=100).pack(anchor="w")

            # Columna 6: Riesgo
            cell6 = ctk.CTkFrame(row_frame, fg_color="transparent")
            cell6.grid(row=0, column=6, sticky="nsew", padx=6, pady=6)

            ctk.CTkLabel(
                cell6,
                text=risk_tag,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=risk_color,
                fg_color="#0f172a",
                corner_radius=6,
                padx=8,
                pady=2,
            ).pack(anchor="center")

    def _create_match_card(self, r: dict, proba: float, fav_p: float, risk_tag: str, risk_color: str):
        card = ctk.CTkFrame(self.scroll_games, corner_radius=10, fg_color=("#f1f5f9", "#1e293b"), border_width=1, border_color=("#cbd5e1", "#334155"))
        card.pack(fill="x", padx=6, pady=6)

        header_sub = ctk.CTkFrame(card, fg_color="transparent")
        header_sub.pack(fill="x", padx=12, pady=(8, 4))

        time_str = pdf_generator._format_game_time(r.get("game_date_utc")).replace("<font color='#64748b'><b>", "").replace("</b></font><br/>", "")
        stage_txt = "⚡ Lineup Confirmado" if r.get("prediction_stage") == "lineup_confirmed" else "📅 Matutina"
        stage_clr = "#22c55e" if r.get("prediction_stage") == "lineup_confirmed" else "#64748b"
        lbl_time = ctk.CTkLabel(header_sub, text=f"{time_str or '⏰ Horario pendiente'}  •  {stage_txt}", font=ctk.CTkFont(size=11, weight="bold"), text_color=stage_clr)
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

        body_sub = ctk.CTkFrame(card, fg_color="transparent")
        body_sub.pack(fill="x", padx=12, pady=4)

        away_name = r.get("away_name") or "Visitante"
        away_logo = self._get_team_logo(r.get("away_abbr"), r.get("away_team_id"), size=(28, 28))

        home_name = r.get("home_name") or "Local"
        home_logo = self._get_team_logo(r.get("home_abbr"), r.get("home_team_id"), size=(28, 28))

        team_away_sub = ctk.CTkFrame(body_sub, fg_color="transparent")
        team_away_sub.pack(side="left", fill="x", expand=True)

        if away_logo:
            lbl_away_logo = ctk.CTkLabel(team_away_sub, image=away_logo, text="")
            lbl_away_logo.pack(side="left", padx=(0, 6))
        lbl_away_name = ctk.CTkLabel(team_away_sub, text=away_name, font=ctk.CTkFont(size=14, weight="bold"), text_color=("1e293b", "#f8fafc"))
        lbl_away_name.pack(side="left")

        lbl_vs = ctk.CTkLabel(body_sub, text="@", font=ctk.CTkFont(size=14, weight="bold"), text_color="#64748b")
        lbl_vs.pack(side="left", padx=10)

        team_home_sub = ctk.CTkFrame(body_sub, fg_color="transparent")
        team_home_sub.pack(side="left", fill="x", expand=True)

        if home_logo:
            lbl_home_logo = ctk.CTkLabel(team_home_sub, image=home_logo, text="")
            lbl_home_logo.pack(side="left", padx=(0, 6))
        lbl_home_name = ctk.CTkLabel(team_home_sub, text=home_name, font=ctk.CTkFont(size=14, weight="bold"), text_color=("1e293b", "#f8fafc"))
        lbl_home_name.pack(side="left")

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

        runs_pred = float(r["total_runs_pred"])
        ou_dir = "OVER 8.5" if runs_pred >= 8.5 else "UNDER 8.5"
        f5_runs = float(r.get("f5_total_runs_pred", runs_pred * 0.55))

        lines_sub = ctk.CTkFrame(card, fg_color="transparent")
        lines_sub.pack(fill="x", padx=12, pady=4)

        lbl_ou = ctk.CTkLabel(lines_sub, text=f"Juego Completo: {ou_dir} ({runs_pred:.1f})", font=ctk.CTkFont(size=12, weight="bold"), text_color="#22c55e" if "OVER" in ou_dir else "#ef4444")
        lbl_ou.pack(side="left", padx=(0, 15))

        lbl_f5 = ctk.CTkLabel(lines_sub, text=f"F5: {f5_runs:.1f} carreras", font=ctk.CTkFont(size=12), text_color=("#64748b", "#cbd5e1"))
        lbl_f5.pack(side="left")

        is_final = bool(r.get("is_final", False))
        if is_final:
            score_str = f"Marcador Real: {int(r['away_score'])}-{int(r['home_score'])}"
            hit = bool(r.get("win_hit", False))
            result_text = "✅ SÍ ACIERTO" if hit else "❌ NO ACIERTO"
            color = "#22c55e" if hit else "#ef4444"
            lbl_res = ctk.CTkLabel(lines_sub, text=f"{score_str}  [{result_text}]", font=ctk.CTkFont(size=12, weight="bold"), text_color=color)
            lbl_res.pack(side="right")
        else:
            best_prop = pdf_generator._best_prop_recommendation(r).replace("<br/>", " ").replace("<font color='#16a34a'><b>", "").replace("<font color='#dc2626'><b>", "").replace("<font color='#475569'>", "").replace("</b></font>", "").replace("</font>", "").replace('<img src="assets/logos/', '').replace('.png" width="15" height="15" valign="middle"/> &nbsp;<b>', ' ').replace('</b>', '')
            lbl_prop = ctk.CTkLabel(lines_sub, text=f"Apuesta: {best_prop}", font=ctk.CTkFont(size=12, weight="bold"), text_color="#d97706")
            lbl_prop.pack(side="right")


def main():
    app = MLBPredictorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
