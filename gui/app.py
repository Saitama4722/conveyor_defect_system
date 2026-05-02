"""Tkinter-приложение для системы дефектоскопии конвейера."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import zmq
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import collections

from detection.inference import DefectDetector
from detection.zmq_receiver import FrameReceiver


WINDOW_TITLE: str = "Система визуальной дефектоскопии — Конвейерная линия"
VIDEO_WIDTH: int = 1460
VIDEO_HEIGHT: int = 480
UPDATE_INTERVAL: int = 33        # мс между обновлениями (~30 FPS)
SPEED_MIN: float = 0.0
SPEED_MAX: float = 1.0
SPEED_STEP: float = 0.05
STATS_MAX_ROWS: int = 100        # максимум строк в истории детекций
BG_COLOR: str = "#1e1e2e"
PANEL_COLOR: str = "#2a2a3e"
ACCENT_COLOR: str = "#7c3aed"
TEXT_COLOR: str = "#e0e0e0"
DANGER_COLOR: str = "#ef4444"
SUCCESS_COLOR: str = "#22c55e"

PRODUCT_TYPES: list[str] = ["metal_nut_0", "screw_0", "bracket_0"]
DEFAULT_PRODUCT: str = "metal_nut_0"
DEFAULT_SPEED: float = 0.1
CAMERA_FRONT: str = "front"
CAMERA_SIDE: str = "side"
CAMERA_FRONT_LABEL: str = "Камера 1: фронтальная"
CAMERA_SIDE_LABEL: str = "Камера 2: боковая"

ZMQ_CMD_PORT: int = 5556
ZMQ_CMD_ADDR: str = f"tcp://localhost:{ZMQ_CMD_PORT}"

CHART_UPDATE_INTERVAL: int = 15  # обновлять графики каждые N кадров
MAX_TIMELINE_POINTS: int = 50    # максимум точек на графике динамики


class ConveyorApp:
    """Главное окно системы дефектоскопии."""

    def __init__(
        self,
        root: tk.Tk,
        receiver: FrameReceiver,
        detector: DefectDetector,
    ) -> None:
        self.root: tk.Tk = root
        self.receiver: FrameReceiver = receiver
        self.detector: DefectDetector = detector

        self._running: bool = False
        self._total_detections: int = 0
        self._detection_counts: dict[str, int] = {}
        self._speed: float = DEFAULT_SPEED
        self._current_product: str = DEFAULT_PRODUCT

        # Ссылка на PhotoImage основного canvas, чтобы его не удалил GC
        self._photo_main: ImageTk.PhotoImage | None = None

        # Будут заполнены в _build_ui
        self._canvas_main: tk.Canvas | None = None
        self._active_camera: tk.StringVar | None = None
        self.speed_scale: tk.Scale | None = None
        self.product_var: tk.StringVar | None = None
        self.total_label: tk.Label | None = None
        self.top_defect_label: tk.Label | None = None
        self.tree: ttk.Treeview | None = None

        # Состояние графиков
        self._chart_frame_count: int = 0
        self._timeline_data: collections.deque = collections.deque(
            maxlen=MAX_TIMELINE_POINTS
        )  # (timestamp, count) пар
        self._last_interval_count: int = 0
        self._fig: Figure | None = None
        self._ax_top: object | None = None
        self._ax_timeline: object | None = None
        self._chart_canvas: FigureCanvasTkAgg | None = None

        # Состояние полноэкранного режима (REQ-сокет создаётся на команду)
        self._fullscreen_camera: str | None = None
        self._photo_front_full: ImageTk.PhotoImage | None = None
        self._photo_side_full: ImageTk.PhotoImage | None = None
        self._fullscreen_window: tk.Toplevel | None = None
        self._fullscreen_canvas: tk.Canvas | None = None

        self._build_ui()
        self._init_cmd_socket()

    # ------------------------------------------------------------------ UI ---

    def _build_ui(self) -> None:
        """Собрать макет окна."""
        self.root.configure(bg=BG_COLOR)

        # --- Заголовок ---
        header = tk.Label(
            self.root,
            text="Система визуальной дефектоскопии",
            font=("Segoe UI", 16, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            pady=8,
        )
        header.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # --- Видеопанель: один большой canvas ---
        video_frame = tk.Frame(self.root, bg=BG_COLOR)
        video_frame.pack(side=tk.TOP, padx=5, pady=5)

        self._active_camera = tk.StringVar(value="front")

        # переключатель камер над canvas
        cam_frame = tk.Frame(video_frame, bg=BG_COLOR)
        cam_frame.pack(pady=2)
        tk.Radiobutton(
            cam_frame, text="Камера 1: фронтальная",
            variable=self._active_camera, value="front",
            bg=BG_COLOR, fg=TEXT_COLOR, selectcolor=PANEL_COLOR,
            font=("Arial", 10),
            command=self._on_camera_switch,
        ).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(
            cam_frame, text="Камера 2: боковая",
            variable=self._active_camera, value="side",
            bg=BG_COLOR, fg=TEXT_COLOR, selectcolor=PANEL_COLOR,
            font=("Arial", 10),
            command=self._on_camera_switch,
        ).pack(side=tk.LEFT, padx=10)

        self._canvas_main = tk.Canvas(
            video_frame,
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT,
            bg="#000000",
            highlightthickness=0,
        )
        self._canvas_main.pack()
        self._canvas_main.bind(
            "<Double-Button-1>",
            lambda e: self._open_fullscreen(self._active_camera.get()),
        )

        # кнопки пресетов под canvas
        btn_frame = tk.Frame(video_frame, bg=BG_COLOR)
        btn_frame.pack(pady=3)
        tk.Button(
            btn_frame, text="⬆ Сверху", bg=PANEL_COLOR, fg=TEXT_COLOR,
            font=("Arial", 10),
            command=lambda: self._send_camera_command(
                self._active_camera.get(), "top"),
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(
            btn_frame, text="↔ Сбоку", bg=PANEL_COLOR, fg=TEXT_COLOR,
            font=("Arial", 10),
            command=lambda: self._send_camera_command(
                self._active_camera.get(), "side"),
        ).pack(side=tk.LEFT, padx=8)

        # --- Панель управления ---
        controls = tk.Frame(self.root, bg=PANEL_COLOR, padx=8, pady=8)
        controls.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        start_btn = tk.Button(
            controls,
            text="Старт",
            bg=SUCCESS_COLOR,
            fg="white",
            activebackground=SUCCESS_COLOR,
            font=("Segoe UI", 10, "bold"),
            width=10,
            command=self.start,
        )
        start_btn.grid(row=0, column=0, padx=4)

        stop_btn = tk.Button(
            controls,
            text="Стоп",
            bg=DANGER_COLOR,
            fg="white",
            activebackground=DANGER_COLOR,
            font=("Segoe UI", 10, "bold"),
            width=10,
            command=self.stop,
        )
        stop_btn.grid(row=0, column=1, padx=4)

        tk.Label(
            controls,
            text="Деталь на ленте:",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=("Segoe UI", 10),
        ).grid(row=0, column=3, padx=(12, 4))

        self.product_var = tk.StringVar(value=DEFAULT_PRODUCT)
        product_menu = tk.OptionMenu(
            controls,
            self.product_var,
            *PRODUCT_TYPES,
            command=self._on_product_change,
        )
        product_menu.configure(
            bg=ACCENT_COLOR, fg="white", activebackground=ACCENT_COLOR, width=12
        )
        product_menu.grid(row=0, column=4, padx=4)

        # --- Статистика ---
        stats = tk.Frame(self.root, bg=PANEL_COLOR, padx=8, pady=8)
        stats.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Левая часть: счётчики и таблица истории
        stats_left = tk.Frame(stats, bg=PANEL_COLOR, width=480)
        stats_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        labels_row = tk.Frame(stats_left, bg=PANEL_COLOR)
        labels_row.pack(side=tk.TOP, fill=tk.X)

        self.total_label = tk.Label(
            labels_row,
            text="Всего обнаружено: 0",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=("Segoe UI", 11, "bold"),
        )
        self.total_label.pack(side=tk.LEFT, padx=4)

        self.top_defect_label = tk.Label(
            labels_row,
            text="Топ дефект: —",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=("Segoe UI", 11, "bold"),
        )
        self.top_defect_label.pack(side=tk.LEFT, padx=20)

        tree_holder = tk.Frame(stats_left, bg=PANEL_COLOR)
        tree_holder.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(8, 0))

        columns = ("time", "camera", "class", "confidence")
        self.tree = ttk.Treeview(
            tree_holder,
            columns=columns,
            show="headings",
            height=6,
        )
        self.tree.heading("time", text="Время")
        self.tree.heading("camera", text="Камера")
        self.tree.heading("class", text="Класс")
        self.tree.heading("confidence", text="Уверенность")
        self.tree.column("time", width=100, anchor="center")
        self.tree.column("camera", width=100, anchor="center")
        self.tree.column("class", width=200, anchor="w")
        self.tree.column("confidence", width=120, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_holder, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Правая часть: графики matplotlib
        stats_right = tk.Frame(stats, bg=PANEL_COLOR)
        stats_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._fig = Figure(figsize=(4, 3), facecolor=BG_COLOR)
        self._ax_top = self._fig.add_subplot(2, 1, 1)
        self._ax_timeline = self._fig.add_subplot(2, 1, 2)
        for ax in (self._ax_top, self._ax_timeline):
            ax.set_facecolor(PANEL_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            for spine in ax.spines.values():
                spine.set_color(TEXT_COLOR)
            ax.title.set_color(TEXT_COLOR)
            ax.xaxis.label.set_color(TEXT_COLOR)
            ax.yaxis.label.set_color(TEXT_COLOR)
        self._ax_top.set_title("Топ-3 дефектов")
        self._ax_timeline.set_title("Динамика обнаружений")
        self._fig.tight_layout()

        self._chart_canvas = FigureCanvasTkAgg(self._fig, master=stats_right)
        self._chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._chart_canvas.draw()

    # ------------------------------------------------------------ lifecycle ---

    def start(self) -> None:
        """Старт: запустить конвейер и обновление кадров."""
        self._running = True
        self._send_camera_command_raw("conveyor:start")
        cam = self._active_camera.get() if self._active_camera else "front"
        self.root.after(2000, lambda: self._send_camera_command(cam, "top"))
        self._schedule_update()

    def stop(self) -> None:
        """Стоп: остановить конвейер; видеопоток продолжает идти."""
        self._running = False
        self._send_camera_command_raw("conveyor:stop")

    def _schedule_update(self) -> None:
        """Поставить следующий тик через root.after."""
        if not self._running:
            return
        self._update_frames()
        self.root.after(UPDATE_INTERVAL, self._schedule_update)

    # ----------------------------------------------------------------- core ---

    def _update_frames(self) -> None:
        """Получить новые кадры, прогнать через детектор, отрисовать активную камеру."""
        if self._canvas_main is None or self._active_camera is None:
            return
        active = self._active_camera.get()
        frame_front = self.receiver.get_frame(CAMERA_FRONT)
        frame_side = self.receiver.get_frame(CAMERA_SIDE)

        # одноразовая диагностика
        if not hasattr(self, '_frame_debug_done'):
            self._frame_debug_done = True
            print(f"[FRAME] active camera: {self._active_camera.get()}")
            if frame_front is not None:
                dets = self.detector.predict(frame_front)
                print(f"[FRAME] detections on frame: {len(dets)}")

        if active == "front" and frame_front is not None:
            detections = self.detector.predict(frame_front)
            annotated = self.detector.draw_results(frame_front, detections)
            display = cv2.resize(annotated, (VIDEO_WIDTH, VIDEO_HEIGHT))
            self._show_frame(self._canvas_main, display)
            if (self._fullscreen_window is not None
                    and self._fullscreen_camera == "front"):
                self._show_fullscreen_frame(annotated, "front")
            if detections:
                self._update_stats(detections, CAMERA_FRONT)
        elif active == "side" and frame_side is not None:
            detections = self.detector.predict(frame_side)
            annotated = self.detector.draw_results(frame_side, detections)
            display = cv2.resize(annotated, (VIDEO_WIDTH, VIDEO_HEIGHT))
            self._show_frame(self._canvas_main, display)
            if (self._fullscreen_window is not None
                    and self._fullscreen_camera == "side"):
                self._show_fullscreen_frame(annotated, "side")
            if detections:
                self._update_stats(detections, CAMERA_SIDE)

    def _show_frame(
        self,
        canvas: tk.Canvas,
        frame: np.ndarray,
    ) -> None:
        """Вывести BGR-кадр на основной canvas (без масштабирования)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)
        # Сохраняем ссылку, иначе PhotoImage уничтожится сборщиком
        self._photo_main = photo

    def _update_stats(self, detections: list[dict], camera: str) -> None:
        """Обновить счётчики и таблицу истории."""
        if self.tree is None:
            return

        timestamp = time.strftime("%H:%M:%S")
        for det in detections:
            class_name = det["class_name"]
            confidence = float(det["confidence"])
            self._total_detections += 1
            self._detection_counts[class_name] = self._detection_counts.get(class_name, 0) + 1

            self.tree.insert(
                "",
                0,
                values=(timestamp, camera, class_name, f"{confidence:.2f}"),
            )
            children = self.tree.get_children()
            if len(children) > STATS_MAX_ROWS:
                self.tree.delete(children[-1])

        if self.total_label is not None:
            self.total_label.config(text=f"Всего обнаружено: {self._total_detections}")
        if self.top_defect_label is not None and self._detection_counts:
            top_class = max(self._detection_counts.items(), key=lambda kv: kv[1])
            self.top_defect_label.config(text=f"Топ дефект: {top_class[0]} ({top_class[1]})")

        # Обновление графиков
        self._last_interval_count += len(detections)
        self._chart_frame_count += 1
        if self._chart_frame_count % CHART_UPDATE_INTERVAL == 0:
            self._update_charts()

    def _update_charts(self) -> None:
        """Перерисовать оба графика по текущему состоянию счётчиков."""
        if self._fig is None or self._ax_top is None or self._ax_timeline is None:
            return

        # Топ-3 дефектов (горизонтальный bar)
        self._ax_top.clear()
        self._ax_top.set_facecolor(PANEL_COLOR)
        self._ax_top.tick_params(colors=TEXT_COLOR)
        for spine in self._ax_top.spines.values():
            spine.set_color(TEXT_COLOR)
        self._ax_top.set_title("Топ-3 дефектов", color=TEXT_COLOR)

        if not self._detection_counts:
            self._ax_top.text(
                0.5, 0.5, "Нет данных",
                transform=self._ax_top.transAxes,
                ha="center", va="center", color=TEXT_COLOR,
            )
        else:
            top3 = sorted(
                self._detection_counts.items(), key=lambda kv: kv[1], reverse=True
            )[:3]
            names = [item[0] for item in top3]
            counts = [item[1] for item in top3]
            self._ax_top.barh(names, counts, color=ACCENT_COLOR)
            self._ax_top.invert_yaxis()

        # Динамика обнаружений (линия)
        self._timeline_data.append(
            (len(self._timeline_data), self._last_interval_count)
        )
        self._last_interval_count = 0

        self._ax_timeline.clear()
        self._ax_timeline.set_facecolor(PANEL_COLOR)
        self._ax_timeline.tick_params(colors=TEXT_COLOR)
        for spine in self._ax_timeline.spines.values():
            spine.set_color(TEXT_COLOR)
        self._ax_timeline.set_title("Динамика обнаружений", color=TEXT_COLOR)

        if self._timeline_data:
            ys = [point[1] for point in self._timeline_data]
            xs = list(range(len(ys)))
            self._ax_timeline.plot(xs, ys, color=SUCCESS_COLOR, linewidth=1.5)

        self._fig.tight_layout()
        if self._chart_canvas is not None:
            self._fig.canvas.draw()

    # --------------------------------------------------------------- inputs ---

    def _init_cmd_socket(self) -> None:
        # REQ сокет создаётся заново для каждой команды
        pass

    def _send_camera_command_raw(self, msg: str) -> str:
        import zmq
        sock = None
        try:
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.SNDTIMEO, 300)
            sock.setsockopt(zmq.RCVTIMEO, 300)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(ZMQ_CMD_ADDR)
            sock.send_string(msg)
            reply = sock.recv_string()
            sock.close()
            return reply
        except Exception as e:
            print(f"[WARN] CMD: {e}")
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass
            return "ERROR"

    def _send_camera_command(self, camera: str, preset: str) -> None:
        reply = self._send_camera_command_raw(
            f"set_camera:{camera}:{preset}")
        print(f"[CMD] {camera} → {preset}: {reply}")

    def _on_camera_switch(self) -> None:
        # при переключении камеры — отправить пресет «сверху»
        if self._active_camera is None:
            return
        self._send_camera_command(self._active_camera.get(), "top")

    def _open_fullscreen(self, camera: str) -> None:
        # открыть отдельное окно с увеличенным видом камеры
        if self._fullscreen_window is not None:
            self._close_fullscreen()
        title = "фронтальная" if camera == "front" else "боковая"
        win = tk.Toplevel(self.root)
        win.title(f"Камера {title}")
        win.geometry("1280x720")
        win.configure(bg=BG_COLOR)
        canvas = tk.Canvas(
            win, width=1280, height=720, bg=BG_COLOR, highlightthickness=0
        )
        canvas.pack(fill=tk.BOTH, expand=True)
        self._fullscreen_window = win
        self._fullscreen_camera = camera
        self._fullscreen_canvas = canvas
        # кнопки позиционирования снизу
        btn_frame = tk.Frame(win, bg=BG_COLOR)
        btn_frame.pack(side=tk.BOTTOM, pady=5)
        tk.Button(
            btn_frame, text="⬆ Сверху", bg=PANEL_COLOR, fg=TEXT_COLOR,
            command=lambda: self._send_camera_command(camera, "top"),
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_frame, text="↔ Сбоку", bg=PANEL_COLOR, fg=TEXT_COLOR,
            command=lambda: self._send_camera_command(camera, "side"),
        ).pack(side=tk.LEFT, padx=5)
        win.protocol("WM_DELETE_WINDOW", self._close_fullscreen)

    def _close_fullscreen(self) -> None:
        if self._fullscreen_window is not None:
            try:
                self._fullscreen_window.destroy()
            except Exception:
                pass
        self._fullscreen_window = None
        self._fullscreen_camera = None
        self._fullscreen_canvas = None
        self._photo_front_full = None
        self._photo_side_full = None

    def _show_fullscreen_frame(self, frame: np.ndarray, camera: str) -> None:
        # отрисовка кадра 1280x720 в полноэкранном canvas
        if self._fullscreen_canvas is None:
            return
        big = cv2.resize(frame, (1280, 720))
        rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self._fullscreen_canvas.delete("all")
        self._fullscreen_canvas.create_image(0, 0, anchor="nw", image=photo)
        if camera == "front":
            self._photo_front_full = photo
        else:
            self._photo_side_full = photo

    def _on_speed_change(self, value: str) -> None:
        """Обработчик слайдера скорости."""
        # NOTE: реальная отправка скорости в Webots — через отдельный ZMQ
        # REQ/REP сокет (будет добавлено позже); пока только сохраняем локально.
        self._speed = float(value)

    def _on_product_change(self, product: str) -> None:
        """Обработчик OptionMenu — выбрать деталь на ленте."""
        self._current_product = product
        self._send_camera_command_raw(f"spawn_product:{product}")
        print(f"[CMD] Запущена деталь: {product}")


def main() -> None:
    """Точка входа: поднимаем приёмник, детектор и GUI."""
    receiver = FrameReceiver()
    receiver.start()

    detector = DefectDetector()

    root = tk.Tk()
    root.title(WINDOW_TITLE)
    root.configure(bg=BG_COLOR)
    root.geometry("1500x900")
    root.resizable(True, True)
    root.minsize(1460, 700)

    app = ConveyorApp(root, receiver, detector)
    app.start()

    try:
        root.mainloop()
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
