from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
                               QVBoxLayout, QWidget)

from ..analysis import AnalysisSettings, ShotResult, analyze_video
from ..camera import CameraSession
from ..colors import (BALL_COLOR_BY_KEY, BALL_COLORS, TRACER_COLORS, BallColor, bgr_to_hex, color_from_sample,
                      hex_to_bgr, sample_color)
from ..path import TracerPath, manual_path
from ..render import STYLES, TracerStyle, draw_candidates, draw_points, draw_tracer
from ..video import VideoSource, export_video
from . import workers
from .widgets import Segmented, SwatchButton, VideoView, section, stat_card

SPEEDS = [("0.1×", 0.1), ("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0)]


def fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ShotTracer – golf ball tracer")
        self.resize(1440, 880)
        self.settings = QSettings("ShotTracer", "ShotTracer")

        # state
        self.src: VideoSource | None = None
        self.idx = 0
        self.result: ShotResult | None = None
        self.path: TracerPath | None = None
        self.path_kind = ""                    # "auto" | "manual"
        self.seed: tuple[int, float, float] | None = None
        self.mode = ""                         # "", "pick", "seed", "manual"
        self.manual_pts: list[tuple[float, float]] = []
        self.manual_launch = 0
        self.ball_color: BallColor = BALL_COLORS[0]
        self.custom_color: BallColor | None = None
        self.style = TracerStyle()
        self._thread = None
        self._worker = None
        # camera
        self.session: CameraSession | None = None
        self._cam_seq = -1                     # last camera frame drawn
        self._cam_status_t = 0.0
        self.rec_path = ""
        self._closing = False

        self.play_timer = QTimer(self)
        self.play_timer.setTimerType(Qt.PreciseTimer)
        self.play_timer.timeout.connect(self._tick)
        self.cam_timer = QTimer(self)               # GUI only draws; capture/tracking are threads
        self.cam_timer.setInterval(15)
        self.cam_timer.timeout.connect(self._cam_draw)

        self._build_ui()
        self._load_settings()
        self._shortcuts()
        self._update_enabled()

    # ================================================================== UI build
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- left: top bar, video, transport
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)

        top = QWidget()
        top.setObjectName("TopBar")
        th = QHBoxLayout(top)
        th.setContentsMargins(18, 10, 18, 10)
        brand = QLabel("Shot")
        brand.setObjectName("Brand")
        brand2 = QLabel("Tracer")
        brand2.setObjectName("BrandAccent")
        th.addWidget(brand)
        th.addWidget(brand2)
        th.addSpacing(18)
        self.file_label = QLabel("")
        self.file_label.setObjectName("Muted")
        th.addWidget(self.file_label, 1)
        self.btn_open = QPushButton("📂  Open Video")
        self.btn_cam = QPushButton("📷  Live Camera")
        self.btn_cam.setCheckable(True)
        self.btn_rec = QPushButton("●  Record")
        self.btn_rec.setObjectName("Record")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setVisible(False)
        self.btn_export = QPushButton("⬇  Export")
        for b in (self.btn_open, self.btn_cam, self.btn_rec, self.btn_export):
            b.setCursor(Qt.PointingHandCursor)
            th.addWidget(b)
        lv.addWidget(top)

        self.view = VideoView()
        lv.addWidget(self.view, 1)

        tr = QWidget()
        tr.setObjectName("Transport")
        tl = QHBoxLayout(tr)
        tl.setContentsMargins(16, 10, 16, 10)
        tl.setSpacing(10)
        self.btn_prev = QPushButton("⏮")
        self.btn_prev.setObjectName("Icon")
        self.btn_play = QPushButton("▶")
        self.btn_play.setObjectName("PlayBtn")
        self.btn_next = QPushButton("⏭")
        self.btn_next.setObjectName("Icon")
        self.btn_prev.setToolTip("Previous frame  (←)")
        self.btn_next.setToolTip("Next frame  (→)")
        self.btn_play.setToolTip("Play / pause  (Space)")
        self.slider = QSlider(Qt.Horizontal)
        self.time_label = QLabel("00:00.00 / 00:00.00")
        self.time_label.setObjectName("Time")
        self.speed = QComboBox()
        for name, _ in SPEEDS:
            self.speed.addItem(name)
        self.speed.setCurrentIndex(3)
        self.speed.setToolTip("Playback speed (slow motion)")
        self.chk_loop = QCheckBox("Loop")
        for w in (self.btn_prev, self.btn_play, self.btn_next):
            w.setCursor(Qt.PointingHandCursor)
            tl.addWidget(w)
        tl.addWidget(self.slider, 1)
        tl.addWidget(self.time_label)
        tl.addWidget(self.speed)
        tl.addWidget(self.chk_loop)
        lv.addWidget(tr)
        outer.addWidget(left, 1)

        # ---- right: side panel
        side = QWidget()
        side.setObjectName("SidePanel")
        side.setFixedWidth(340)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(14, 14, 14, 14)
        iv.setSpacing(12)
        scroll.setWidget(inner)
        sv.addWidget(scroll)
        outer.addWidget(side)

        iv.addWidget(self._ball_section())
        iv.addWidget(self._track_section())
        iv.addWidget(self._tracer_section())
        iv.addWidget(self._manual_section())
        iv.addWidget(self._stats_section())
        iv.addStretch(1)

        self.statusBar().showMessage("Open a video of a golf swing to start.")

        # signals
        self.btn_open.clicked.connect(self._open_dialog)
        self.btn_cam.toggled.connect(self._toggle_camera)
        self.btn_rec.toggled.connect(self._toggle_record)
        self.btn_export.clicked.connect(self._export)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(1))
        self.slider.valueChanged.connect(self._on_slider)
        self.speed.currentIndexChanged.connect(self._apply_speed)
        self.view.clicked.connect(self._on_view_click)
        self.view.fileDropped.connect(self.open_video)

    def _ball_section(self):
        card, lay = section("Ball color")
        grid = QGridLayout()
        grid.setSpacing(4)
        self.ball_swatches: dict[str, SwatchButton] = {}
        from PySide6.QtWidgets import QButtonGroup
        self.ball_group = QButtonGroup(self)
        items = [(c.key, c.swatch, c.name) for c in BALL_COLORS] + [("custom", "#2a303a", "Custom (picked from video)")]
        for i, (key, hexc, name) in enumerate(items):
            b = SwatchButton(hexc, name)
            if key == "custom":
                b.badge = "+"
            self.ball_group.addButton(b)
            self.ball_swatches[key] = b
            grid.addWidget(b, i // 4, i % 4)
            b.clicked.connect(lambda _=False, k=key: self._select_ball(k))
        lay.addLayout(grid)
        self.ball_label = QLabel("")
        self.ball_label.setObjectName("Muted")
        lay.addWidget(self.ball_label)
        self.btn_pick = QPushButton("🎯  Pick ball color from video")
        self.btn_pick.setCheckable(True)
        self.btn_pick.setToolTip("Pause on a frame where the ball is visible and click it")
        self.btn_pick.toggled.connect(lambda on: self._set_mode("pick" if on else ""))
        lay.addWidget(self.btn_pick)
        return card

    def _track_section(self):
        card, lay = section("Auto track")
        self.btn_seed = QPushButton("⛳  Mark ball at address (optional)")
        self.btn_seed.setCheckable(True)
        self.btn_seed.setToolTip("Pause before the swing and click the ball on the ground.\n"
                                 "Helps when there are other moving objects of the same color.")
        self.btn_seed.toggled.connect(lambda on: self._set_mode("seed" if on else ""))
        lay.addWidget(self.btn_seed)
        row = QHBoxLayout()
        self.seed_label = QLabel("No address mark")
        self.seed_label.setObjectName("Muted")
        self.btn_clear_seed = QPushButton("Clear")
        self.btn_clear_seed.clicked.connect(self._clear_seed)
        row.addWidget(self.seed_label, 1)
        row.addWidget(self.btn_clear_seed)
        lay.addLayout(row)

        srow = QHBoxLayout()
        srow.addWidget(QLabel("Sensitivity"))
        self.sens_value = QLabel("50")
        self.sens_value.setObjectName("Muted")
        srow.addStretch(1)
        srow.addWidget(self.sens_value)
        lay.addLayout(srow)
        self.sens = QSlider(Qt.Horizontal)
        self.sens.setRange(0, 100)
        self.sens.setValue(50)
        self.sens.setToolTip("Higher finds faint / small / blurred balls but also more false detections")
        self.sens.valueChanged.connect(self._on_sens)
        lay.addWidget(self.sens)
        self.chk_stab = QCheckBox("Stabilize hand-held video")
        self.chk_stab.setChecked(True)
        lay.addWidget(self.chk_stab)

        self.btn_track = QPushButton("Auto Track Shot")
        self.btn_track.setObjectName("Primary")
        self.btn_track.setCursor(Qt.PointingHandCursor)
        self.btn_track.clicked.connect(self._analyze)
        lay.addWidget(self.btn_track)
        prow = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel_worker)
        prow.addWidget(self.progress, 1)
        prow.addWidget(self.btn_cancel)
        self.progress_row = QWidget()
        self.progress_row.setLayout(prow)
        prow.setContentsMargins(0, 0, 0, 0)
        self.progress_row.setVisible(False)
        lay.addWidget(self.progress_row)
        return card

    def _tracer_section(self):
        card, lay = section("Tracer")
        row = QHBoxLayout()
        row.setSpacing(2)
        from PySide6.QtWidgets import QButtonGroup
        self.tracer_group = QButtonGroup(self)
        self.tracer_swatches: dict[str, SwatchButton] = {}
        for name, hexc in TRACER_COLORS:
            b = SwatchButton(hexc, name, size=24)
            self.tracer_group.addButton(b)
            self.tracer_swatches[hexc] = b
            row.addWidget(b)
            b.clicked.connect(lambda _=False, h=hexc: self._set_tracer_color(h))
        row.addStretch(1)
        lay.addLayout(row)
        self.seg_style = Segmented(list(STYLES))
        self.seg_style.changed.connect(self._set_tracer_style)
        lay.addWidget(self.seg_style)

        trow = QHBoxLayout()
        trow.addWidget(QLabel("Thickness"))
        self.thick = QSlider(Qt.Horizontal)
        self.thick.setRange(2, 16)
        self.thick.setValue(6)
        self.thick.valueChanged.connect(self._set_thickness)
        trow.addWidget(self.thick, 1)
        lay.addLayout(trow)

        erow = QHBoxLayout()
        erow.addWidget(QLabel("Extend tail"))
        self.extend = QSlider(Qt.Horizontal)
        self.extend.setRange(0, 60)
        self.extend.setValue(0)
        self.extend.setToolTip("Extrapolate the tracer past the last frame the ball was seen")
        self.extend_value = QLabel("0 fr")
        self.extend_value.setObjectName("Muted")
        self.extend.valueChanged.connect(self._on_extend)
        erow.addWidget(self.extend, 1)
        erow.addWidget(self.extend_value)
        lay.addLayout(erow)

        self.chk_marker = QCheckBox("Ball marker")
        self.chk_marker.setChecked(True)
        self.chk_marker.toggled.connect(self._set_marker)
        self.chk_debug = QCheckBox("Show detections (debug)")
        self.chk_debug.toggled.connect(lambda _: self._render())
        lay.addWidget(self.chk_marker)
        lay.addWidget(self.chk_debug)
        return card

    def _manual_section(self):
        card, lay = section("Manual tracer")
        info = QLabel("If auto tracking misses, click the ball, the apex and the landing spot.")
        info.setObjectName("Muted")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.btn_manual = QPushButton("✏  Draw tracer manually")
        self.btn_manual.setCheckable(True)
        self.btn_manual.toggled.connect(lambda on: self._set_mode("manual" if on else ""))
        lay.addWidget(self.btn_manual)
        drow = QHBoxLayout()
        drow.addWidget(QLabel("Flight time"))
        self.duration = QSlider(Qt.Horizontal)
        self.duration.setRange(5, 80)       # tenths of a second
        self.duration.setValue(30)
        self.duration_value = QLabel("3.0 s")
        self.duration_value.setObjectName("Muted")
        self.duration.valueChanged.connect(self._on_duration)
        drow.addWidget(self.duration, 1)
        drow.addWidget(self.duration_value)
        lay.addLayout(drow)
        self.btn_clear_tracer = QPushButton("Clear tracer")
        self.btn_clear_tracer.clicked.connect(self._clear_tracer)
        lay.addWidget(self.btn_clear_tracer)
        return card

    def _stats_section(self):
        card, lay = section("Shot data")
        grid = QGridLayout()
        grid.setSpacing(8)
        c1, self.st_hang = stat_card("Hang time")
        c2, self.st_launch = stat_card("Impact at")
        c3, self.st_points = stat_card("Ball detections")
        c4, self.st_conf = stat_card("Confidence")
        for i, c in enumerate((c1, c2, c3, c4)):
            grid.addWidget(c, i // 2, i % 2)
        lay.addLayout(grid)
        return card

    def _shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self._step(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self._step(1))
        QShortcut(QKeySequence("Ctrl+O"), self, self._open_dialog)
        QShortcut(QKeySequence("Ctrl+E"), self, self._export)
        QShortcut(QKeySequence(Qt.Key_Escape), self, lambda: self._set_mode(""))

    # ================================================================ settings
    def _load_settings(self):
        s = self.settings
        custom = s.value("custom_ball", "")
        if custom:
            self.custom_color = color_from_sample(hex_to_bgr(custom))
            self.ball_swatches["custom"].set_color(custom)
            self.ball_swatches["custom"].badge = ""
        self._select_ball(s.value("ball", "white"), save=False)
        self._set_tracer_color(s.value("tracer_color", "#ff2d2d"), save=False)
        self.seg_style.set_value(s.value("tracer_style", "Glow"))
        self.style.style = self.seg_style.value()
        self.thick.setValue(int(s.value("thickness", 6)))
        self.sens.setValue(int(s.value("sensitivity", 50)))

    def _save(self, key, value):
        self.settings.setValue(key, value)

    # ============================================================ ball colour
    def _select_ball(self, key: str, save: bool = True):
        if key == "custom":
            if self.custom_color is None:
                self.ball_swatches[self.ball_color.key].setChecked(True)
                self.btn_pick.setChecked(True)
                return
            self.ball_color = self.custom_color
        else:
            self.ball_color = BALL_COLOR_BY_KEY.get(key, BALL_COLORS[0])
        self.ball_swatches[self.ball_color.key].setChecked(True)
        self.ball_label.setText(f"Tracking a <b>{self.ball_color.name.lower()}</b> ball")
        if save:
            self._save("ball", self.ball_color.key)
        if self.session is not None:
            self._restart_live()

    def _apply_picked_color(self, x, y):
        frame = self.src.read_full(self.idx) if self.src else None
        if frame is None:
            return
        bgr = sample_color(frame, x, y, radius=max(2, int(frame.shape[1] / 640)))
        self.custom_color = color_from_sample(bgr)
        hexc = bgr_to_hex(bgr)
        sw = self.ball_swatches["custom"]
        sw.badge = ""
        sw.set_color(hexc)
        self._save("custom_ball", hexc)
        self._select_ball("custom")
        self.statusBar().showMessage(f"Ball color picked: {hexc}. Now press Auto Track Shot.", 6000)

    # =============================================================== tracer
    def _set_tracer_color(self, hexc: str, save=True):
        self.style.color = hexc
        if hexc in self.tracer_swatches:
            self.tracer_swatches[hexc].setChecked(True)
        if save:
            self._save("tracer_color", hexc)
        self._render()

    def _set_tracer_style(self, name):
        self.style.style = name
        self._save("tracer_style", name)
        self._render()

    def _set_thickness(self, v):
        self.style.thickness = float(v)
        self._save("thickness", v)
        self._render()

    def _set_marker(self, on):
        self.style.ball_marker = on
        self._render()

    def _on_extend(self, v):
        self.extend_value.setText(f"{v} fr")
        if self.path_kind == "auto" and self.result is not None:
            self.path = self.result.build_path(v)
            self._render()

    def _on_sens(self, v):
        self.sens_value.setText(str(v))
        self._save("sensitivity", v)
        if self.session is not None:
            self._restart_live()

    def _on_duration(self, v):
        self.duration_value.setText(f"{v / 10:.1f} s")
        if self.src is not None and self.path_kind == "manual" and len(self.manual_pts) == 3:
            self._build_manual()

    # ================================================================== modes
    def _set_mode(self, mode: str):
        if mode and self.src is None:
            self.statusBar().showMessage("Open a video first.", 4000)
            mode = ""
        self.mode = mode
        for btn, m in ((self.btn_pick, "pick"), (self.btn_seed, "seed"), (self.btn_manual, "manual")):
            btn.blockSignals(True)
            btn.setChecked(mode == m)
            btn.blockSignals(False)
        if mode:
            self._pause()
        if mode == "manual":
            self.manual_pts = []
        self.view.set_picking(bool(mode))
        self._update_hint()
        self._render()

    def _update_hint(self):
        hints = {
            "pick": "Click the ball to sample its color",
            "seed": "Click the ball sitting at address (before the swing)",
        }
        if self.mode == "manual":
            steps = ["1/3  Click the ball at impact", "2/3  Click the highest point of the flight",
                     "3/3  Click where the ball lands / disappears"]
            self.view.set_hint(steps[min(len(self.manual_pts), 2)])
        else:
            self.view.set_hint(hints.get(self.mode, ""))

    def _on_view_click(self, x, y):
        if self.src is None:
            return
        if self.mode == "pick":
            self._apply_picked_color(x, y)
            self._set_mode("")
        elif self.mode == "seed":
            self.seed = (self.idx, x, y)
            self.seed_label.setText(f"Ball marked at {fmt_time(self.idx / self.src.fps)}")
            frame = self.src.read_full(self.idx)
            if frame is not None:
                bgr = sample_color(frame, x, y)
                hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
                if not self.ball_color.mask(hsv, 0.6)[0, 0]:
                    self.statusBar().showMessage(
                        f"Heads-up: the clicked spot doesn't look {self.ball_color.name.lower()}. "
                        "Pick the correct ball color if tracking fails.", 8000)
            self._set_mode("")
        elif self.mode == "manual":
            if not self.manual_pts:
                self.manual_launch = self.idx
            self.manual_pts.append((x, y))
            if len(self.manual_pts) == 3:
                self._build_manual()
                self._set_mode("")
                self._seek(max(0, self.manual_launch - int(self.src.fps * 0.5)))
                self._play()
            else:
                self._update_hint()
                self._render()

    def _build_manual(self):
        n = int(round(self.duration.value() / 10 * self.src.fps))
        self.path = manual_path(*self.manual_pts, launch_frame=self.manual_launch, n_frames=n)
        self.path_kind = "manual"
        self.st_hang.setText(f"{self.duration.value() / 10:.1f} s")
        self.st_launch.setText(fmt_time(self.manual_launch / self.src.fps))
        self.st_points.setText("manual")
        self.st_conf.setText("–")
        self._update_enabled()
        self._render()

    def _clear_seed(self):
        self.seed = None
        self.seed_label.setText("No address mark")
        self._render()

    def _clear_tracer(self):
        self.path = None
        self.path_kind = ""
        self.manual_pts = []
        for w in (self.st_hang, self.st_launch, self.st_points, self.st_conf):
            w.setText("–")
        self._update_enabled()
        self._render()

    # ================================================================== video
    def _busy(self) -> bool:
        if self._thread is not None:
            self.statusBar().showMessage("Please wait for the current task to finish (or cancel it).", 4000)
            return True
        return False

    def _open_dialog(self):
        if self._busy():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open swing video", self.settings.value("last_dir", ""),
                                              "Videos (*.mp4 *.mov *.avi *.mkv *.m4v *.wmv *.webm);;All files (*)")
        if path:
            self.open_video(path)

    def open_video(self, path: str):
        if self._busy():
            return
        if self.btn_cam.isChecked():
            self.btn_cam.setChecked(False)
        try:
            src = VideoSource(path)
        except IOError as e:
            QMessageBox.warning(self, "Open video", str(e))
            return
        self._close_video()
        self.src = src
        self._save("last_dir", str(Path(path).parent))
        self.file_label.setText(f"{Path(path).name}   ·   {src.width}×{src.height}   ·   {src.fps:.0f} fps")
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, src.count - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.idx = 0
        self._clear_seed()
        self._clear_tracer()
        self.result = None
        self._apply_speed()
        self._update_enabled()
        self._render()
        self.statusBar().showMessage("Choose the ball color, then press Auto Track Shot.")

    def _close_video(self):
        self._pause()
        if self.src is not None:
            self.src.release()
        self.src = None

    def _render(self):
        if self.src is None:
            if self.session is None:
                self.view.set_frame(None)
            return
        frame = self.src.read_display(self.idx)
        if frame is None:
            self._sync_count()
            return
        img = frame.copy()
        ds = self.src.display_scale
        if self.path is not None:
            pts = self.path.until(self.idx)
            if len(pts):
                draw_tracer(img, pts * ds, self.style, head_active=self.idx <= self.path.end)
        if self.chk_debug.isChecked() and self.result is not None:
            draw_candidates(img, self.result.candidates.get(self.idx), ds)
            if self.result.found:
                sel = self.result.frames <= self.idx
                draw_points(img, self.result.pts[sel] * ds, (0, 220, 255), 2)
        self._show(img, (self.src.width, self.src.height))

        markers = []
        if self.seed is not None and (self.path is None or self.idx < self.path.start):
            markers.append((self.seed[1], self.seed[2], "ball"))
        if self.mode == "manual":
            for i, (x, y) in enumerate(self.manual_pts):
                markers.append((x, y, ("start", "apex", "land")[i]))
        self.view.set_markers(markers)
        self._update_time()

    def _show(self, img_bgr, video_size):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.view.set_frame(qimg, video_size)

    def _update_time(self):
        if self.src is None:
            return
        if self.slider.maximum() != max(0, self.src.count - 1):
            self.slider.blockSignals(True)
            self.slider.setRange(0, max(0, self.src.count - 1))
            self.slider.blockSignals(False)
        self.time_label.setText(f"{fmt_time(self.idx / self.src.fps)} / {fmt_time(self.src.count / self.src.fps)}")
        self.slider.blockSignals(True)
        self.slider.setValue(self.idx)
        self.slider.blockSignals(False)

    def _sync_count(self):
        self.slider.setRange(0, max(0, self.src.count - 1))
        self.idx = min(self.idx, max(0, self.src.count - 1))

    # =============================================================== playback
    def _apply_speed(self):
        if self.src is None:
            return
        rate = SPEEDS[self.speed.currentIndex()][1]
        self.play_timer.setInterval(max(1, int(round(1000 / (self.src.fps * rate)))))

    def _toggle_play(self):
        if self.play_timer.isActive():
            self._pause()
        else:
            self._play()

    def _play(self):
        if self.src is None or self.mode:
            return
        if self.idx >= self.src.count - 1:
            self.idx = 0
        self._apply_speed()
        self.play_timer.start()
        self.btn_play.setText("❚❚")

    def _pause(self):
        self.play_timer.stop()
        self.btn_play.setText("▶")

    def _tick(self):
        if self.src is None:
            return self._pause()
        # read ahead instead of trusting the container's frame count, which can be wrong
        if self.src.read_display(self.idx + 1) is None:
            self._sync_count()
            if self.chk_loop.isChecked():
                start = self.path.start - int(self.src.fps * 0.5) if self.path is not None else 0
                self.idx = max(0, start)
            else:
                return self._pause()
        else:
            self.idx += 1
        self._render()

    def _step(self, d):
        if self.src is None:
            return
        self._pause()
        self._seek(self.idx + d)

    def _seek(self, i):
        self.idx = int(np.clip(i, 0, max(0, self.src.count - 1)))
        self._render()

    def _on_slider(self, v):
        if self.src is not None:
            self._seek(v)

    # =============================================================== analysis
    def _analyze(self):
        if self.src is None:
            self.statusBar().showMessage("Open a video first.", 4000)
            return
        if self._busy():
            return
        self._pause()
        self._set_mode("")
        settings = AnalysisSettings(self.ball_color, self.sens.value() / 100.0, self.chk_stab.isChecked())
        self._run_worker(workers.Worker(analyze_video, self.src.path, settings, seed=self.seed),
                         self._on_analyzed, f"Tracking {self.ball_color.name.lower()} ball…")

    def _on_analyzed(self, res: ShotResult):
        if self._closing:
            return
        self._end_worker()
        self.result = res
        if not res.found:
            if self.path_kind == "auto":
                self._clear_tracer()      # the old auto tracer and its stats are stale now
            else:
                self._update_enabled()    # keep a manual tracer the user drew
            self._render()
            QMessageBox.information(
                self, "No ball flight found",
                "Couldn't find the ball's flight.\n\n"
                "Try:\n"
                "  •  Make sure the right ball color is selected (or pick it from the video)\n"
                "  •  Mark the ball at address before the swing\n"
                "  •  Raise the sensitivity\n"
                "  •  Or draw the tracer manually")
            return
        self.path = res.build_path(self.extend.value())
        self.path_kind = "auto"
        self.st_hang.setText(f"{res.hang_time:.2f} s")
        self.st_launch.setText(fmt_time(res.launch_frame / res.fps))
        self.st_points.setText(str(len(res.frames)))
        self.st_conf.setText(res.confidence)
        self._update_enabled()
        self.statusBar().showMessage(
            f"Ball tracked for {res.hang_time:.2f} s ({res.confidence.lower()} confidence). "
            "Replay it, change the tracer look, or Export.", 10000)
        self._seek(max(0, res.launch_frame - int(res.fps * 0.6)))
        self._play()

    # ================================================================= export
    def _export(self):
        if self._busy():
            return
        if self.src is None or self.path is None:
            self.statusBar().showMessage("Track a shot (or draw one manually) before exporting.", 5000)
            return
        default = str(Path(self.src.path).with_name(Path(self.src.path).stem + "_tracer.mp4"))
        out, _ = QFileDialog.getSaveFileName(self, "Export video with tracer", default, "MP4 video (*.mp4)")
        if not out:
            return
        if Path(out).resolve() == Path(self.src.path).resolve():
            QMessageBox.warning(self, "Export", "Choose a different file name – this would overwrite the source video.")
            return
        self._pause()
        self._export_path = out
        self._run_worker(workers.Worker(export_video, self.src.path, self.path, replace(self.style), out),
                         self._on_exported, "Exporting…")

    def _on_exported(self, out):
        if self._closing:
            return
        self._end_worker()
        box = QMessageBox(self)
        box.setWindowTitle("Export finished")
        box.setText(f"Saved:\n{out}")
        open_btn = box.addButton("Open video", QMessageBox.AcceptRole)
        box.addButton("Close", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is open_btn and hasattr(os, "startfile"):
            os.startfile(out)

    # ================================================================ workers
    def _run_worker(self, worker, on_done, message):
        if self._thread is not None:
            return
        self._worker = worker
        worker.progress.connect(self.progress.setValue)
        worker.done.connect(on_done)
        worker.failed.connect(self._on_worker_failed)
        self.progress.setValue(0)
        self.progress_row.setVisible(True)
        self.statusBar().showMessage(message)
        self._thread = workers.start(worker, self)
        self._update_enabled()

    def _end_worker(self):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._worker = None
        self.progress_row.setVisible(False)
        self._update_enabled()

    def _on_worker_failed(self, msg):
        if self._closing:
            return
        self._end_worker()
        if msg != "Cancelled":
            QMessageBox.warning(self, "Error", msg)
        self.statusBar().showMessage(msg, 5000)

    def _cancel_worker(self):
        if self._worker is not None:
            self._worker.cancel()

    def _update_enabled(self):
        busy = self._thread is not None
        has_video = self.src is not None
        live = self.session is not None
        for w in (self.btn_track, self.btn_seed, self.btn_pick, self.btn_manual):
            w.setEnabled(has_video and not busy and not live)
        self.btn_export.setEnabled(has_video and self.path is not None and not busy and not live)
        self.btn_open.setEnabled(not busy)
        self.btn_cam.setEnabled(not busy)
        for w in (self.btn_play, self.btn_prev, self.btn_next, self.slider, self.speed):
            w.setEnabled(has_video and not live)

    # ================================================================= camera
    def _toggle_camera(self, on):
        if on:
            session = CameraSession(self.ball_color, self.sens.value() / 100.0)
            if not session.open():
                QMessageBox.warning(self, "Camera", "No camera found.")
                self.btn_cam.blockSignals(True)
                self.btn_cam.setChecked(False)
                self.btn_cam.blockSignals(False)
                return
            self._set_mode("")
            self._close_video()
            self.result = None
            self._clear_seed()
            self._clear_tracer()
            self.file_label.setText("Live camera")
            self.session = session
            self._cam_seq = -1
            self.btn_rec.setVisible(True)
            self.view.set_hint("LIVE  ·  swing away – the tracer appears automatically")
            self.cam_timer.start()
        else:
            if self.btn_rec.isChecked():
                self.btn_rec.setChecked(False)
            self.cam_timer.stop()
            if self.session is not None:
                self.session.close()
            self.session = None
            self.btn_rec.setVisible(False)
            self.view.set_hint("")
            if self.src is None:
                self.file_label.setText("")
                self.view.set_frame(None)
        self._update_enabled()

    def _restart_live(self):
        """Ball colour / sensitivity changed: the tracking thread rebuilds its tracker."""
        if self.session is not None:
            self.session.configure(self.ball_color, self.sens.value() / 100.0)

    def _cam_draw(self):
        """Runs on the GUI thread: only draws the newest frame and its tracer."""
        session = self.session
        if session is None:
            return
        if session.error:
            msg = session.error
            self.btn_cam.setChecked(False)
            QMessageBox.warning(self, "Camera", msg)
            return
        frame, seq, path = session.latest()
        if frame is None or seq == self._cam_seq:
            return
        self._cam_seq = seq
        h, w = frame.shape[:2]
        s = min(1.0, 1280 / max(w, h))
        img = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame.copy()
        if path is not None:
            draw_tracer(img, path.pts * s, self.style, head_active=seq - path.end < 10)
        if session.recording:
            cv2.circle(img, (24, 24), 8, (77, 72, 229), -1, cv2.LINE_AA)
            cv2.putText(img, f"REC {session.recording_seconds:4.1f}s", (40, 31), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
        self._show(img, (w, h))
        now = time.perf_counter()
        if now - self._cam_status_t > 0.5:
            self._cam_status_t = now
            self.statusBar().showMessage(f"Camera {session.camera_fps:.0f} fps  ·  "
                                         f"ball tracking {session.tracking_fps:.0f} fps")

    def _toggle_record(self, on):
        if on and self.session is not None:
            folder = Path.home() / "Videos" / "ShotTracer"
            folder.mkdir(parents=True, exist_ok=True)
            self.rec_path = str(folder / time.strftime("swing_%Y%m%d_%H%M%S.mp4"))
            err = self.session.start_recording(self.rec_path)
            if err:
                QMessageBox.warning(self, "Record", err)
                self.btn_rec.blockSignals(True)
                self.btn_rec.setChecked(False)
                self.btn_rec.blockSignals(False)
                return
            self.btn_rec.setText("■  Stop")
        elif self.session is not None and self.session.recording:
            self.session.stop_recording()
            self.btn_rec.setText("●  Record")
            if self._closing:
                return                     # just finalise the file; no new work while closing
            path = self.rec_path
            self.btn_cam.setChecked(False)
            self.open_video(path)
            if self.src is not None and self.src.path == path:
                self._analyze()

    # ================================================================== close
    def closeEvent(self, e):
        self._closing = True
        if self.btn_cam.isChecked():
            self.btn_cam.setChecked(False)      # finalises a recording; starts nothing new
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.wait()                 # every stage checks the cancel flag, so this is short
            self._thread = None
            self._worker = None
        self._close_video()
        super().closeEvent(e)
