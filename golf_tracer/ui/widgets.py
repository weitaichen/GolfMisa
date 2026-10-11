from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from .style import ACCENT

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm"}


class VideoView(QWidget):
    """Displays frames (aspect-fit), maps clicks to video pixels, shows a zoom loupe
    while picking so a tiny ball can be clicked precisely."""

    clicked = Signal(float, float)
    fileDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 300)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img: QImage | None = None
        self._video_size = (1, 1)
        self._hint = ""
        self._picking = False
        self._mouse: QPointF | None = None
        self._markers: list[tuple[float, float, str]] = []
        self._target = QRectF()

    # ------------------------------------------------------------------ state
    def set_frame(self, img: QImage | None, video_size=(1, 1)):
        self._img = img
        self._video_size = video_size
        self.update()

    def set_hint(self, text: str):
        self._hint = text
        self.update()

    def set_picking(self, on: bool):
        self._picking = on
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        self.update()

    def set_markers(self, markers):
        self._markers = list(markers)
        self.update()

    # --------------------------------------------------------------- geometry
    def _fit_rect(self) -> QRectF:
        if self._img is None:
            return QRectF()
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _to_video(self, pos: QPointF):
        r = self._target
        if r.isEmpty() or not r.contains(pos):
            return None
        return ((pos.x() - r.x()) / r.width() * self._video_size[0],
                (pos.y() - r.y()) / r.height() * self._video_size[1])

    def _to_widget(self, x, y) -> QPointF:
        r = self._target
        return QPointF(r.x() + x / self._video_size[0] * r.width(), r.y() + y / self._video_size[1] * r.height())

    # ---------------------------------------------------------------- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#07080a"))
        self._target = self._fit_rect()
        if self._img is None:
            self._paint_empty(p)
            return
        p.drawImage(self._target, self._img)

        for x, y, label in self._markers:
            c = self._to_widget(x, y)
            p.setPen(QPen(QColor(ACCENT), 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, 9, 9)
            p.drawLine(c + QPointF(-14, 0), c + QPointF(-5, 0))
            p.drawLine(c + QPointF(5, 0), c + QPointF(14, 0))
            if label:
                p.setPen(QColor("white"))
                p.setFont(QFont("Segoe UI", 9, QFont.Bold))
                p.drawText(c + QPointF(13, -10), label)

        if self._hint:
            p.setFont(QFont("Segoe UI", 10, QFont.DemiBold))
            fm = p.fontMetrics()
            w = fm.horizontalAdvance(self._hint) + 32
            box = QRectF((self.width() - w) / 2, 14, w, 32)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(10, 12, 16, 210))
            p.drawRoundedRect(box, 16, 16)
            p.setPen(QColor(ACCENT))
            p.drawText(box, Qt.AlignCenter, self._hint)

        if self._picking and self._mouse is not None and self._target.contains(self._mouse):
            self._paint_loupe(p)

    def _paint_empty(self, p: QPainter):
        p.setPen(QPen(QColor("#2a303a"), 2, Qt.DashLine))
        r = self.rect().adjusted(40, 40, -40, -40)
        p.drawRoundedRect(r, 18, 18)
        p.setPen(QColor("#e8eaed"))
        p.setFont(QFont("Segoe UI", 16, QFont.Bold))
        p.drawText(r.adjusted(0, -40, 0, -40), Qt.AlignCenter, "⛳  Drop a swing video here")
        p.setPen(QColor("#7d8696"))
        p.setFont(QFont("Segoe UI", 11))
        p.drawText(r.adjusted(0, 20, 0, 20), Qt.AlignCenter,
                   "or use  Open Video  /  Live Camera  above")

    def _paint_loupe(self, p: QPainter):
        img_x = (self._mouse.x() - self._target.x()) / self._target.width() * self._img.width()
        img_y = (self._mouse.y() - self._target.y()) / self._target.height() * self._img.height()
        src_r = 18
        src = QRectF(img_x - src_r, img_y - src_r, 2 * src_r, 2 * src_r)
        size = 132
        cx = self._mouse.x() + 90 if self._mouse.x() + 90 + size / 2 < self.width() else self._mouse.x() - 90
        cy = min(max(self._mouse.y() - 90, size / 2 + 4), self.height() - size / 2 - 4)
        dst = QRectF(cx - size / 2, cy - size / 2, size, size)
        clip = QPainterPath()
        clip.addEllipse(dst)
        p.save()
        p.setClipPath(clip)
        p.fillRect(dst, QColor("black"))
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        p.drawImage(dst, self._img, src)
        p.restore()
        p.setPen(QPen(QColor(ACCENT), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(dst)
        p.setPen(QPen(QColor(255, 255, 255, 200), 1))
        p.drawLine(QPointF(cx - 10, cy), QPointF(cx + 10, cy))
        p.drawLine(QPointF(cx, cy - 10), QPointF(cx, cy + 10))

    # ------------------------------------------------------------------ input
    def mouseMoveEvent(self, e):
        self._mouse = e.position()
        if self._picking:
            self.update()

    def leaveEvent(self, _):
        self._mouse = None
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            v = self._to_video(e.position())
            if v is not None:
                self.clicked.emit(*v)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(Path(u.toLocalFile()).suffix.lower() in VIDEO_EXTS
                                          for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if Path(u.toLocalFile()).suffix.lower() in VIDEO_EXTS:
                self.fileDropped.emit(u.toLocalFile())
                break


class SwatchButton(QPushButton):
    """Round colour swatch, checkable."""

    def __init__(self, color: str, tooltip: str = "", size: int = 30, parent=None):
        super().__init__(parent)
        self.color = color
        self.setCheckable(True)
        self.setFixedSize(size + 8, size + 8)
        self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)
        self.badge = ""

    def set_color(self, color: str):
        self.color = color
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        if self.isChecked():
            p.setPen(QPen(QColor(ACCENT), 2.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5))
            r = r.adjusted(2, 2, -2, -2)
        c = QColor(self.color)
        p.setPen(QPen(QColor(255, 255, 255, 60) if c.lightness() < 200 else QColor("#3a4250"), 1))
        p.setBrush(c)
        p.drawEllipse(r)
        if self.badge:
            p.setPen(QColor("#111") if c.lightness() > 140 else QColor("white"))
            p.setFont(QFont("Segoe UI", 10, QFont.Bold))
            p.drawText(r, Qt.AlignCenter, self.badge)
        elif self.underMouse() and not self.isChecked():
            p.setPen(QPen(QColor(255, 255, 255, 90), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5))


class Segmented(QWidget):
    changed = Signal(str)

    def __init__(self, options, current=None, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        for opt in options:
            b = QPushButton(opt)
            b.setObjectName("Seg")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            self.group.addButton(b)
            lay.addWidget(b)
            self.buttons[opt] = b
            b.clicked.connect(lambda _=False, o=opt: self.changed.emit(o))
        self.set_value(current or options[0])

    def set_value(self, v):
        if v in self.buttons:
            self.buttons[v].setChecked(True)

    def value(self) -> str:
        b = self.group.checkedButton()
        return b.text() if b else ""


def section(title: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("Card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(14, 12, 14, 14)
    lay.setSpacing(10)
    t = QLabel(title.upper())
    t.setObjectName("SectionTitle")
    lay.addWidget(t)
    return card, lay


def stat_card(label: str) -> tuple[QFrame, QLabel]:
    f = QFrame()
    f.setObjectName("StatCard")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(10, 8, 10, 8)
    lay.setSpacing(0)
    v = QLabel("–")
    v.setObjectName("Stat")
    l_ = QLabel(label)
    l_.setObjectName("StatLabel")
    lay.addWidget(v)
    lay.addWidget(l_)
    return f, v
