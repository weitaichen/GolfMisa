"""Drawing the shot tracer (glow / solid / comet) and debug overlays."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .colors import hex_to_bgr

STYLES = ("Glow", "Solid", "Comet")
_SHIFT = 4                     # sub-pixel precision for cv2 drawing
_ONE = 1 << _SHIFT


@dataclass
class TracerStyle:
    color: str = "#ff2d2d"
    style: str = "Glow"
    thickness: float = 6.0     # line width at 1080p; scaled with frame height
    ball_marker: bool = True


def _fixed(pts: np.ndarray) -> np.ndarray:
    return np.round(pts * _ONE).astype(np.int32).reshape(-1, 1, 2)


def _lighten(bgr, amount=0.6):
    return tuple(int(c + (255 - c) * amount) for c in bgr)


def _roi(img, pts, pad):
    h, w = img.shape[:2]
    x0 = int(max(0, np.floor(pts[:, 0].min() - pad)))
    y0 = int(max(0, np.floor(pts[:, 1].min() - pad)))
    x1 = int(min(w, np.ceil(pts[:, 0].max() + pad)))
    y1 = int(min(h, np.ceil(pts[:, 1].max() + pad)))
    return x0, y0, x1, y1


def draw_tracer(img: np.ndarray, pts: np.ndarray, style: TracerStyle, head_active: bool = True):
    """Draw the tracer polyline `pts` (image pixel coords) onto `img` in place."""
    if len(pts) == 0:
        return img
    color = hex_to_bgr(style.color)
    th = max(1, int(round(style.thickness * img.shape[0] / 1080.0)))
    pad = th * 5 + 4
    x0, y0, x1, y1 = _roi(img, pts, pad)
    if x1 <= x0 or y1 <= y0:
        return img
    roi = img[y0:y1, x0:x1]
    local = pts - np.array([x0, y0], float)
    poly = _fixed(local)

    if len(pts) >= 2:
        if style.style == "Solid":
            cv2.polylines(roi, [poly], False, (20, 20, 20), th + 2, cv2.LINE_AA, _SHIFT)
            cv2.polylines(roi, [poly], False, color, th, cv2.LINE_AA, _SHIFT)
        elif style.style == "Comet":
            _comet(roi, local, color, th)
        else:
            _blend(roi, _soft_line(roi.shape[:2], local, th * 3, th * 1.3), color, 0.65)
            cv2.polylines(roi, [poly], False, color, th, cv2.LINE_AA, _SHIFT)
            cv2.polylines(roi, [poly], False, _lighten(color), max(1, th // 3), cv2.LINE_AA, _SHIFT)

    if style.ball_marker and head_active:
        _marker(img, pts[-1], th, color)
    return img


def _marker(img, head, th, color):
    """White ball dot with a coloured halo, drawn in a small patch around the ball."""
    r = max(2, int(round(th * 1.1)))
    x0, y0, x1, y1 = _roi(img, head[None, :], r * 5)
    if x1 <= x0 or y1 <= y0:
        return
    patch = img[y0:y1, x0:x1]
    c = (int((head[0] - x0) * _ONE), int((head[1] - y0) * _ONE))
    glow = np.zeros(patch.shape[:2], np.uint8)
    cv2.circle(glow, c, r * 2 * _ONE, 255, -1, cv2.LINE_AA, _SHIFT)
    _blend(patch, cv2.GaussianBlur(glow, (0, 0), r), color, 0.8)
    cv2.circle(patch, c, r * _ONE, (255, 255, 255), -1, cv2.LINE_AA, _SHIFT)


def _soft_line(shape, pts, width, sigma, k=4):
    """Blurred line mask, computed at 1/k resolution (blur is the expensive part)."""
    h, w = shape
    small = np.zeros((max(1, h // k), max(1, w // k)), np.uint8)
    cv2.polylines(small, [_fixed(pts / k)], False, 255, max(1, int(round(width / k))), cv2.LINE_AA, _SHIFT)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / k))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _blend(roi, alpha: np.ndarray, color, strength: float = 1.0):
    """roi = roi * (1 - a) + color * a  (keeps the tracer hue true on any background)."""
    a = alpha.astype(np.float32) * (strength / 255.0)
    solid = np.empty_like(roi)
    solid[:] = color
    roi[:] = cv2.blendLinear(roi, solid, 1.0 - a, a)


def _comet(roi, local, color, th):
    """Tail fades in and thickens towards the ball."""
    n = len(local)
    alpha = np.zeros(roi.shape[:2], np.uint8)
    for i in range(1, n):
        f = i / (n - 1)
        a = int(255 * f ** 1.5)
        w = max(1, int(round(th * (0.35 + 0.65 * f))))
        cv2.line(alpha, tuple(int(v) for v in local[i - 1] * _ONE), tuple(int(v) for v in local[i] * _ONE),
                 a, w, cv2.LINE_AA, _SHIFT)
    h, w = alpha.shape
    small = cv2.resize(alpha, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    soft = cv2.resize(cv2.GaussianBlur(small, (0, 0), max(0.5, th / 4)), (w, h))
    _blend(roi, cv2.max(alpha, cv2.convertScaleAbs(soft, alpha=0.8)), color)


def draw_candidates(img, cands: np.ndarray | None, scale: float = 1.0):
    if cands is None:
        return img
    for x, y in cands * scale:
        cv2.circle(img, (int(x), int(y)), 6, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def draw_points(img, pts: np.ndarray, color=(0, 255, 255), r=2):
    for x, y in pts:
        cv2.circle(img, (int(round(x)), int(round(y))), r, color, -1, cv2.LINE_AA)
    return img
