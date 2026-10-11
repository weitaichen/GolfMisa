"""Ball colour profiles (HSV ranges) and the tracer colour palette.

OpenCV HSV scale: H 0-179, S 0-255, V 0-255.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

HSV = tuple[int, int, int]
HSVRange = tuple[HSV, HSV]


@dataclass(frozen=True)
class BallColor:
    key: str
    name: str
    ranges: tuple[HSVRange, ...]
    swatch: str  # hex colour used by the UI

    @property
    def chromatic(self) -> bool:
        return all(lo[1] >= 50 for lo, _ in self.ranges)

    def mask(self, hsv: np.ndarray, tolerance: float = 0.5) -> np.ndarray:
        """Binary mask (0/255) of pixels matching this ball colour.

        `tolerance` (0..1) widens hue and saturation/value bounds so motion-blurred
        balls, whose colour is diluted by the background, are still matched.
        """
        dh = int(round(10 * tolerance))
        dsv = int(round(60 * tolerance))
        out = None
        for lo, hi in self.ranges:
            full_hue = lo[0] <= 0 and hi[0] >= 179
            s_floor = 35 if lo[1] >= 50 else 0
            lo2 = np.array([0 if full_hue else max(0, lo[0] - dh),
                            max(s_floor, lo[1] - dsv),
                            max(0, lo[2] - dsv)], np.uint8)
            hi2 = np.array([179 if full_hue else min(179, hi[0] + dh),
                            min(255, hi[1] + dsv),
                            min(255, hi[2] + dsv)], np.uint8)
            m = cv2.inRange(hsv, lo2, hi2)
            out = m if out is None else cv2.bitwise_or(out, m)
        return out


BALL_COLORS: list[BallColor] = [
    BallColor("white", "White", (((0, 0, 165), (179, 70, 255)),), "#f4f4f4"),
    BallColor("yellow", "Yellow", (((18, 70, 110), (38, 255, 255)),), "#ffe23d"),
    BallColor("orange", "Orange", (((5, 110, 120), (20, 255, 255)),), "#ff8a1f"),
    BallColor("red", "Red", (((0, 100, 80), (6, 255, 255)), ((165, 100, 80), (179, 255, 255))), "#ff2b2b"),
    BallColor("pink", "Pink", (((140, 50, 120), (170, 255, 255)),), "#ff4fc8"),
    BallColor("green", "Lime", (((36, 80, 110), (80, 255, 255)),), "#7dff3a"),
    BallColor("blue", "Blue", (((92, 90, 70), (128, 255, 255)),), "#2f8cff"),
]

BALL_COLOR_BY_KEY = {c.key: c for c in BALL_COLORS}


def bgr_to_hex(bgr) -> str:
    b, g, r = (int(v) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def color_from_sample(bgr) -> BallColor:
    """Build a custom ball profile around a colour sampled from the video."""
    h, s, v = (int(x) for x in cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0])
    if s < 45:  # white / grey / silver ball
        ranges = (((0, 0, max(60, v - 70)), (179, min(255, s + 50), 255)),)
    else:
        lo_s, lo_v = min(s, max(50, s - 90)), min(v, max(60, v - 90))   # always include the sample
        h_lo, h_hi = h - 10, h + 10
        if h_lo < 0:
            ranges = (((0, lo_s, lo_v), (h_hi, 255, 255)), ((180 + h_lo, lo_s, lo_v), (179, 255, 255)))
        elif h_hi > 179:
            ranges = (((h_lo, lo_s, lo_v), (179, 255, 255)), ((0, lo_s, lo_v), (h_hi - 180, 255, 255)))
        else:
            ranges = (((h_lo, lo_s, lo_v), (h_hi, 255, 255)),)
    return BallColor("custom", "Custom", ranges, bgr_to_hex(bgr))


def sample_color(frame: np.ndarray, x: float, y: float, radius: int = 2):
    """Representative BGR colour of the object clicked at (x, y).

    The background colour is estimated from a ring around the click; the pixels in the
    small patch that differ most from it are the ball. This works for a white ball on
    grass as well as a red ball on a white tee mat, and tolerates a slightly-off click."""
    hgt, wid = frame.shape[:2]
    xi, yi = int(round(x)), int(round(y))

    def box(r):
        return frame[max(0, yi - r):min(hgt, yi + r + 1), max(0, xi - r):min(wid, xi + r + 1)]

    patch = box(radius).reshape(-1, 3).astype(np.float32)
    if len(patch) == 0:
        return (255, 255, 255)
    outer = box(radius * 4 + 2).astype(np.float32)
    ring = np.concatenate([outer[0], outer[-1], outer[:, 0], outer[:, -1]])
    background = np.median(ring, axis=0)
    dist = np.linalg.norm(patch - background, axis=1)
    keep = dist >= 0.5 * dist.max()
    return tuple(int(c) for c in np.median(patch[keep], axis=0))


# Tracer palette (name, hex) - SmoothSwing style bright colours.
TRACER_COLORS: list[tuple[str, str]] = [
    ("Red", "#ff2d2d"),
    ("Orange", "#ff8c1a"),
    ("Yellow", "#ffe600"),
    ("Green", "#2dff6a"),
    ("Cyan", "#22e4ff"),
    ("Blue", "#3d7bff"),
    ("Purple", "#b44dff"),
    ("White", "#ffffff"),
]
