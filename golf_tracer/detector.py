"""Per-frame ball candidate detection: symmetric temporal difference ∩ ball colour mask.

A ball in flight is the only thing that is both (a) moving and (b) the ball colour,
so intersecting a colour mask with a motion mask removes the static ball-coloured
background (clouds, white lines, flags) and the moving non-ball-coloured golfer.

Motion at frame t is "differs from frame t-k AND from frame t+k". Using several
baselines k = 1, 2, 4 keeps detecting the ball once it is far away and moves less
than its own diameter per frame.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .colors import BallColor

BASELINES = (1, 2, 4)
_HALF = max(BASELINES)


@dataclass
class Candidate:
    x: float      # centre, original video pixels
    y: float
    area: float   # blob area in processing pixels
    size: float   # sqrt(area) / processing diagonal (resolution independent)
    elong: float  # bounding box aspect ratio (motion blur makes streaks)


@dataclass
class DetectorSettings:
    sensitivity: float = 0.5   # 0 = strict, 1 = very permissive
    proc_width: int = 960      # frames are downscaled so the long side is this size, for speed
    stabilize: bool = True     # compensate small camera shake (hand-held phone)
    max_candidates: int = 30


@dataclass
class _Frame:
    idx: int
    img: np.ndarray            # blurred, downscaled BGR
    pos: np.ndarray            # cumulative content offset (camera shake), processing pixels
    gray: np.ndarray | None    # half-size float gray for shake estimation
    lo: np.ndarray | None = None   # 3x3 min / max filtered img (tolerance for sub-pixel shifts)
    hi: np.ndarray | None = None


def _shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate image content by whole pixels, replicating the borders."""
    h, w = img.shape[:2]
    dx = int(np.clip(dx, -(w - 1), w - 1))
    dy = int(np.clip(dy, -(h - 1), h - 1))
    src = img[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return cv2.copyMakeBorder(src, max(0, dy), max(0, -dy), max(0, dx), max(0, -dx), cv2.BORDER_REPLICATE)


class BallDetector:
    """Streaming detector. Push frames in order; results lag `_HALF` frames because the
    motion test for frame t needs frames up to t + _HALF."""

    def __init__(self, color: BallColor, settings: DetectorSettings | None = None):
        self.color = color
        self.s = settings or DetectorSettings()
        self._buf: deque[_Frame] = deque(maxlen=2 * _HALF + 1)
        self.scale: float | None = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._window = None
        self._prev: _Frame | None = None
        self._primed = False

    def reset(self):
        self._buf.clear()
        self._prev = None
        self._primed = False
        self.scale = None          # re-derive geometry from the next frame

    # ------------------------------------------------------------------ helpers
    def _init_geometry(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        self.scale = min(1.0, self.s.proc_width / float(max(w, h)))
        self.size = (max(1, int(round(w * self.scale))), max(1, int(round(h * self.scale))))
        self.diag = math.hypot(*self.size)
        self.max_area = (0.035 * max(self.size)) ** 2
        self._gsize = (max(16, self.size[0] // 2), max(16, self.size[1] // 2))
        self._window = cv2.createHanningWindow(self._gsize, cv2.CV_32F)

    def _prep(self, idx: int, frame: np.ndarray) -> _Frame:
        if self.scale is None:
            self._init_geometry(frame)
        if self.scale < 1.0:
            small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        else:
            small = frame.copy()
        small = cv2.GaussianBlur(small, (3, 3), 0)
        pos = np.zeros(2)
        gray = None
        if self.s.stabilize:
            gray = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), self._gsize,
                              interpolation=cv2.INTER_AREA).astype(np.float32)
            if self._prev is not None and self._prev.gray is not None:
                (dx, dy), resp = cv2.phaseCorrelate(self._prev.gray, gray, self._window)
                d = np.array([dx, dy]) * 2.0          # half-size -> processing pixels
                if resp < 0.05 or abs(d[0]) > 0.05 * self.size[0] or abs(d[1]) > 0.05 * self.size[1]:
                    d[:] = 0
                pos = self._prev.pos + d
        f = _Frame(idx, small, pos, gray, cv2.erode(small, self._kernel), cv2.dilate(small, self._kernel))
        self._prev = f
        return f

    def _aligned(self, f: _Frame, ref: _Frame) -> tuple[np.ndarray, np.ndarray]:
        """Min/max-filtered neighbour frame, shifted to line up with `ref`. Whole-pixel
        shifts are enough: the 3x3 min/max tolerance absorbs the sub-pixel remainder."""
        dx, dy = (int(round(v)) for v in ref.pos - f.pos)
        if dx == 0 and dy == 0:
            return f.lo, f.hi
        return _shift(f.lo, dx, dy), _shift(f.hi, dx, dy)

    def _diff(self, cur: np.ndarray, other: tuple[np.ndarray, np.ndarray], thr: int) -> np.ndarray:
        """Pixels of `cur` outside the 3x3 neighbourhood range of the other frame, so
        high-contrast edges shifting by a pixel (shake, noise) don't count as motion.
        Max over B,G,R so colour-only contrast (red ball on green grass) counts."""
        lo, hi = other
        d = cv2.max(cv2.subtract(cur, hi), cv2.subtract(lo, cur))
        b_, g_, r_ = cv2.split(d)
        m = cv2.max(cv2.max(b_, g_), r_)
        return cv2.threshold(m, thr, 255, cv2.THRESH_BINARY)[1]

    # --------------------------------------------------------------------- API
    def push(self, idx: int, frame: np.ndarray) -> list:
        """Add frame `idx`. Returns a list of (frame_index, candidates) for earlier frames
        that are now complete (usually one; the first frames of a clip come out together
        once the buffer has filled)."""
        self._buf.append(self._prep(idx, frame))
        if len(self._buf) < self._buf.maxlen:
            return []
        if not self._primed:
            self._primed = True
            return [self._detect_at(c) for c in range(0, _HALF + 1)]
        return [self._detect_at(_HALF)]

    def flush(self) -> list:
        """Detections for the trailing frames still in the buffer."""
        buf = list(self._buf)
        out = []
        start = _HALF + 1 if self._primed else 0
        for c in range(start, len(buf)):
            out.append(self._detect_at(c))
        self._buf.clear()
        self._primed = False
        return out

    def threshold(self) -> int:
        return int(round(np.interp(self.s.sensitivity, [0.0, 1.0], [36, 8])))

    def _detect_at(self, c: int):
        buf = self._buf
        cur = buf[c]
        thr = self.threshold()
        motion = None
        for k in BASELINES:
            before, after = c - k >= 0, c + k < len(buf)
            if before and after:
                m = cv2.bitwise_and(self._diff(cur.img, self._aligned(buf[c - k], cur), thr),
                                    self._diff(cur.img, self._aligned(buf[c + k], cur), thr))
            elif k == 1 and (before or after):
                # one-sided test, only at the very start / end of the clip
                m = self._diff(cur.img, self._aligned(buf[c - k if before else c + k], cur), thr)
            else:
                continue
            motion = m if motion is None else cv2.bitwise_or(motion, m)
        if motion is None:
            return cur.idx, []
        motion = cv2.dilate(motion, self._kernel)
        cmask = self.color.mask(cv2.cvtColor(cur.img, cv2.COLOR_BGR2HSV), self.s.sensitivity)
        comb = cv2.bitwise_and(motion, cmask)
        return cur.idx, self._blobs(comb)

    def _blobs(self, comb) -> list[Candidate]:
        n, _, stats, cents = cv2.connectedComponentsWithStats(comb, connectivity=8)
        inv = 1.0 / self.scale
        W, H = self.size
        out: list[Candidate] = []
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if area > self.max_area:
                continue
            if bx <= 2 or by <= 2 or bx + bw >= W - 2 or by + bh >= H - 2:
                continue      # frame edges flicker with shake / stabilisation borders
            elong = max(bw, bh) / max(1, min(bw, bh))
            if elong > 8:
                continue
            cx, cy = cents[i]
            out.append(Candidate(cx * inv, cy * inv, float(area), math.sqrt(area) / self.diag, float(elong)))
        if len(out) > self.s.max_candidates:
            out.sort(key=lambda cand: -cand.area)
            out = out[: self.s.max_candidates]
        return out
