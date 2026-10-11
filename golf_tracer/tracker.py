"""Multi-hypothesis linking of ball candidates into tracks, and picking the ball flight.

Every candidate either extends an existing track (if it falls inside the track's
prediction gate) or starts a new one. At the end the track that looks most like a
golf ball flight - long, fast, smooth, small, consistently detected - wins.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .detector import Candidate
from .path import outlier_mask


@dataclass
class Track:
    tid: int
    frames: list[int] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    sizes: list[float] = field(default_factory=list)

    def __len__(self):
        return len(self.frames)

    def add(self, frame: int, c: Candidate):
        self.frames.append(frame)
        self.xs.append(c.x)
        self.ys.append(c.y)
        self.sizes.append(c.size)

    @property
    def last_frame(self) -> int:
        return self.frames[-1]

    def velocity(self) -> tuple[float, float]:
        if len(self) < 2:
            return 0.0, 0.0
        dt = max(1, self.frames[-1] - self.frames[-2])
        return (self.xs[-1] - self.xs[-2]) / dt, (self.ys[-1] - self.ys[-2]) / dt

    def speed(self) -> float:
        return math.hypot(*self.velocity())

    def predict(self, frame: int, window: int = 6) -> tuple[float, float]:
        n = len(self)
        if n == 1:
            return self.xs[-1], self.ys[-1]
        k = min(n, window)
        t = np.asarray(self.frames[-k:], float) - self.frames[-1]
        deg = 1 if k < 4 else 2
        dt = frame - self.frames[-1]
        cx = np.polyfit(t, self.xs[-k:], deg)
        cy = np.polyfit(t, self.ys[-k:], deg)
        return float(np.polyval(cx, dt)), float(np.polyval(cy, dt))

    def points(self) -> np.ndarray:
        return np.column_stack([self.xs, self.ys])

    def slice(self, a: int, b: int | None = None) -> "Track":
        out = Track(self.tid)
        out.frames, out.xs, out.ys, out.sizes = self.frames[a:b], self.xs[a:b], self.ys[a:b], self.sizes[a:b]
        return out


class MultiTracker:
    def __init__(self, frame_size: tuple[int, int], fps: float = 30.0, clutter_decay: float = 1.0):
        self.w, self.h = frame_size
        self.diag = math.hypot(self.w, self.h)
        self.fps = fps
        # longer dropouts (ball crossing a cloud) are bridged later by `stitch`
        self.max_missed = max(4, int(round(fps * 0.1)))
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 0
        self.max_active = 160
        self.clutter = ClutterMap(frame_size, self.diag, clutter_decay)

    def _gate(self, tr: Track, gap: int) -> float:
        n, d = len(tr), self.diag
        if n == 1:
            return min(0.2, 0.14 * gap) * d
        sp = tr.speed()
        if n == 2:
            return max(0.03 * d, 0.7 * sp) + 0.5 * sp * (gap - 1)
        # uncertainty grows with the number of missed frames, but only additively
        return max(0.015 * d, 0.45 * sp) + (0.3 * sp + 0.002 * d) * (gap - 1)

    def update(self, frame: int, cands: list[Candidate]):
        xy = np.array([[c.x, c.y] for c in cands]) if cands else np.zeros((0, 2))
        self.clutter.add(xy)
        if cands:
            pairs = []
            for ti, tr in enumerate(self.active):
                gap = frame - tr.last_frame
                if gap <= 0:
                    continue
                px, py = tr.predict(frame)
                g = self._gate(tr, gap)
                d = np.hypot(xy[:, 0] - px, xy[:, 1] - py)
                ok = d < g
                n = len(tr)
                if n >= 3 and ok.any():
                    vx, vy = tr.velocity()
                    vn = math.hypot(vx, vy)
                    if vn * gap > 0.004 * self.diag:
                        sx, sy = xy[:, 0] - tr.xs[-1], xy[:, 1] - tr.ys[-1]
                        step = np.hypot(sx, sy)
                        cos = (sx * vx + sy * vy) / (step * vn + 1e-9)
                        # direction changes gradually (<60 deg per step) once a track is established
                        ok &= cos > (0.5 if n >= 4 else -0.1)
                        # a ball's image speed changes smoothly: it never stops dead or jumps
                        if vn * gap > 0.006 * self.diag:
                            ok &= step > 0.5 * vn * gap
                        ok &= step < 2.2 * vn * gap + 0.01 * self.diag
                pen = 1.0 if n >= 3 else (1.6 if n == 2 else 3.0)
                for ci in np.nonzero(ok)[0]:
                    pairs.append((d[ci] / g * pen, -n, ti, int(ci)))
            pairs.sort()
            used_t, used_c = set(), set()
            for _, _, ti, ci in pairs:
                if ti in used_t or ci in used_c:
                    continue
                used_t.add(ti)
                used_c.add(ci)
                self.active[ti].add(frame, cands[ci])
        else:
            used_c = set()

        survivors = []
        for tr in self.active:
            limit = self.max_missed if len(tr) >= 3 else 2
            if frame - tr.last_frame > limit:
                if len(tr) >= 4:
                    self.finished.append(tr)
            else:
                survivors.append(tr)
        for ci, c in enumerate(cands):
            if ci not in used_c and c.area >= 2:   # 1-px blobs may only extend tracks
                tr = Track(self._next_id)
                self._next_id += 1
                tr.add(frame, c)
                survivors.append(tr)
        if len(survivors) > self.max_active:
            survivors.sort(key=lambda t: (len(t), t.last_frame), reverse=True)
            survivors = survivors[: self.max_active]
        self.active = survivors

    def all_tracks(self, min_len: int = 4) -> list[Track]:
        return [t for t in self.finished + self.active if len(t) >= min_len]

    def prune_finished(self, before_frame: int):
        self.finished = [t for t in self.finished if t.last_frame >= before_frame]


# ---------------------------------------------------------------------- scoring
class ClutterMap:
    """Counts candidates per coarse grid cell over time. Spots that keep firing
    (flickering edges, swaying leaves, water) are clutter; a ball passes through once."""

    def __init__(self, frame_size: tuple[int, int], diag: float, decay: float = 1.0):
        self.cell = max(4.0, 0.012 * diag)
        self.gw = int(frame_size[0] / self.cell) + 1
        self.gh = int(frame_size[1] / self.cell) + 1
        self.heat = np.zeros((self.gh, self.gw), np.float32)
        self.decay = decay

    def _cells(self, xy: np.ndarray):
        cx = np.clip((xy[:, 0] / self.cell).astype(int), 0, self.gw - 1)
        cy = np.clip((xy[:, 1] / self.cell).astype(int), 0, self.gh - 1)
        return cy, cx

    def add(self, xy: np.ndarray):
        if self.decay < 1.0:
            self.heat *= self.decay
        if len(xy):
            np.add.at(self.heat, self._cells(xy), 1.0)

    def clutter(self, pts: np.ndarray) -> float:
        """Median count of *other* detections in the cells this track passes through."""
        cy, cx = self._cells(pts)
        flat = cy * self.gw + cx
        _, inv, own = np.unique(flat, return_inverse=True, return_counts=True)
        other = self.heat[cy, cx] - own[inv]
        return float(np.median(np.maximum(other, 0)))


def score_track(tr: Track, diag: float, seed: tuple[int, float, float] | None = None,
                clutter: ClutterMap | None = None, fps: float = 30.0) -> float:
    n = len(tr)
    if n < 4:
        return 0.0
    p = tr.points()
    t = np.asarray(tr.frames, float)
    steps = np.diff(p, axis=0)
    step_len = np.hypot(steps[:, 0], steps[:, 1])
    path = step_len.sum()
    duration = t[-1] - t[0]
    if path < 1e-6 or duration <= 0:
        return 0.0
    if path / diag / (duration / fps) < 0.024:   # < 2.4 % of the diagonal per second: not a ball in flight
        return 0.0
    # turning: a ball's direction changes gradually from step to step; noise zig-zags
    moving = step_len > 0.5
    s = steps[moving] / step_len[moving, None]
    turn = float(np.clip(np.mean(np.sum(s[1:] * s[:-1], axis=1)), 0, 1)) if len(s) >= 2 else 0.0
    # jitter: how far each point is from the line between its neighbours, relative to step size
    w = (t[1:-1] - t[:-2]) / (t[2:] - t[:-2])
    interp = p[:-2] + (p[2:] - p[:-2]) * w[:, None]
    dev = np.hypot(*(p[1:-1] - interp).T)
    local = 0.5 * (step_len[:-1] + step_len[1:]) + 1.0
    jitter = float(np.median(dev / local))
    coverage = n / (duration + 1)
    size = float(np.median(tr.sizes))
    size_factor = 1.0 / (1.0 + (size / 0.012) ** 2)

    score = math.sqrt(n) * (path / diag) * turn ** 2 / (1 + 4 * jitter) * coverage * size_factor * 4
    if clutter is not None:
        score /= 1 + (clutter.clutter(p) / 3.0) ** 2
    if p[min(3, n - 1), 1] < p[0, 1]:       # golf shots leave upwards in the image
        score *= 1.3
    if seed is not None:
        sf, sx, sy = seed
        d = math.hypot(p[0, 0] - sx, p[0, 1] - sy) / diag
        if tr.frames[0] < sf:
            score *= 0.3
        elif d < 0.12:
            score *= 1 + 3 * (1 - d / 0.12)
        else:
            score *= 0.5
    return float(score)


def _predict(fs: list, xs: list, ys: list, f: float, k: int = 8):
    """Quadratic (or linear) fit of the last k points evaluated at time f.
    Returns (x, y, speed at the last point)."""
    t = np.asarray(fs[-k:], float) - fs[-1]
    deg = 2 if len(t) >= 4 else 1
    cx = np.polyfit(t, xs[-k:], deg)
    cy = np.polyfit(t, ys[-k:], deg)
    dt = f - fs[-1]
    v = math.hypot(np.polyval(np.polyder(cx), 0.0), np.polyval(np.polyder(cy), 0.0))
    return float(np.polyval(cx, dt)), float(np.polyval(cy, dt)), v


def _gate_ok(F, X, Y, cur, c, diag, tight=False):
    """Indices of candidates `c` (N, 2) consistent with the trajectory at time `cur`,
    best first."""
    px, py, v = _predict(F, X, Y, cur)
    gap = cur - F[-1]
    if tight:
        tol = max(0.008 * diag, 0.35 * v) + (0.1 * v + 0.001 * diag) * (gap - 1)
    else:
        tol = max(0.012 * diag, 0.35 * v) + (0.15 * v + 0.002 * diag) * (gap - 1)
    d = np.hypot(c[:, 0] - px, c[:, 1] - py)
    ok = d < tol
    if v * gap > 0.006 * diag:     # speed must stay plausible: a ball never stops dead
        step = np.hypot(c[:, 0] - X[-1], c[:, 1] - Y[-1])
        ok &= (step > 0.4 * v * gap) & (step < 2.2 * v * gap + 0.01 * diag)
    idx = np.nonzero(ok)[0]
    return idx[np.argsort(d[idx])]


def _confirm(F, X, Y, cur, c, cands, diag, sign, f_min, f_max, look=6, need=3) -> bool:
    """After a dropout, accept a re-acquired candidate only if the following frames
    continue smoothly from it (noise near clouds / edges doesn't)."""
    F, X, Y = F + [cur], X + [float(c[0])], Y + [float(c[1])]
    hits = 0
    for t in range(cur + 1, cur + 1 + look):
        real = sign * t
        if real < f_min or real > f_max:
            break
        cc = cands.get(real)
        if cc is None or len(cc) == 0:
            continue
        idx = _gate_ok(F, X, Y, t, cc, diag, tight=True)
        if len(idx):
            F.append(t)
            X.append(float(cc[idx[0], 0]))
            Y.append(float(cc[idx[0], 1]))
            hits += 1
            if hits >= need:
                return True
    return False


def _grow_one_way(fs, xs, ys, cands: dict, diag: float, max_gap: int, sign: int, f_min: int, f_max: int):
    """Extend a trajectory frame by frame through raw per-frame candidates.
    sign=+1 grows forwards in time, -1 backwards (time is mirrored so the same code works)."""
    F = [sign * f for f in fs]
    X, Y = list(xs), list(ys)
    if sign < 0:
        F, X, Y = F[::-1], X[::-1], Y[::-1]
    cur = F[-1]
    while cur - F[-1] <= max_gap:
        cur += 1
        real = sign * cur
        if real < f_min or real > f_max:
            break
        c = cands.get(real)
        if c is None or len(c) == 0:
            continue
        for i in _gate_ok(F, X, Y, cur, c, diag):
            if cur - F[-1] < 3 or _confirm(F, X, Y, cur, c[i], cands, diag, sign, f_min, f_max):
                F.append(cur)
                X.append(float(c[i, 0]))
                Y.append(float(c[i, 1]))
                break
    if sign < 0:
        F, X, Y = F[::-1], X[::-1], Y[::-1]
    return [sign * f for f in F], X, Y


def grow_track(tr: Track, cands: dict, diag: float, max_gap: int, f_min: int, f_max: int,
               backwards: bool = True) -> Track:
    """Re-trace the flight through the raw detections starting from a confident
    fragment. Bridges dropouts (e.g. a white ball crossing a white cloud) that broke
    the online tracker into pieces."""
    if len(tr) < 4:
        return tr
    fs, xs, ys = _grow_one_way(tr.frames, tr.xs, tr.ys, cands, diag, max_gap, +1, f_min, f_max)
    if backwards:
        fs, xs, ys = _grow_one_way(fs, xs, ys, cands, diag, max_gap // 3, -1, f_min, f_max)
    out = Track(tr.tid)
    size = float(np.median(tr.sizes))
    known = dict(zip(tr.frames, tr.sizes))
    out.frames, out.xs, out.ys = list(fs), list(xs), list(ys)
    out.sizes = [known.get(f, size) for f in fs]
    return out


def clean_track(tr: Track, diag: float) -> Track:
    """Drop points that don't fit the smooth flight (noise grabbed during a dropout)."""
    if len(tr) < 6:
        return tr
    keep = outlier_mask(np.asarray(tr.frames, float), tr.points(), tol=max(2.0, 0.004 * diag))
    out = Track(tr.tid)
    for i in np.nonzero(keep)[0]:
        out.frames.append(tr.frames[i]); out.xs.append(tr.xs[i])
        out.ys.append(tr.ys[i]); out.sizes.append(tr.sizes[i])
    # a flight starts with consecutive detections: drop a few stray points before a time gap
    while len(out) > 6:
        gaps = np.diff(out.frames[:5])
        cut = next((i + 1 for i, g in enumerate(gaps) if g > 3), None)
        if cut is None:
            break
        out = out.slice(cut)
    return out


def split_stationary_start(tr: Track, diag: float) -> tuple[Track, tuple[float, float] | None]:
    """A track often begins on the ball sitting at address (the club passing over it
    looks like motion). Cut that part off; return (flight, address position)."""
    n = len(tr)
    p = tr.points()
    still = 0.006 * diag
    k = 0
    while k + 1 < n and math.hypot(*(p[k + 1] - p[0])) < still:
        k += 1
    if k == 0 or n - k - 1 < 4:
        return tr, None
    addr = p[: k + 1].mean(axis=0)
    return tr.slice(k + 1), (float(addr[0]), float(addr[1]))


def pick_best(tracks: list[Track], diag: float, fps: float,
              seed: tuple[int, float, float] | None = None,
              clutter: ClutterMap | None = None,
              cands: dict | None = None, frame_range: tuple[int, int] = (0, 10 ** 9)):
    """Pick the ball flight. Returns (flight track, score, address position or None).

    cands: raw candidates per frame {frame: (N, 2) array}; when given the winning
    fragment is re-grown through them to recover the full flight."""
    if not tracks:
        return None, 0.0, None
    best, best_score, addr = None, 0.0, None
    for t in tracks:
        flight, a = split_stationary_start(t, diag)
        flight = clean_track(flight, diag)
        sc = score_track(flight, diag, seed, clutter, fps)
        if sc > best_score:
            best, best_score, addr = flight, sc, a
    if best is None:
        return None, 0.0, None
    if cands is not None:
        f_min = max(frame_range[0], seed[0] if seed else 0)
        best = grow_track(best, cands, diag, max(10, int(fps * 0.5)), f_min, frame_range[1],
                          backwards=addr is None)
        best = clean_track(best, diag)
    return best, max(best_score, score_track(best, diag, seed, clutter, fps)), addr
