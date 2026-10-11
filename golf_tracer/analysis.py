"""Whole-video shot analysis (offline) and real-time tracking (live camera)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .colors import BallColor
from .detector import BallDetector, DetectorSettings
from .path import TracerPath, smooth_path
from .tracker import MultiTracker, pick_best, score_track, split_stationary_start

MIN_SCORE = 2.5      # real flights score ~5-20, noise / wrong-colour tracks < 2.5


class Cancelled(Exception):
    pass


@dataclass
class AnalysisSettings:
    color: BallColor
    sensitivity: float = 0.5
    stabilize: bool = True
    proc_width: int = 960
    start_frame: int = 0
    end_frame: int | None = None


@dataclass
class ShotResult:
    fps: float
    frame_size: tuple[int, int]
    frames: np.ndarray | None = None       # tracked ball frames (address point first, if found)
    pts: np.ndarray | None = None
    address: tuple[float, float] | None = None
    score: float = 0.0
    candidates: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.frames is not None and len(self.frames) >= 2

    @property
    def launch_frame(self) -> int:
        return int(self.frames[0])

    @property
    def last_frame(self) -> int:
        return int(self.frames[-1])

    @property
    def hang_time(self) -> float:
        return (self.last_frame - self.launch_frame) / self.fps

    @property
    def confidence(self) -> str:
        if self.score > 8.0:
            return "High"
        if self.score > 4.0:
            return "Medium"
        return "Low"

    def build_path(self, extend: int = 0) -> TracerPath | None:
        if not self.found:
            return None
        return smooth_path(self.frames, self.pts, self.frame_size, extend, anchored=self.address is not None)


def _video_props(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 1000:
        fps = 30.0
    return fps, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), \
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def _read_frame(path: str, idx: int):
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _color_blobs(frame, color: BallColor, proc_width: int, tolerance=0.5):
    """Stationary ball-coloured, ball-shaped blobs: (x, y, area) in original pixels."""
    h, w = frame.shape[:2]
    s = min(1.0, proc_width / max(w, h))
    small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
    mask = color.mask(cv2.cvtColor(cv2.GaussianBlur(small, (3, 3), 0), cv2.COLOR_BGR2HSV), tolerance)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_area = (0.03 * max(small.shape[:2])) ** 2
    out = []
    for i in range(1, n):
        bw, bh, a = stats[i, 2], stats[i, 3], stats[i, 4]
        if 3 <= a <= max_area and max(bw, bh) / max(1, min(bw, bh)) < 2.2:
            out.append((cents[i][0] / s, cents[i][1] / s, a / (s * s)))
    return out


def find_address_ball(path, color, frames, pts, frame_size, proc_width=960):
    """Find the ball sitting at address just before launch, by looking back along the
    flight direction from the first tracked position. Returns (x, y) or None."""
    if len(frames) < 3 or frames[0] < 4:
        return None
    diag = math.hypot(*frame_size)
    pre = _read_frame(path, int(frames[0]) - 4)
    if pre is None:
        return None
    p0 = pts[0]
    back = p0 - pts[min(2, len(pts) - 1)]
    nb = np.hypot(*back)
    if nb < 1e-6:
        return None
    back /= nb
    best, best_cost = None, 1e18
    for x, y, _ in _color_blobs(pre, color, proc_width):
        v = np.array([x, y]) - p0
        along = float(v @ back)
        perp = abs(float(v[0] * back[1] - v[1] * back[0]))
        if -0.02 * diag < along < 0.25 * diag and perp < 0.05 * diag + 0.15 * max(along, 0):
            cost = perp + 0.1 * along
            if cost < best_cost:
                best, best_cost = (x, y), cost
    if best is None:
        return None
    # the real ball has left the tee once the shot is in the air
    post = _read_frame(path, int(frames[min(len(frames) - 1, 4)]))
    if post is not None:
        r = 0.01 * diag
        for x, y, _ in _color_blobs(post, color, proc_width):
            if math.hypot(x - best[0], y - best[1]) < r:
                return None
    return best


def _with_address(frames, pts, addr, min_frame=0):
    """Prepend the address position at the estimated moment of impact.
    Returns (frames, pts, prepended)."""
    still = 0.006 * math.hypot(*(pts.max(axis=0) - pts.min(axis=0)) + 1e-9) + 6.0
    while len(frames) > 4 and math.hypot(pts[0][0] - addr[0], pts[0][1] - addr[1]) < still:
        frames, pts = frames[1:], pts[1:]        # impact frame: ball hasn't really left yet
    sp = np.hypot(*(pts[1] - pts[0])) / max(1, frames[1] - frames[0])
    d = math.hypot(pts[0][0] - addr[0], pts[0][1] - addr[1])
    lead = int(np.clip(round(d / max(sp, 1.0)), 1, 6))
    launch = max(min_frame, int(frames[0]) - lead)
    if launch >= frames[0]:
        return frames, pts, False
    return np.concatenate([[launch], frames]), np.vstack([[addr], pts]), True


def analyze_video(path: str, settings: AnalysisSettings, seed: tuple[int, float, float] | None = None,
                  progress=None, cancelled=None) -> ShotResult:
    """Detect and track the ball through the whole clip.

    seed: optional (frame, x, y) of the ball at address, clicked by the user.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps, w, h, count = _video_props(cap)
    det = BallDetector(settings.color, DetectorSettings(settings.sensitivity, settings.proc_width,
                                                        settings.stabilize))
    trk = MultiTracker((w, h), fps)
    start = max(0, settings.start_frame)
    end = settings.end_frame                 # None: read to the end of the file (frame counts lie)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    total = (end - start + 1) if end is not None else max(0, count - start)
    cands_by_frame: dict[int, np.ndarray] = {}

    def handle(results):
        for fi, cands in results:
            trk.update(fi, cands)
            if cands:
                cands_by_frame[fi] = np.array([[c.x, c.y] for c in cands], np.float32)

    idx = start
    try:
        while end is None or idx <= end:
            if cancelled and cancelled():
                raise Cancelled()
            ok, frame = cap.read()
            if not ok:
                break
            handle(det.push(idx, frame))
            idx += 1
            if progress and total > 0 and idx % 5 == 0:
                progress(min(99, int(100 * (idx - start) / total)))
        handle(det.flush())
    finally:
        cap.release()

    result = ShotResult(fps, (w, h), candidates=cands_by_frame)
    if idx == start:
        raise IOError(f"No frames could be read from: {path}")
    best, score, still_addr = pick_best(trk.all_tracks(), trk.diag, fps, seed, trk.clutter,
                                        cands_by_frame, (start, idx))
    if best is None or score < MIN_SCORE:
        if progress:
            progress(100)
        return result
    frames = np.asarray(best.frames)
    pts = best.points()
    if seed is not None:
        addr = (seed[1], seed[2])
        min_frame = seed[0]
    else:
        addr = still_addr or find_address_ball(path, settings.color, frames, pts, (w, h), settings.proc_width)
        min_frame = 0
    if addr is not None:
        frames, pts, prepended = _with_address(frames, pts, addr, min_frame)
        if not prepended:
            addr = None              # no room for the address before the flight: don't anchor to it
    result.frames, result.pts, result.address, result.score = frames, pts, addr, score
    if progress:
        progress(100)
    return result


class LiveTracker:
    """Frame-by-frame tracking for the live camera view."""

    def __init__(self, color: BallColor, sensitivity: float = 0.5, fps: float = 30.0, hold_seconds: float = 4.0):
        self.det = BallDetector(color, DetectorSettings(sensitivity, proc_width=640, stabilize=True))
        self.max_history = int(fps * 8)  # cap per-track history so long sessions stay cheap
        self.fps = fps
        self.trk: MultiTracker | None = None
        self.idx = 0
        self.hold = int(hold_seconds * fps)
        self.last: TracerPath | None = None
        self.min_score = 3.0
        self.warmup = int(fps)          # let the clutter map learn the scene before showing tracers
        self._shown = None
        self._scores: dict[int, tuple[int, float]] = {}

    def process(self, frame, idx: int | None = None) -> TracerPath | None:
        """Push a frame; returns the tracer to draw on it (or None).

        idx: the frame's number in the camera stream. Pass it when frames may be skipped,
        so gaps count as elapsed time; by default frames are numbered consecutively."""
        if idx is not None:
            self.idx = idx
        if self.trk is None:
            self.trk = MultiTracker((frame.shape[1], frame.shape[0]), self.fps, clutter_decay=0.99)
            self.trk.max_active = 80
        results = self.det.push(self.idx, frame)
        self.idx += 1
        for r in results:
            self.trk.update(*r)
        if results:
            self.trk.prune_finished(self.idx - self.hold)
            for t in self.trk.active:
                if len(t) > self.max_history:
                    keep = slice(-self.max_history, None)
                    t.frames, t.xs, t.ys, t.sizes = t.frames[keep], t.xs[keep], t.ys[keep], t.sizes[keep]
            # cheap per-frame choice: score raw active tracks, smooth only the winner
            best, best_score = None, self.min_score
            scores = {}
            for t in self.trk.active:
                if len(t) >= 5:
                    cached = self._scores.get(t.tid)
                    if cached and cached[0] == len(t):
                        sc = cached[1]
                    else:
                        sc = score_track(split_stationary_start(t, self.trk.diag)[0], self.trk.diag, None,
                                         self.trk.clutter, self.fps)
                    scores[t.tid] = (len(t), sc)
                    if sc >= best_score and self.idx > self.warmup:
                        best, best_score = t, sc
            self._scores = scores
            if best is not None and (best.tid, len(best)) != self._shown:
                self._shown = (best.tid, len(best))
                flight, addr = split_stationary_start(best, self.trk.diag)
                frames, pts = np.asarray(flight.frames), flight.points()
                anchored = False
                if addr is not None:
                    frames, pts, anchored = _with_address(frames, pts, addr)
                self.last = smooth_path(frames, pts, (self.trk.w, self.trk.h), anchored=anchored)
        if self.last is not None and self.idx - self.last.end > self.hold:
            self.last = None
        return self.last
