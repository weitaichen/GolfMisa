"""Smooth tracer paths: robust curve fitting of tracked points, and manual tracers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TracerPath:
    frames: np.ndarray   # consecutive frame indices
    pts: np.ndarray      # (N, 2) ball position (video pixels) at each frame

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])

    def __len__(self):
        return len(self.frames)

    def until(self, frame: int) -> np.ndarray:
        """Points the ball has travelled through up to and including `frame`."""
        if frame < self.start:
            return self.pts[:0]
        return self.pts[: min(len(self.frames), frame - self.start + 1)]


def _polyval2(coef: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Evaluate a polynomial with (deg+1, 2) coefficients at u -> (len(u), 2)."""
    out = np.zeros((len(u), 2))
    for c in coef:
        out = out * u[:, None] + c
    return out


def _loess(u, p, u_out, w_rob, frac=0.12, kmin=6, kmax=12, leave_one_out=False):
    n = len(u)
    k = int(np.clip(n * frac, kmin, kmax))
    k = min(k, n)
    out = np.empty((len(u_out), 2))
    for i, uo in enumerate(u_out):
        d = np.abs(u - uo)
        h = np.partition(d, k - 1)[k - 1] * 1.05 + 1e-9
        w = np.clip(1 - (d / h) ** 3, 0, 1) ** 3 * w_rob
        if leave_one_out:
            w[i] = 0.0
        sel = w > 1e-6
        if sel.sum() < 3:
            out[i] = [np.interp(uo, u, p[:, 0]), np.interp(uo, u, p[:, 1])]
            continue
        deg = 2 if sel.sum() >= 5 else 1
        coef = np.polyfit(u[sel] - uo, p[sel], deg, w=np.sqrt(w[sel]))
        out[i] = coef[-1]
    return out


def _robust_fit(t, p, t_out, tol):
    tm = 0.5 * (t[0] + t[-1])
    ts = max(0.5 * (t[-1] - t[0]), 1.0)
    u, u_out = (t - tm) / ts, (t_out - tm) / ts
    n = len(t)
    w = np.ones(n)
    for deg in (2, 3, 4, 5):
        if n < deg + 3:
            break
        w = np.ones(n)
        for _ in range(4):
            coef = np.polyfit(u, p, deg, w=np.sqrt(w))
            r = np.hypot(*(p - _polyval2(coef, u)).T)
            s = max(np.median(r) * 1.4826, tol * 0.5)
            w = np.clip(1 - (r / (4.5 * s)) ** 2, 0, 1) ** 2
            if w.sum() < deg + 2:
                w = np.ones(n)
                break
        inl = w > 0.1
        if inl.mean() >= 0.8 and np.sqrt(np.mean(r[inl] ** 2)) <= tol and r[inl].max() <= 2.5 * tol:
            return _polyval2(coef, u_out)
    return _loess(u, p, u_out, np.maximum(w, 0.05))


def outlier_mask(t, p, tol) -> np.ndarray:
    """Boolean inlier mask of points against a robust local (LOESS) fit."""
    tm = 0.5 * (t[0] + t[-1])
    ts = max(0.5 * (t[-1] - t[0]), 1.0)
    u = (t - tm) / ts
    w = np.ones(len(t))
    keep = np.ones(len(t), bool)
    for _ in range(4):
        # wide window so a few garbage points at the end of a track are outvoted, and each
        # point judged by its neighbours only so a bad endpoint can't pull the fit to itself
        fit = _loess(u, p, u, np.maximum(w, 1e-3), frac=0.3, kmin=10, kmax=25, leave_one_out=True)
        r = np.hypot(*(p - fit).T)
        s = max(np.median(r) * 1.4826, tol * 0.5)
        w = np.clip(1 - (r / (4.5 * s)) ** 2, 0, 1) ** 2
        keep = r < max(3.5 * s, 1.5 * tol)
    return keep


def _extrapolate(frames, pts, extend, frame_size):
    k = min(10, len(frames))
    t = frames[-k:].astype(float) - frames[-1]
    coef = np.polyfit(t, pts[-k:], 2)
    w, h = frame_size
    new_f, new_p = [], []
    for i in range(1, extend + 1):
        q = _polyval2(coef, np.array([float(i)]))[0]
        if not (-0.02 * w <= q[0] <= 1.02 * w and -0.02 * h <= q[1] <= 1.02 * h):
            break
        new_f.append(frames[-1] + i)
        new_p.append(q)
    if not new_f:
        return frames, pts
    return np.concatenate([frames, new_f]), np.vstack([pts, new_p])


def smooth_path(frames, pts, frame_size, extend: int = 0, anchored: bool = False) -> TracerPath:
    """Fit a smooth curve through tracked ball positions and sample it every frame.

    anchored: the first point is the ball at address. Launch is an impulse no smooth
    curve fits, so the flight is fitted on its own and joined to the address by a line."""
    if anchored and len(frames) >= 3:
        flight = smooth_path(frames[1:], pts[1:], frame_size, extend)
        f0 = int(frames[0])
        n = flight.start - f0
        if n <= 0:
            return flight
        u = (np.arange(n) / n)[:, None]
        lead = np.asarray(pts[0], float) * (1 - u) + flight.pts[0] * u
        return TracerPath(np.concatenate([np.arange(f0, flight.start), flight.frames]),
                          np.vstack([lead, flight.pts]))
    frames = np.asarray(frames, float)
    pts = np.asarray(pts, float)
    order = np.argsort(frames, kind="stable")
    frames, pts = frames[order], pts[order]
    out_f = np.arange(int(frames[0]), int(frames[-1]) + 1)
    diag = float(np.hypot(*frame_size))
    if len(frames) < 5:
        out = np.column_stack([np.interp(out_f, frames, pts[:, 0]), np.interp(out_f, frames, pts[:, 1])])
    else:
        out = _robust_fit(frames, pts, out_f.astype(float), tol=max(2.0, 0.0025 * diag))
    if extend > 0 and len(out_f) >= 5:
        out_f, out = _extrapolate(out_f, out, extend, frame_size)
    return TracerPath(out_f.astype(int), out)


def manual_path(start, apex, end, launch_frame: int, n_frames: int) -> TracerPath:
    """Tracer through three clicked points (ball, apex, landing) as a quadratic Bezier,
    with the ball slowing down along the curve like a real shot."""
    s, a, e = (np.asarray(v, float) for v in (start, apex, end))
    c = 2 * a - 0.5 * (s + e)             # control point so the curve passes through the apex
    n_frames = max(2, int(n_frames))
    u = np.linspace(0, 1, n_frames)
    tau = 1 - (1 - u) ** 1.35
    pts = ((1 - tau) ** 2)[:, None] * s + (2 * (1 - tau) * tau)[:, None] * c + (tau ** 2)[:, None] * e
    return TracerPath(np.arange(launch_frame, launch_frame + n_frames), pts)
