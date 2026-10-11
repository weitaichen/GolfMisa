// Linking ball candidates into tracks and picking the ball's flight.
//
// Every candidate extends a track whose prediction gate it falls in, or starts a new
// one. The flight is the track that looks most like a golf ball: long, fast, smooth,
// small, consistently detected and not in a spot that keeps flickering. The winning
// fragment is then re-traced through the raw candidates to bridge dropouts.

import { median, outlierMask, polyfit, polyval, polyder0 } from './fit.js';

export class Track {
  constructor(tid) {
    this.tid = tid;
    this.frames = []; this.xs = []; this.ys = []; this.sizes = [];
  }
  get length() { return this.frames.length; }
  get lastFrame() { return this.frames[this.frames.length - 1]; }
  add(frame, c) { this.frames.push(frame); this.xs.push(c.x); this.ys.push(c.y); this.sizes.push(c.size); }
  velocity() {
    const n = this.length;
    if (n < 2) return [0, 0];
    const dt = Math.max(1, this.frames[n - 1] - this.frames[n - 2]);
    return [(this.xs[n - 1] - this.xs[n - 2]) / dt, (this.ys[n - 1] - this.ys[n - 2]) / dt];
  }
  speed() { return Math.hypot(...this.velocity()); }
  predict(frame, window = 6) {
    const n = this.length;
    if (n === 1) return [this.xs[0], this.ys[0]];
    const k = Math.min(n, window);
    const last = this.lastFrame;
    const t = this.frames.slice(-k).map((f) => f - last);
    const [cx, cy] = polyfit(t, [this.xs.slice(-k), this.ys.slice(-k)], k < 4 ? 1 : 2);
    return [polyval(cx, frame - last), polyval(cy, frame - last)];
  }
  slice(a, b) {
    const out = new Track(this.tid);
    out.frames = this.frames.slice(a, b); out.xs = this.xs.slice(a, b);
    out.ys = this.ys.slice(a, b); out.sizes = this.sizes.slice(a, b);
    return out;
  }
}

/** Counts candidates per coarse grid cell. Spots that keep firing (flickering edges,
 *  leaves, water) are clutter; a ball passes through once. */
export class ClutterMap {
  constructor([w, h], diag, decay = 1) {
    this.cell = Math.max(4, 0.012 * diag);
    this.gw = Math.floor(w / this.cell) + 1;
    this.gh = Math.floor(h / this.cell) + 1;
    this.heat = new Float32Array(this.gw * this.gh);
    this.decay = decay;
  }
  _cell(x, y) {
    const cx = Math.min(this.gw - 1, Math.max(0, Math.floor(x / this.cell)));
    const cy = Math.min(this.gh - 1, Math.max(0, Math.floor(y / this.cell)));
    return cy * this.gw + cx;
  }
  add(cands) {
    if (this.decay < 1) for (let i = 0; i < this.heat.length; i++) this.heat[i] *= this.decay;
    for (const c of cands) this.heat[this._cell(c.x, c.y)] += 1;
  }
  /** Median count of *other* detections in the cells a track passes through. */
  clutter(xs, ys) {
    const cells = xs.map((x, i) => this._cell(x, ys[i]));
    const own = new Map();
    for (const c of cells) own.set(c, (own.get(c) || 0) + 1);
    return median(cells.map((c) => Math.max(0, this.heat[c] - own.get(c))));
  }
}

export class MultiTracker {
  constructor(frameSize, fps = 30, clutterDecay = 1) {
    [this.w, this.h] = frameSize;
    this.diag = Math.hypot(this.w, this.h);
    this.fps = fps;
    this.maxMissed = Math.max(4, Math.round(fps * 0.1)); // longer dropouts: bridged by growTrack
    this.active = [];
    this.finished = [];
    this.nextId = 0;
    this.maxActive = 160;
    this.clutter = new ClutterMap(frameSize, this.diag, clutterDecay);
  }

  _gate(tr, gap) {
    const n = tr.length, d = this.diag;
    if (n === 1) return Math.min(0.2, 0.14 * gap) * d;
    const sp = tr.speed();
    if (n === 2) return Math.max(0.03 * d, 0.7 * sp) + 0.5 * sp * (gap - 1);
    return Math.max(0.015 * d, 0.45 * sp) + (0.3 * sp + 0.002 * d) * (gap - 1);
  }

  update(frame, cands) {
    this.clutter.add(cands);
    const usedC = new Set();
    if (cands.length) {
      const pairs = [];
      this.active.forEach((tr, ti) => {
        const gap = frame - tr.lastFrame;
        if (gap <= 0) return;
        const [px, py] = tr.predict(frame);
        const g = this._gate(tr, gap);
        const n = tr.length;
        const [vx, vy] = tr.velocity();
        const vn = Math.hypot(vx, vy);
        const pen = n >= 3 ? 1 : n === 2 ? 1.6 : 3;
        cands.forEach((c, ci) => {
          const d = Math.hypot(c.x - px, c.y - py);
          if (d >= g) return;
          if (n >= 3 && vn * gap > 0.004 * this.diag) {
            const sx = c.x - tr.xs[n - 1], sy = c.y - tr.ys[n - 1];
            const step = Math.hypot(sx, sy);
            const cos = (sx * vx + sy * vy) / (step * vn + 1e-9);
            // direction changes gradually (<60 deg per step) once a track is established
            if (cos <= (n >= 4 ? 0.5 : -0.1)) return;
            // a ball's image speed changes smoothly: it never stops dead or jumps
            if (vn * gap > 0.006 * this.diag && step <= 0.5 * vn * gap) return;
            if (step >= 2.2 * vn * gap + 0.01 * this.diag) return;
          }
          pairs.push([(d / g) * pen, -n, ti, ci]);
        });
      });
      pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      const usedT = new Set();
      for (const [, , ti, ci] of pairs) {
        if (usedT.has(ti) || usedC.has(ci)) continue;
        usedT.add(ti); usedC.add(ci);
        this.active[ti].add(frame, cands[ci]);
      }
    }
    let survivors = [];
    for (const tr of this.active) {
      const limit = tr.length >= 3 ? this.maxMissed : 2;
      if (frame - tr.lastFrame > limit) { if (tr.length >= 4) this.finished.push(tr); }
      else survivors.push(tr);
    }
    cands.forEach((c, ci) => {
      if (!usedC.has(ci) && c.area >= 2) { // 1-px blobs may only extend tracks
        const tr = new Track(this.nextId++);
        tr.add(frame, c);
        survivors.push(tr);
      }
    });
    if (survivors.length > this.maxActive) {
      survivors.sort((a, b) => b.length - a.length || b.lastFrame - a.lastFrame);
      survivors = survivors.slice(0, this.maxActive);
    }
    this.active = survivors;
  }

  allTracks(minLen = 4) { return [...this.finished, ...this.active].filter((t) => t.length >= minLen); }
  pruneFinished(beforeFrame) { this.finished = this.finished.filter((t) => t.lastFrame >= beforeFrame); }
}

/** Cheap pre-check: does the track move like a ball in flight at all? */
export function isMoving(tr, diag, fps = 30) {
  const n = tr.length;
  if (n < 4) return false;
  let path = 0;
  for (let i = 1; i < n; i++) path += Math.hypot(tr.xs[i] - tr.xs[i - 1], tr.ys[i] - tr.ys[i - 1]);
  const duration = tr.frames[n - 1] - tr.frames[0];
  return duration > 0 && path / diag / (duration / fps) >= 0.024;
}

export function scoreTrack(tr, diag, seed = null, clutter = null, fps = 30) {
  const n = tr.length;
  if (n < 4) return 0;
  const { xs, ys, frames: t } = tr;
  const sx = [], sy = [], sl = [];
  for (let i = 1; i < n; i++) {
    sx.push(xs[i] - xs[i - 1]); sy.push(ys[i] - ys[i - 1]); sl.push(Math.hypot(sx[i - 1], sy[i - 1]));
  }
  const path = sl.reduce((a, b) => a + b, 0);
  const duration = t[n - 1] - t[0];
  if (path < 1e-6 || duration <= 0) return 0;
  if (path / diag / (duration / fps) < 0.024) return 0; // < 2.4 % of the diagonal per second
  // turning: a ball's direction changes gradually from step to step; noise zig-zags
  const dirs = [];
  for (let i = 0; i < sl.length; i++) if (sl[i] > 0.5) dirs.push([sx[i] / sl[i], sy[i] / sl[i]]);
  let turn = 0;
  if (dirs.length >= 2) {
    let s = 0;
    for (let i = 1; i < dirs.length; i++) s += dirs[i][0] * dirs[i - 1][0] + dirs[i][1] * dirs[i - 1][1];
    turn = Math.min(1, Math.max(0, s / (dirs.length - 1)));
  }
  // jitter: distance from the line between neighbours, relative to step size
  const rel = [];
  for (let i = 1; i < n - 1; i++) {
    const w = (t[i] - t[i - 1]) / (t[i + 1] - t[i - 1]);
    const ix = xs[i - 1] + (xs[i + 1] - xs[i - 1]) * w, iy = ys[i - 1] + (ys[i + 1] - ys[i - 1]) * w;
    rel.push(Math.hypot(xs[i] - ix, ys[i] - iy) / (0.5 * (sl[i - 1] + sl[i]) + 1));
  }
  const jitter = median(rel);
  const coverage = n / (duration + 1);
  const size = median(tr.sizes);
  const sizeFactor = 1 / (1 + (size / 0.012) ** 2);
  let score = Math.sqrt(n) * (path / diag) * turn ** 2 / (1 + 4 * jitter) * coverage * sizeFactor * 4;
  if (clutter) score /= 1 + (clutter.clutter(xs, ys) / 3) ** 2;
  if (ys[Math.min(3, n - 1)] < ys[0]) score *= 1.3; // golf shots leave upwards in the image
  if (seed) {
    const [sf, sxx, syy] = seed;
    const d = Math.hypot(xs[0] - sxx, ys[0] - syy) / diag;
    if (t[0] < sf) score *= 0.3;
    else if (d < 0.12) score *= 1 + 3 * (1 - d / 0.12);
    else score *= 0.5;
  }
  return score;
}

// ------------------------------------------------------------------ re-growing
function predictK(F, X, Y, f, k = 8) {
  const last = F[F.length - 1];
  const t = F.slice(-k).map((v) => v - last);
  const [cx, cy] = polyfit(t, [X.slice(-k), Y.slice(-k)], t.length >= 4 ? 2 : 1);
  return [polyval(cx, f - last), polyval(cy, f - last), Math.hypot(polyder0(cx), polyder0(cy))];
}

/** Indices of candidates (Float64Array x,y pairs) consistent with the trajectory, best first. */
function gateOk(F, X, Y, cur, c, diag, tight = false) {
  const [px, py, v] = predictK(F, X, Y, cur);
  const gap = cur - F[F.length - 1];
  const tol = tight
    ? Math.max(0.008 * diag, 0.35 * v) + (0.1 * v + 0.001 * diag) * (gap - 1)
    : Math.max(0.012 * diag, 0.35 * v) + (0.15 * v + 0.002 * diag) * (gap - 1);
  const out = [];
  for (let i = 0; i < c.length / 2; i++) {
    const x = c[2 * i], y = c[2 * i + 1];
    const d = Math.hypot(x - px, y - py);
    if (d >= tol) continue;
    if (v * gap > 0.006 * diag) { // speed must stay plausible
      const step = Math.hypot(x - X[X.length - 1], y - Y[Y.length - 1]);
      if (!(step > 0.4 * v * gap && step < 2.2 * v * gap + 0.01 * diag)) continue;
    }
    out.push([d, i]);
  }
  return out.sort((a, b) => a[0] - b[0]).map((p) => p[1]);
}

/** After a dropout, accept a re-acquired candidate only if the following frames
 *  continue smoothly from it (noise near clouds / edges doesn't). */
function confirm(F, X, Y, cur, cx, cy, cands, diag, sign, fMin, fMax, look = 6, need = 3) {
  F = [...F, cur]; X = [...X, cx]; Y = [...Y, cy];
  let hits = 0;
  for (let t = cur + 1; t < cur + 1 + look; t++) {
    const real = sign * t;
    if (real < fMin || real > fMax) break;
    const cc = cands.get(real);
    if (!cc || !cc.length) continue;
    const idx = gateOk(F, X, Y, t, cc, diag, true);
    if (idx.length) {
      F.push(t); X.push(cc[2 * idx[0]]); Y.push(cc[2 * idx[0] + 1]);
      if (++hits >= need) return true;
    }
  }
  return false;
}

function growOneWay(fs, xs, ys, cands, diag, maxGap, sign, fMin, fMax) {
  let F = fs.map((f) => sign * f), X = [...xs], Y = [...ys];
  if (sign < 0) { F.reverse(); X.reverse(); Y.reverse(); }
  let cur = F[F.length - 1];
  while (cur - F[F.length - 1] <= maxGap) {
    cur += 1;
    const real = sign * cur;
    if (real < fMin || real > fMax) break;
    const c = cands.get(real);
    if (!c || !c.length) continue;
    for (const i of gateOk(F, X, Y, cur, c, diag)) {
      if (cur - F[F.length - 1] < 3 || confirm(F, X, Y, cur, c[2 * i], c[2 * i + 1], cands, diag, sign, fMin, fMax)) {
        F.push(cur); X.push(c[2 * i]); Y.push(c[2 * i + 1]);
        break;
      }
    }
  }
  if (sign < 0) { F.reverse(); X.reverse(); Y.reverse(); }
  return [F.map((f) => sign * f), X, Y];
}

/** Re-trace the flight through the raw detections starting from a confident fragment. */
export function growTrack(tr, cands, diag, maxGap, fMin, fMax, backwards = true) {
  if (tr.length < 4) return tr;
  let [fs, xs, ys] = growOneWay(tr.frames, tr.xs, tr.ys, cands, diag, maxGap, 1, fMin, fMax);
  if (backwards) [fs, xs, ys] = growOneWay(fs, xs, ys, cands, diag, Math.floor(maxGap / 3), -1, fMin, fMax);
  const out = new Track(tr.tid);
  const size = median(tr.sizes);
  const known = new Map(tr.frames.map((f, i) => [f, tr.sizes[i]]));
  out.frames = fs; out.xs = xs; out.ys = ys;
  out.sizes = fs.map((f) => known.get(f) ?? size);
  return out;
}

/** Drop points that don't fit the smooth flight (noise grabbed during a dropout). */
export function cleanTrack(tr, diag) {
  if (tr.length < 6) return tr;
  const keep = outlierMask(tr.frames, tr.xs, tr.ys, Math.max(2, 0.004 * diag));
  let out = new Track(tr.tid);
  keep.forEach((k, i) => {
    if (k) { out.frames.push(tr.frames[i]); out.xs.push(tr.xs[i]); out.ys.push(tr.ys[i]); out.sizes.push(tr.sizes[i]); }
  });
  // a flight starts with consecutive detections: drop stray points before a time gap
  while (out.length > 6) {
    let cut = -1;
    for (let i = 0; i < Math.min(4, out.length - 1); i++) if (out.frames[i + 1] - out.frames[i] > 3) { cut = i + 1; break; }
    if (cut < 0) break;
    out = out.slice(cut);
  }
  return out;
}

/** The ball sitting at address often starts a track (the club passing over it looks
 *  like motion). Cut it off; returns [flight, address or null]. */
export function splitStationaryStart(tr, diag) {
  const n = tr.length;
  const still = 0.006 * diag;
  let k = 0;
  while (k + 1 < n && Math.hypot(tr.xs[k + 1] - tr.xs[0], tr.ys[k + 1] - tr.ys[0]) < still) k++;
  if (k === 0 || n - k - 1 < 4) return [tr, null];
  let ax = 0, ay = 0;
  for (let i = 0; i <= k; i++) { ax += tr.xs[i]; ay += tr.ys[i]; }
  return [tr.slice(k + 1), [ax / (k + 1), ay / (k + 1)]];
}

/** Pick the ball flight. Returns [track, score, address]. */
export function pickBest(tracks, diag, fps, seed = null, clutter = null, cands = null, frameRange = [0, 1e9]) {
  let best = null, bestScore = 0, addr = null;
  for (const t of tracks) {
    let [flight, a] = splitStationaryStart(t, diag);
    if (!isMoving(flight, diag, fps)) continue; // stationary clutter: skip the costly cleaning
    flight = cleanTrack(flight, diag);
    const sc = scoreTrack(flight, diag, seed, clutter, fps);
    if (sc > bestScore) { best = flight; bestScore = sc; addr = a; }
  }
  if (!best) return [null, 0, null];
  if (cands) {
    const fMin = Math.max(frameRange[0], seed ? seed[0] : 0);
    best = growTrack(best, cands, diag, Math.max(10, Math.floor(fps * 0.5)), fMin, frameRange[1], addr === null);
    best = cleanTrack(best, diag);
  }
  return [best, Math.max(bestScore, scoreTrack(best, diag, seed, clutter, fps)), addr];
}
