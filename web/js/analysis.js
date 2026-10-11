// Whole-clip shot analysis (streaming frames in) and live camera tracking.

import { BallDetector } from './detector.js';
import { smoothPath, TracerPath } from './fit.js';
import { MultiTracker, pickBest, scoreTrack, splitStationaryStart } from './tracker.js';

export const MIN_SCORE = 2.5; // real flights score ~5-20; noise / wrong-colour tracks < 2.5

/** Processing size: long side `procLong` px, even dimensions. */
export function procSize(w, h, procLong = 640) {
  const s = Math.min(1, procLong / Math.max(w, h));
  return [Math.max(2, Math.round((w * s) / 2) * 2), Math.max(2, Math.round((h * s) / 2) * 2)];
}

function withAddress(frames, xs, ys, addr, minFrame = 0) {
  frames = [...frames]; xs = [...xs]; ys = [...ys];
  const ext = Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const still = 0.006 * ext + 6;
  while (frames.length > 4 && Math.hypot(xs[0] - addr[0], ys[0] - addr[1]) < still) {
    frames.shift(); xs.shift(); ys.shift(); // impact frame: ball hasn't really left yet
  }
  const sp = Math.hypot(xs[1] - xs[0], ys[1] - ys[0]) / Math.max(1, frames[1] - frames[0]);
  const d = Math.hypot(xs[0] - addr[0], ys[0] - addr[1]);
  const lead = Math.min(6, Math.max(1, Math.round(d / Math.max(sp, 1))));
  const launch = Math.max(minFrame, frames[0] - lead);
  if (launch >= frames[0]) return [frames, xs, ys, false];
  return [[launch, ...frames], [addr[0], ...xs], [addr[1], ...ys], true];
}

export class ShotResult {
  constructor(o) { Object.assign(this, o); }
  get found() { return !!this.frames && this.frames.length >= 2; }
  get launchFrame() { return this.frames[0]; }
  get lastFrame() { return this.frames[this.frames.length - 1]; }
  get hangTime() { return (this.lastFrame - this.launchFrame) / this.fps; }
  get confidence() { return this.score > 8 ? 'High' : this.score > 4 ? 'Medium' : 'Low'; }
  buildPath(extend = 0) {
    if (!this.found) return null;
    return smoothPath(this.frames, this.xs, this.ys, this.frameSize, { extend, anchored: !!this.address });
  }
}

/** Ball at address just before launch: a stationary ball-coloured blob behind the
 *  first tracked position, which is gone once the ball is in the air. */
function findAddressBall(blobs, frames, xs, ys, frameSize) {
  if (frames.length < 3) return null;
  const at = (target, before) => { // stored frame nearest to target (frames may be skipped)
    let best = null;
    for (const f of blobs.keys()) if ((before ? f <= target : f >= target) && (best === null || Math.abs(f - target) < Math.abs(best - target))) best = f;
    return best === null ? [] : blobs.get(best);
  };
  const diag = Math.hypot(...frameSize);
  const pre = at(frames[0] - 4, true);
  if (!pre) return null;
  let bx = xs[0] - xs[Math.min(2, xs.length - 1)], by = ys[0] - ys[Math.min(2, ys.length - 1)];
  const nb = Math.hypot(bx, by);
  if (nb < 1e-6) return null;
  bx /= nb; by /= nb;
  let best = null, bestCost = Infinity;
  for (const [x, y] of pre) {
    const vx = x - xs[0], vy = y - ys[0];
    const along = vx * bx + vy * by;
    const perp = Math.abs(vx * by - vy * bx);
    if (along > -0.02 * diag && along < 0.25 * diag && perp < 0.05 * diag + 0.15 * Math.max(along, 0)) {
      const cost = perp + 0.1 * along;
      if (cost < bestCost) { best = [x, y]; bestCost = cost; }
    }
  }
  if (!best) return null;
  const post = at(frames[Math.min(frames.length - 1, 4)], false);
  if (post && post.some(([x, y]) => Math.hypot(x - best[0], y - best[1]) < 0.01 * diag)) return null;
  return best;
}

/** Feed every frame of a clip (RGBA at procSize), then call finish(). */
export class Analyzer {
  constructor({ color, sensitivity = 0.5, fps = 30, frameSize, procSize: ps, seed = null, stabilize = true }) {
    this.fps = fps;
    this.frameSize = frameSize;
    this.seed = seed;
    this.det = new BallDetector(color, ps, frameSize, { sensitivity, stabilize, blobs: true });
    this.trk = new MultiTracker(frameSize, fps);
    this.cands = new Map();
    this.blobs = new Map(); // frame -> ball-shaped colour blobs (a few numbers per frame)
    this.first = null;
    this.last = null;
  }

  _handle(results) {
    for (const [fi, cands, blobs] of results) {
      this.trk.update(fi, cands);
      if (cands.length) this.cands.set(fi, Float64Array.from(cands.flatMap((c) => [c.x, c.y])));
      if (blobs) this.blobs.set(fi, blobs);
    }
  }

  push(idx, rgba) {
    if (this.last !== null && idx <= this.last) return; // duplicate / out-of-order frame
    if (this.first === null) this.first = idx;
    this.last = idx;
    this._handle(this.det.push(idx, rgba));
  }

  /** Candidates per frame (for the debug overlay). */
  candidates() { return this.cands; }

  finish() {
    this._handle(this.det.flush());
    const base = { fps: this.fps, frameSize: this.frameSize, frames: null, xs: null, ys: null, address: null, score: 0 };
    if (this.first === null) throw new Error('No frames could be read from the video.');
    const [best, score, stillAddr] = pickBest(this.trk.allTracks(), this.trk.diag, this.fps, this.seed,
      this.trk.clutter, this.cands, [this.first, this.last + 1]);
    if (!best || score < MIN_SCORE) return new ShotResult(base);
    let frames = best.frames, xs = best.xs, ys = best.ys;
    let addr, minFrame = 0;
    if (this.seed) { addr = [this.seed[1], this.seed[2]]; minFrame = this.seed[0]; }
    else addr = stillAddr || findAddressBall(this.blobs, frames, xs, ys, this.frameSize);
    if (addr) {
      let prepended;
      [frames, xs, ys, prepended] = withAddress(frames, xs, ys, addr, minFrame);
      if (!prepended) addr = null; // no room before the flight: don't anchor to it
    }
    return new ShotResult({ ...base, frames, xs, ys, address: addr, score });
  }
}

/** Frame-by-frame tracking for the live camera view. */
export class LiveTracker {
  constructor({ color, sensitivity = 0.5, fps = 30, frameSize, procSize: ps, holdSeconds = 4 }) {
    this.det = new BallDetector(color, ps, frameSize, { sensitivity, stabilize: true });
    this.fps = fps;
    this.trk = new MultiTracker(frameSize, fps, 0.99);
    this.trk.maxActive = 80;
    this.hold = Math.round(holdSeconds * fps);
    this.warmup = Math.round(fps); // let the clutter map learn the scene first
    this.minScore = 3;
    this.maxHistory = Math.round(fps * 8);
    this.idx = 0;
    this.last = null;
    this.shown = null;
    this.scores = new Map();
  }

  /** Push frame number `idx` (camera frame counter; gaps = skipped frames). */
  process(idx, rgba) {
    this.idx = idx;
    const results = this.det.push(idx, rgba);
    for (const [fi, cands] of results) this.trk.update(fi, cands);
    if (results.length) {
      this.trk.pruneFinished(idx - this.hold);
      let best = null, bestScore = this.minScore;
      const scores = new Map();
      for (const t of this.trk.active) {
        if (t.length > this.maxHistory) {
          const k = t.length - this.maxHistory;
          t.frames.splice(0, k); t.xs.splice(0, k); t.ys.splice(0, k); t.sizes.splice(0, k);
        }
        if (t.length < 5) continue;
        // keyed by the newest frame: a track at its history cap keeps the same length
        const cached = this.scores.get(t.tid);
        const sc = cached && cached[0] === t.lastFrame ? cached[1]
          : scoreTrack(splitStationaryStart(t, this.trk.diag)[0], this.trk.diag, null, this.trk.clutter, this.fps);
        scores.set(t.tid, [t.lastFrame, sc]);
        if (sc >= bestScore && idx > this.warmup) { best = t; bestScore = sc; }
      }
      this.scores = scores;
      const key = best && `${best.tid}:${best.lastFrame}`;
      if (best && key !== this.shown) {
        this.shown = key;
        const [flight, addr] = splitStationaryStart(best, this.trk.diag);
        let f = flight.frames, x = flight.xs, y = flight.ys, anchored = false;
        if (addr) [f, x, y, anchored] = withAddress(f, x, y, addr);
        this.last = smoothPath(f, x, y, [this.trk.w, this.trk.h], { anchored });
      }
    }
    if (this.last && idx - this.last.end > this.hold) this.last = null;
    return this.last;
  }
}

export { TracerPath };
