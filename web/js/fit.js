// Curve fitting: weighted polynomial least squares, robust LOESS, outlier removal,
// tracer path smoothing and the manual (3-tap) tracer.

export const median = (arr) => {
  if (!arr.length) return 0;
  const s = Float64Array.from(arr).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
};

/** Weighted least-squares polynomial fit of several y series sharing x.
 *  Coefficients ascending (c0 + c1 x + ...). `w` are effective weights (per squared
 *  residual), or null for unweighted. Returns one coefficient array per series. */
export function polyfit(x, ys, deg, w = null) {
  const m = deg + 1;
  const A = Array.from({ length: m }, () => new Float64Array(m));
  const B = ys.map(() => new Float64Array(m));
  const pw = new Float64Array(2 * m);
  for (let i = 0; i < x.length; i++) {
    const wi = w ? w[i] : 1;
    if (!(wi > 0)) continue;
    let p = 1;
    for (let k = 0; k < 2 * m - 1; k++) { pw[k] = p; p *= x[i]; }
    for (let r = 0; r < m; r++) {
      for (let c = 0; c < m; c++) A[r][c] += wi * pw[r + c];
      for (let s = 0; s < ys.length; s++) B[s][r] += wi * pw[r] * ys[s][i];
    }
  }
  return B.map((b) => solve(A.map((row) => Float64Array.from(row)), Float64Array.from(b)));
}

function solve(A, b) {
  const n = b.length;
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]];
    [b[c], b[piv]] = [b[piv], b[c]];
    const d = A[c][c] || 1e-12;
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / d;
      if (!f) continue;
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  const out = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= A[r][k] * out[k];
    out[r] = s / (A[r][r] || 1e-12);
  }
  return out;
}

export const polyval = (c, x) => {
  let y = 0;
  for (let k = c.length - 1; k >= 0; k--) y = y * x + c[k];
  return y;
};

export const polyder0 = (c) => (c.length > 1 ? c[1] : 0); // derivative at x = 0

function interp(x, xs, ys) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let i = 1;
  while (xs[i] < x) i++;
  const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1] || 1);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
}

/** Locally weighted quadratic regression of (px, py) over u, evaluated at uOut. */
function loess(u, px, py, uOut, wRob, { frac = 0.12, kmin = 6, kmax = 12, leaveOneOut = false } = {}) {
  const n = u.length;
  const k = Math.min(n, Math.floor(Math.min(Math.max(n * frac, kmin), kmax)));
  const ox = new Float64Array(uOut.length);
  const oy = new Float64Array(uOut.length);
  const d = new Float64Array(n);
  for (let i = 0; i < uOut.length; i++) {
    const uo = uOut[i];
    for (let j = 0; j < n; j++) d[j] = Math.abs(u[j] - uo);
    const h = Float64Array.from(d).sort()[k - 1] * 1.05 + 1e-9;
    const su = [], sx = [], sy = [], sw = [];
    for (let j = 0; j < n; j++) {
      if (leaveOneOut && j === i) continue;
      const t = Math.min(1, d[j] / h);
      const w = (1 - t * t * t) ** 3 * wRob[j];
      if (w > 1e-6) { su.push(u[j] - uo); sx.push(px[j]); sy.push(py[j]); sw.push(w); }
    }
    if (su.length < 3) { ox[i] = interp(uo, u, px); oy[i] = interp(uo, u, py); continue; }
    const [cx, cy] = polyfit(su, [sx, sy], su.length >= 5 ? 2 : 1, sw);
    ox[i] = cx[0];
    oy[i] = cy[0];
  }
  return [ox, oy];
}

function normalize(t) {
  const tm = 0.5 * (t[0] + t[t.length - 1]);
  const ts = Math.max(0.5 * (t[t.length - 1] - t[0]), 1);
  return [tm, ts];
}

function robustFit(t, px, py, tOut, tol) {
  const [tm, ts] = normalize(t);
  const u = t.map((v) => (v - tm) / ts);
  const uOut = tOut.map((v) => (v - tm) / ts);
  const n = t.length;
  let w = new Float64Array(n).fill(1);
  for (const deg of [2, 3, 4, 5]) {
    if (n < deg + 3) break;
    w = new Float64Array(n).fill(1);
    let cx, cy, r;
    for (let it = 0; it < 4; it++) {
      [cx, cy] = polyfit(u, [px, py], deg, w);
      r = u.map((uu, i) => Math.hypot(px[i] - polyval(cx, uu), py[i] - polyval(cy, uu)));
      const s = Math.max(median(r) * 1.4826, tol * 0.5);
      w = Float64Array.from(r, (ri) => Math.max(0, 1 - (ri / (4.5 * s)) ** 2) ** 2);
      if (w.reduce((a, b) => a + b, 0) < deg + 2) { w = new Float64Array(n).fill(1); break; }
    }
    const inl = r.filter((_, i) => w[i] > 0.1);
    if (inl.length / n >= 0.8 && Math.sqrt(inl.reduce((a, v) => a + v * v, 0) / inl.length) <= tol
        && Math.max(...inl) <= 2.5 * tol) {
      return [Float64Array.from(uOut, (v) => polyval(cx, v)), Float64Array.from(uOut, (v) => polyval(cy, v))];
    }
  }
  return loess(u, px, py, uOut, w.map((v) => Math.max(v, 0.05)));
}

/** Inlier mask of points against a robust local fit. Each point is judged by its
 *  neighbours only (leave-one-out) so a bad endpoint can't pull the fit to itself. */
export function outlierMask(t, px, py, tol) {
  const [tm, ts] = normalize(t);
  const u = t.map((v) => (v - tm) / ts);
  const n = t.length;
  let w = new Float64Array(n).fill(1);
  let keep = new Array(n).fill(true);
  for (let it = 0; it < 4; it++) {
    const [fx, fy] = loess(u, px, py, u, w.map((v) => Math.max(v, 1e-3)),
      { frac: 0.3, kmin: 10, kmax: 25, leaveOneOut: true });
    const r = u.map((_, i) => Math.hypot(px[i] - fx[i], py[i] - fy[i]));
    const s = Math.max(median(r) * 1.4826, tol * 0.5);
    w = Float64Array.from(r, (ri) => Math.max(0, 1 - (ri / (4.5 * s)) ** 2) ** 2);
    keep = r.map((ri) => ri < Math.max(3.5 * s, 1.5 * tol));
  }
  return keep;
}

/** A tracer: ball position for every frame from `start` to `end`. */
export class TracerPath {
  constructor(frames, xs, ys) {
    this.frames = Int32Array.from(frames);
    this.xs = Float64Array.from(xs);
    this.ys = Float64Array.from(ys);
  }
  get start() { return this.frames[0]; }
  get end() { return this.frames[this.frames.length - 1]; }
  get length() { return this.frames.length; }
  /** Number of points travelled through up to and including `frame`. */
  until(frame) {
    if (frame < this.start) return 0;
    return Math.min(this.frames.length, frame - this.start + 1);
  }
  toJSON() { return { frames: Array.from(this.frames), xs: Array.from(this.xs), ys: Array.from(this.ys) }; }
  static fromJSON(o) { return o ? new TracerPath(o.frames, o.xs, o.ys) : null; }
}

function extrapolate(frames, xs, ys, extend, [w, h]) {
  const k = Math.min(10, frames.length);
  const last = frames[frames.length - 1];
  const t = frames.slice(-k).map((f) => f - last);
  const [cx, cy] = polyfit(t, [xs.slice(-k), ys.slice(-k)], 2);
  const F = [...frames], X = [...xs], Y = [...ys];
  for (let i = 1; i <= extend; i++) {
    const qx = polyval(cx, i), qy = polyval(cy, i);
    if (!(qx >= -0.02 * w && qx <= 1.02 * w && qy >= -0.02 * h && qy <= 1.02 * h)) break;
    F.push(last + i); X.push(qx); Y.push(qy);
  }
  return [F, X, Y];
}

/** Fit a smooth curve through tracked positions, sampled every frame.
 *  anchored: the first point is the ball at address; launch is an impulse no smooth
 *  curve fits, so the flight is fitted alone and joined to the address by a line. */
export function smoothPath(frames, xs, ys, frameSize, { extend = 0, anchored = false } = {}) {
  if (anchored && frames.length >= 3) {
    const flight = smoothPath(frames.slice(1), xs.slice(1), ys.slice(1), frameSize, { extend });
    const f0 = frames[0];
    const n = flight.start - f0;
    if (n <= 0) return flight;
    const F = [], X = [], Y = [];
    for (let i = 0; i < n; i++) {
      const u = i / n;
      F.push(f0 + i); X.push(xs[0] * (1 - u) + flight.xs[0] * u); Y.push(ys[0] * (1 - u) + flight.ys[0] * u);
    }
    return new TracerPath([...F, ...flight.frames], [...X, ...flight.xs], [...Y, ...flight.ys]);
  }
  const order = frames.map((_, i) => i).sort((a, b) => frames[a] - frames[b]);
  const t = order.map((i) => frames[i]);
  const px = order.map((i) => xs[i]);
  const py = order.map((i) => ys[i]);
  const outF = [];
  for (let f = t[0]; f <= t[t.length - 1]; f++) outF.push(f);
  const diag = Math.hypot(frameSize[0], frameSize[1]);
  let ox, oy;
  if (t.length < 5) {
    ox = outF.map((f) => interp(f, t, px));
    oy = outF.map((f) => interp(f, t, py));
  } else {
    [ox, oy] = robustFit(t, px, py, outF, Math.max(2, 0.0025 * diag));
  }
  let F = outF, X = Array.from(ox), Y = Array.from(oy);
  if (extend > 0 && F.length >= 5) [F, X, Y] = extrapolate(F, X, Y, extend, frameSize);
  return new TracerPath(F, X, Y);
}

/** Tracer through three taps (ball, apex, landing): quadratic Bezier through the apex,
 *  with the ball slowing down along it like a real shot. */
export function manualPath(start, apex, end, launchFrame, nFrames) {
  const cx = 2 * apex[0] - 0.5 * (start[0] + end[0]);
  const cy = 2 * apex[1] - 0.5 * (start[1] + end[1]);
  const n = Math.max(2, Math.round(nFrames));
  const F = [], X = [], Y = [];
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    const t = 1 - (1 - u) ** 1.35;
    const a = (1 - t) ** 2, b = 2 * (1 - t) * t, c = t * t;
    F.push(launchFrame + i);
    X.push(a * start[0] + b * cx + c * end[0]);
    Y.push(a * start[1] + b * cy + c * end[1]);
  }
  return new TracerPath(F, X, Y);
}
