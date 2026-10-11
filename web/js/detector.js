// Per-frame ball candidates: "moving AND ball-coloured".
//
// Motion at frame t = differs from frames t-k AND t+k (k = 1, 2, 4, so a far-away,
// slow ball is still seen), compared against the 3x3 min/max of the other frame so a
// 1-px edge shift from camera shake isn't motion. Hand-held shake is estimated per
// frame and compensated with whole-pixel shifts. Frames arrive as RGBA at processing
// size; candidate positions are returned in original video pixels.
//
// The loops below are written flat (no per-pixel function calls or allocations)
// because this is the hot path on a phone.

import { buildMaskLUT } from './colors.js';

const BASELINES = [1, 2, 4];
const HALF = 4;

/** [1 2 1] x [1 2 1] / 16 per channel; RGBA in -> RGB out. */
function blur3(src, w, h, tmp, out) {
  for (let y = 0; y < h; y++) {
    const r4 = y * w * 4, r3 = y * w * 3;
    for (let x = 0; x < w; x++) {
      const i = r4 + x * 4, o = r3 + x * 3;
      const il = x > 0 ? i - 4 : i, ir = x < w - 1 ? i + 4 : i;
      tmp[o] = src[il] + 2 * src[i] + src[ir];
      tmp[o + 1] = src[il + 1] + 2 * src[i + 1] + src[ir + 1];
      tmp[o + 2] = src[il + 2] + 2 * src[i + 2] + src[ir + 2];
    }
  }
  const row = w * 3;
  for (let y = 0; y < h; y++) {
    const o0 = y * row, u0 = y > 0 ? o0 - row : o0, d0 = y < h - 1 ? o0 + row : o0;
    for (let k = 0; k < row; k++) out[o0 + k] = (tmp[u0 + k] + 2 * tmp[o0 + k] + tmp[d0 + k] + 8) >> 4;
  }
}

/** Separable 3x3 min (lo) and max (hi) of an RGB image. */
function minMax3(img, w, h, tlo, thi, lo, hi) {
  const row = w * 3;
  for (let y = 0; y < h; y++) {
    const o0 = y * row;
    for (let c = 0; c < 3; c++) { // left / right edge pixels
      const a = img[o0 + c], b = img[o0 + 3 + c];
      tlo[o0 + c] = a < b ? a : b; thi[o0 + c] = a > b ? a : b;
      const e = o0 + row - 3 + c, f = img[e], g = img[e - 3];
      tlo[e] = f < g ? f : g; thi[e] = f > g ? f : g;
    }
    for (let k = o0 + 3, end = o0 + row - 3; k < end; k++) {
      const a = img[k - 3], b = img[k], d = img[k + 3];
      let m = a < b ? a : b; tlo[k] = m < d ? m : d;
      m = a > b ? a : b; thi[k] = m > d ? m : d;
    }
  }
  const n = w * h * 3;
  for (let k = 0; k < n; k++) {
    const u = k >= row ? k - row : k, d = k + row < n ? k + row : k;
    let a = tlo[u], b = tlo[k], e = tlo[d];
    let m = a < b ? a : b; lo[k] = m < e ? m : e;
    a = thi[u]; b = thi[k]; e = thi[d];
    m = a > b ? a : b; hi[k] = m > e ? m : e;
  }
}

function halfGray(img, w, h, g) {
  const gw = w >> 1, gh = h >> 1, row = w * 3;
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const o = 2 * y * row + 6 * x, o2 = o + row;
      g[y * gw + x] = (img[o] * 77 + img[o + 1] * 150 + img[o + 2] * 29 + img[o + 3] * 77 + img[o + 4] * 150
        + img[o + 5] * 29 + img[o2] * 77 + img[o2 + 1] * 150 + img[o2 + 2] * 29 + img[o2 + 3] * 77
        + img[o2 + 4] * 150 + img[o2 + 5] * 29) >> 10;
    }
  }
}

function sad(prev, cur, gw, gh, dx, dy, m, step) {
  let s = 0;
  for (let y = m; y < gh - m; y += step) {
    const rc = y * gw, rp = (y - dy) * gw - dx;
    for (let x = m; x < gw - m; x += step) {
      const d = cur[rc + x] - prev[rp + x];
      s += d < 0 ? -d : d;
    }
  }
  return s;
}

function quarter(g, gw, gh, out) {
  const qw = gw >> 1, qh = gh >> 1;
  for (let y = 0; y < qh; y++)
    for (let x = 0; x < qw; x++) {
      const o = 2 * y * gw + 2 * x;
      out[y * qw + x] = (g[o] + g[o + 1] + g[o + gw] + g[o + gw + 1]) >> 2;
    }
  return out;
}

/** Whole-pixel translation of `cur` content relative to `prev` (half-size gray), in
 *  processing pixels: coarse search at quarter size, refined at half size. Zero
 *  unless the shift clearly explains the frame difference better than no shift. */
function estimateShift(prev, cur, gw, gh, qPrev, qCur) {
  const qw = gw >> 1, qh = gh >> 1;
  let bx = 0, by = 0, best = Infinity;
  for (let dy = -2; dy <= 2; dy++)
    for (let dx = -2; dx <= 2; dx++) {
      const v = sad(qPrev, qCur, qw, qh, dx, dy, 4, 1);
      if (v < best) { best = v; bx = dx; by = dy; }
    }
  const zero = sad(prev, cur, gw, gh, 0, 0, 7, 2);
  let fx = 0, fy = 0;
  best = Infinity;
  for (let dy = 2 * by - 1; dy <= 2 * by + 1; dy++)
    for (let dx = 2 * bx - 1; dx <= 2 * bx + 1; dx++) {
      const v = sad(prev, cur, gw, gh, dx, dy, 7, 2);
      if (v < best) { best = v; fx = dx; fy = dy; }
    }
  return best < 0.92 * zero ? [fx * 2, fy * 2] : [0, 0];
}

/** 8-connected components of a 0/1 mask: [{area, x0, y0, x1, y1, cx, cy}]. */
export function components(mask, w, h, labels = new Int32Array(w * h), maxComponents = 4096) {
  labels.fill(0);
  const parent = [0];
  const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  const union = (a, b) => {
    a = find(a); b = find(b);
    if (a === b) return a;
    if (a < b) { parent[b] = a; return a; }
    parent[a] = b; return b;
  };
  let next = 1;
  for (let y = 0; y < h; y++) {
    const r = y * w;
    for (let x = 0; x < w; x++) {
      const i = r + x;
      if (!mask[i]) continue;
      let lab = x > 0 ? labels[i - 1] : 0;
      if (y > 0) {
        const up = i - w;
        if (x > 0 && labels[up - 1]) lab = lab ? union(lab, labels[up - 1]) : labels[up - 1];
        if (labels[up]) lab = lab ? union(lab, labels[up]) : labels[up];
        if (x < w - 1 && labels[up + 1]) lab = lab ? union(lab, labels[up + 1]) : labels[up + 1];
      }
      if (!lab) { lab = next++; parent.push(lab); }
      labels[i] = lab;
    }
  }
  const stats = new Map();
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const l = labels[y * w + x];
      if (!l) continue;
      const root = find(l);
      let s = stats.get(root);
      if (!s) {
        if (stats.size >= maxComponents) continue;
        s = { area: 0, x0: x, y0: y, x1: x, y1: y, sx: 0, sy: 0 };
        stats.set(root, s);
      }
      s.area++; s.sx += x; s.sy += y;
      if (x < s.x0) s.x0 = x;
      if (x > s.x1) s.x1 = x;
      if (y < s.y0) s.y0 = y;
      if (y > s.y1) s.y1 = y;
    }
  }
  return [...stats.values()].map((s) => ({ ...s, cx: s.sx / s.area, cy: s.sy / s.area }));
}

export class BallDetector {
  /** procSize: [w, h] of the RGBA frames pushed; origSize: video size for output coords.
   *  blobs: also report each frame's ball-shaped, ball-coloured blobs (moving or not),
   *  used afterwards to find the ball sitting at address. */
  constructor(color, procSize, origSize, { sensitivity = 0.5, stabilize = true, maxCandidates = 30, blobs = false } = {}) {
    [this.w, this.h] = procSize;
    const n = this.w * this.h;
    this.scale = this.w / origSize[0];
    this.diag = Math.hypot(this.w, this.h);
    this.maxArea = (0.035 * Math.max(this.w, this.h)) ** 2;
    this.blobArea = (0.03 * Math.max(this.w, this.h)) ** 2;
    this.thr = Math.round(36 + (8 - 36) * Math.min(1, Math.max(0, sensitivity)));
    this.lut = buildMaskLUT(color, sensitivity);
    this.stabilize = stabilize;
    this.maxCandidates = maxCandidates;
    this.wantBlobs = blobs;
    this.buf = [];
    this.free = [];
    this.primed = false;
    this.prev = null;
    // scratch buffers
    this.tmp16 = new Uint16Array(n * 3);
    this.tlo = new Uint8Array(n * 3);
    this.thi = new Uint8Array(n * 3);
    this.cmask = new Uint8Array(n);
    this.near = new Uint8Array(n);
    this.motion = new Uint8Array(n);
    this.passes = new Uint8Array(n);
    this.comb = new Uint8Array(n);
    this.labels = new Int32Array(n);
    this.half = new Uint8Array((this.w >> 1) * (this.h >> 1));
    this.halfLabels = new Int32Array((this.w >> 1) * (this.h >> 1));
    this.xmap = new Int32Array(this.w);
    this.ymap = new Int32Array(this.h);
  }

  _prep(idx, rgba) {
    const { w, h } = this;
    const n = w * h;
    const f = this.free.pop() || {
      img: new Uint8Array(n * 3), lo: new Uint8Array(n * 3), hi: new Uint8Array(n * 3),
      gray: new Uint8Array((w >> 1) * (h >> 1)),
      q: new Uint8Array((w >> 2) * (h >> 2)),
    };
    blur3(rgba, w, h, this.tmp16, f.img);
    minMax3(f.img, w, h, this.tlo, this.thi, f.lo, f.hi);
    f.idx = idx;
    f.pos = [0, 0];
    if (this.stabilize) {
      halfGray(f.img, w, h, f.gray);
      quarter(f.gray, w >> 1, h >> 1, f.q);
      if (this.prev) {
        const d = estimateShift(this.prev.gray, f.gray, w >> 1, h >> 1, this.prev.q, f.q);
        f.pos = [this.prev.pos[0] + d[0], this.prev.pos[1] + d[1]];
      }
    }
    this.prev = f;
    return f;
  }

  _shiftOut() {
    const old = this.buf.shift();
    if (old !== this.prev) this.free.push(old);
  }

  /** Add frame `idx`; returns [[frameIndex, candidates], ...] for completed frames. */
  push(idx, rgba) {
    this.buf.push(this._prep(idx, rgba));
    if (this.buf.length > 2 * HALF + 1) this._shiftOut();
    if (this.buf.length < 2 * HALF + 1) return [];
    if (!this.primed) {
      this.primed = true;
      const out = [];
      for (let c = 0; c <= HALF; c++) out.push(this._detectAt(c));
      return out;
    }
    return [this._detectAt(HALF)];
  }

  /** Detections for the frames still buffered at the end of the clip. */
  flush() {
    const out = [];
    for (let c = this.primed ? HALF + 1 : 0; c < this.buf.length; c++) out.push(this._detectAt(c));
    this.buf = [];
    this.free = [];
    this.prev = null;
    this.primed = false;
    return out;
  }

  /** Ball-shaped blobs of the colour mask, found on a half-size copy (cheap; the ball at
   *  address only needs to be located roughly). Returns [[x, y], ...] in video pixels. */
  _blobs() {
    const { w, h, cmask, half } = this;
    const hw = w >> 1, hh = h >> 1;
    for (let y = 0; y < hh; y++) {
      const r0 = 2 * y * w, r1 = r0 + w;
      for (let x = 0; x < hw; x++) {
        const c = 2 * x;
        half[y * hw + x] = cmask[r0 + c] | cmask[r0 + c + 1] | cmask[r1 + c] | cmask[r1 + c + 1];
      }
    }
    const inv = 2 / this.scale;
    const maxArea = this.blobArea / 4;
    const out = [];
    for (const s of components(half, hw, hh, this.halfLabels, 512)) {
      const bw = s.x1 - s.x0 + 1, bh = s.y1 - s.y0 + 1;
      if (s.area <= maxArea && Math.max(bw, bh) / Math.max(1, Math.min(bw, bh)) < 2.2)
        out.push([(s.cx + 0.25) * inv, (s.cy + 0.25) * inv]);
    }
    return out;
  }

  _detectAt(c) {
    const { w, h, thr, buf, lut, cmask, near, motion, comb, xmap, ymap, passes } = this;
    const n = w * h;
    const cur = buf[c];
    const img = cur.img;

    // colour mask, and the pixels within 1 px of it (the only place motion matters)
    let any = false;
    for (let i = 0, o = 0; i < n; i++, o += 3) {
      const m = lut[((img[o] >> 2) << 12) | ((img[o + 1] >> 2) << 6) | (img[o + 2] >> 2)];
      cmask[i] = m;
      any = any || m === 1;
    }
    const blobs = this.wantBlobs ? (any ? this._blobs() : []) : null;
    if (!any) return [cur.idx, [], blobs];
    near.fill(0);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (!cmask[y * w + x]) continue;
        for (let yy = y > 0 ? y - 1 : 0; yy <= (y < h - 1 ? y + 1 : y); yy++) {
          const r = yy * w;
          near[r + (x > 0 ? x - 1 : 0)] = 1; near[r + x] = 1; near[r + (x < w - 1 ? x + 1 : x)] = 1;
        }
      }
    }

    // motion: per baseline, differs from both neighbours (or one at the clip edges)
    motion.fill(0);
    const tests = [];
    for (const k of BASELINES) {
      const before = c - k >= 0, after = c + k < buf.length;
      if (before && after) tests.push([buf[c - k], buf[c + k]]);
      else if (k === 1 && (before || after)) tests.push([buf[before ? c - k : c + k]]);
    }
    for (const pair of tests) {
      passes.fill(0);
      for (let p = 0; p < pair.length; p++) {
        const o = pair[p];
        const dx = cur.pos[0] - o.pos[0], dy = cur.pos[1] - o.pos[1];
        for (let x = 0; x < w; x++) { const v = x - dx; xmap[x] = (v < 0 ? 0 : v >= w ? w - 1 : v) * 3; }
        for (let y = 0; y < h; y++) { const v = y - dy; ymap[y] = (v < 0 ? 0 : v >= h ? h - 1 : v) * w * 3; }
        const lo = o.lo, hi = o.hi;
        for (let y = 0; y < h; y++) {
          const yo = ymap[y], r = y * w;
          for (let x = 0; x < w; x++) {
            const i = r + x;
            if (!near[i] || motion[i] || passes[i] !== p) continue;
            const ci = i * 3, oi = yo + xmap[x];
            const v0 = img[ci], v1 = img[ci + 1], v2 = img[ci + 2];
            if (v0 - hi[oi] > thr || lo[oi] - v0 > thr || v1 - hi[oi + 1] > thr || lo[oi + 1] - v1 > thr
                || v2 - hi[oi + 2] > thr || lo[oi + 2] - v2 > thr) passes[i] = p + 1;
          }
        }
      }
      for (let i = 0; i < n; i++) if (passes[i] === pair.length) motion[i] = 1;
    }

    // candidates = colour mask AND motion dilated by 1 px
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        let m = 0;
        if (cmask[i]) {
          for (let yy = y > 0 ? y - 1 : 0; yy <= (y < h - 1 ? y + 1 : y) && !m; yy++) {
            const r = yy * w;
            m = motion[r + (x > 0 ? x - 1 : 0)] | motion[r + x] | motion[r + (x < w - 1 ? x + 1 : x)];
          }
        }
        comb[i] = m;
      }
    }
    const inv = 1 / this.scale;
    let cands = [];
    for (const s of components(comb, w, h, this.labels)) {
      if (s.area > this.maxArea) continue;
      if (s.x0 <= 2 || s.y0 <= 2 || s.x1 + 1 >= w - 2 || s.y1 + 1 >= h - 2) continue; // edges flicker
      const bw = s.x1 - s.x0 + 1, bh = s.y1 - s.y0 + 1;
      const elong = Math.max(bw, bh) / Math.max(1, Math.min(bw, bh));
      if (elong > 8) continue;
      cands.push({ x: s.cx * inv, y: s.cy * inv, area: s.area, size: Math.sqrt(s.area) / this.diag, elong });
    }
    if (cands.length > this.maxCandidates) cands = cands.sort((a, b) => b.area - a.area).slice(0, this.maxCandidates);
    return [cur.idx, cands, blobs];
  }
}
