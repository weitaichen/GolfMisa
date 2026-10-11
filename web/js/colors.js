// Ball colour profiles (HSV ranges, OpenCV scale: H 0-179, S 0-255, V 0-255),
// colour sampling and the tracer palette.

export const BALL_COLORS = [
  { key: 'white', name: 'White', ranges: [[[0, 0, 165], [179, 70, 255]]], swatch: '#f4f4f4' },
  { key: 'yellow', name: 'Yellow', ranges: [[[18, 70, 110], [38, 255, 255]]], swatch: '#ffe23d' },
  { key: 'orange', name: 'Orange', ranges: [[[5, 110, 120], [20, 255, 255]]], swatch: '#ff8a1f' },
  { key: 'red', name: 'Red', ranges: [[[0, 100, 80], [6, 255, 255]], [[165, 100, 80], [179, 255, 255]]], swatch: '#ff2b2b' },
  { key: 'pink', name: 'Pink', ranges: [[[140, 50, 120], [170, 255, 255]]], swatch: '#ff4fc8' },
  { key: 'green', name: 'Lime', ranges: [[[36, 80, 110], [80, 255, 255]]], swatch: '#7dff3a' },
  { key: 'blue', name: 'Blue', ranges: [[[92, 90, 70], [128, 255, 255]]], swatch: '#2f8cff' },
];

export const TRACER_COLORS = [
  ['Red', '#ff2d2d'], ['Orange', '#ff8c1a'], ['Yellow', '#ffe600'], ['Green', '#2dff6a'],
  ['Cyan', '#22e4ff'], ['Blue', '#3d7bff'], ['Purple', '#b44dff'], ['White', '#ffffff'],
];

/** RGB -> HSV with OpenCV's 8-bit convention. */
export function rgbToHsv(r, g, b) {
  const v = Math.max(r, g, b);
  const d = v - Math.min(r, g, b);
  const s = v === 0 ? 0 : Math.round((255 * d) / v);
  let h = 0;
  if (d !== 0) {
    if (v === r) h = (60 * (g - b)) / d;
    else if (v === g) h = 120 + (60 * (b - r)) / d;
    else h = 240 + (60 * (r - g)) / d;
    if (h < 0) h += 360;
  }
  return [Math.round(h / 2) % 180, s, v];
}

/** HSV bounds of a profile, widened by `tolerance` (0..1) so motion-blurred,
 *  washed-out balls still match. */
export function rangeBounds(color, tolerance) {
  const dh = Math.round(10 * tolerance);
  const dsv = Math.round(60 * tolerance);
  return color.ranges.map(([lo, hi]) => {
    const fullHue = lo[0] <= 0 && hi[0] >= 179;
    const sFloor = lo[1] >= 50 ? 35 : 0;
    return [
      [fullHue ? 0 : Math.max(0, lo[0] - dh), Math.max(sFloor, lo[1] - dsv), Math.max(0, lo[2] - dsv)],
      [fullHue ? 179 : Math.min(179, hi[0] + dh), Math.min(255, hi[1] + dsv), Math.min(255, hi[2] + dsv)],
    ];
  });
}

export function matches(color, tolerance, r, g, b) {
  const [h, s, v] = rgbToHsv(r, g, b);
  return rangeBounds(color, tolerance).some(([lo, hi]) =>
    h >= lo[0] && h <= hi[0] && s >= lo[1] && s <= hi[1] && v >= lo[2] && v <= hi[2]);
}

const lutCache = new Map();

/** 64x64x64 lookup table: lut[(r>>2)<<12 | (g>>2)<<6 | (b>>2)] = 1 if the colour matches.
 *  Turns the per-pixel HSV test into one array read. Cached per profile + tolerance. */
export function buildMaskLUT(color, tolerance) {
  const key = JSON.stringify([color.ranges, tolerance]);
  const hit = lutCache.get(key);
  if (hit) return hit;
  const bounds = rangeBounds(color, tolerance);
  const lut = new Uint8Array(1 << 18);
  for (let r = 0; r < 64; r++) {
    for (let g = 0; g < 64; g++) {
      for (let b = 0; b < 64; b++) {
        const [h, s, v] = rgbToHsv(r * 4 + 2, g * 4 + 2, b * 4 + 2);
        for (const [lo, hi] of bounds) {
          if (h >= lo[0] && h <= hi[0] && s >= lo[1] && s <= hi[1] && v >= lo[2] && v <= hi[2]) {
            lut[(r << 12) | (g << 6) | b] = 1;
            break;
          }
        }
      }
    }
  }
  if (lutCache.size >= 8) lutCache.delete(lutCache.keys().next().value);
  lutCache.set(key, lut);
  return lut;
}

export function rgbToHex([r, g, b]) {
  return '#' + [r, g, b].map((c) => Math.round(c).toString(16).padStart(2, '0')).join('');
}

export function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

/** Custom profile built around a colour sampled from the video. */
export function colorFromSample([r, g, b]) {
  const [h, s, v] = rgbToHsv(r, g, b);
  let ranges;
  if (s < 45) { // white / grey / silver ball
    ranges = [[[0, 0, Math.max(60, v - 70)], [179, Math.min(255, s + 50), 255]]];
  } else {
    const loS = Math.min(s, Math.max(50, s - 90));
    const loV = Math.min(v, Math.max(60, v - 90));
    const hLo = h - 10;
    const hHi = h + 10;
    if (hLo < 0) ranges = [[[0, loS, loV], [hHi, 255, 255]], [[180 + hLo, loS, loV], [179, 255, 255]]];
    else if (hHi > 179) ranges = [[[hLo, loS, loV], [179, 255, 255]], [[0, loS, loV], [hHi - 180, 255, 255]]];
    else ranges = [[[hLo, loS, loV], [hHi, 255, 255]]];
  }
  return { key: 'custom', name: 'Custom', ranges, swatch: rgbToHex([r, g, b]) };
}

const median = (a) => {
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Colour of the object tapped at (x, y) in RGBA image data. The background colour
 *  comes from a ring around the tap; the patch pixels most different from it are the
 *  ball - works for a white ball on grass and a red ball on a white mat. */
export function sampleColor(data, width, height, x, y, radius = 2) {
  const xi = Math.round(x);
  const yi = Math.round(y);
  const px = (xx, yy) => {
    const i = (yy * width + xx) * 4;
    return [data[i], data[i + 1], data[i + 2]];
  };
  const box = (r) => {
    const out = [];
    for (let yy = Math.max(0, yi - r); yy <= Math.min(height - 1, yi + r); yy++)
      for (let xx = Math.max(0, xi - r); xx <= Math.min(width - 1, xi + r); xx++) out.push(px(xx, yy));
    return out;
  };
  const patch = box(radius);
  if (!patch.length) return [255, 255, 255];
  const R = radius * 4 + 2;
  const x0 = Math.max(0, xi - R), x1 = Math.min(width - 1, xi + R);
  const y0 = Math.max(0, yi - R), y1 = Math.min(height - 1, yi + R);
  const ring = [];
  for (let xx = x0; xx <= x1; xx++) ring.push(px(xx, y0), px(xx, y1));
  for (let yy = y0; yy <= y1; yy++) ring.push(px(x0, yy), px(x1, yy));
  const bg = [0, 1, 2].map((c) => median(ring.map((p) => p[c])));
  const dist = patch.map((p) => Math.hypot(p[0] - bg[0], p[1] - bg[1], p[2] - bg[2]));
  const thr = 0.5 * Math.max(...dist);
  const keep = patch.filter((_, i) => dist[i] >= thr);
  return [0, 1, 2].map((c) => Math.round(median(keep.map((p) => p[c]))));
}
