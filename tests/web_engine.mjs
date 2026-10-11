// Runs the browser tracking engine (web/js) in Node on a frame dump.
//   node tests/web_engine.mjs <dump.json> <colorKey|r,g,b>
// The dump (written by tests/test_web_engine.py) holds RGBA frames at processing size.
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { Analyzer } from '../web/js/analysis.js';
import { BALL_COLORS, colorFromSample } from '../web/js/colors.js';

const [, , metaPath, colorArg] = process.argv;
const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
const raw = readFileSync(meta.frames);
const color = colorArg.includes(',')
  ? colorFromSample(colorArg.split(',').map(Number))
  : BALL_COLORS.find((c) => c.key === colorArg);

const [pw, ph] = meta.proc_size;
const an = new Analyzer({ color, fps: meta.fps, frameSize: meta.size, procSize: [pw, ph] });
const frameBytes = pw * ph * 4;
const t0 = performance.now();
for (let i = 0; i < meta.count; i++) {
  an.push(i, new Uint8ClampedArray(raw.buffer, raw.byteOffset + i * frameBytes, frameBytes));
}
const res = an.finish();
const ms = (performance.now() - t0) / meta.count;
const path = res.buildPath();
console.log(JSON.stringify({
  found: res.found,
  score: res.score,
  ms_per_frame: ms,
  address: res.address,
  launch: res.found ? res.launchFrame : null,
  path: path ? path.toJSON() : null,
}));
