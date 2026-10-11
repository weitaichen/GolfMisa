// ShotTracer PWA: UI, playback, camera and export. Ball tracking runs in a Web Worker
// (js/worker.js); everything stays on the device.
//
// Long operations (open, analyse, export, camera start) set S.busy before their first
// await, so they can't overlap; their event listeners hang off an AbortController and
// are removed when the operation ends.

import { BALL_COLORS, TRACER_COLORS, colorFromSample, hexToRgb, matches, rgbToHex, sampleColor } from './colors.js';
import { procSize, ShotResult } from './analysis.js';
import { manualPath, TracerPath } from './fit.js';
import { STYLES, drawMarker, drawPoints, drawTracer } from './render.js';

const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const octx = overlay.getContext('2d');
const stage = $('stage');
const hasRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

const store = {
  get(k, d) { try { const v = localStorage.getItem('shottracer.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('shottracer.' + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

const S = {
  loaded: false, name: '', url: null, fps: 30, vw: 0, vh: 0,
  frame: 0,                      // frame index on screen
  gen: 0,                        // bumped whenever the video source changes
  result: null, path: null, pathKind: '', cands: null,
  seed: null, mode: '', manual: [], manualLaunch: 0,
  ball: BALL_COLORS[0], custom: null,
  style: { color: '#ff2d2d', style: 'Glow', thickness: 6, marker: true },
  extend: 0, debug: false,
  busy: false, camera: null, livePath: null, liveIdx: 0,
};

// ------------------------------------------------------------------ helpers
let toastTimer = 0;
function toast(msg, ms = 3500) {
  const t = $('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}

function fmt(sec) {
  sec = Math.max(0, sec || 0);
  return `${Math.floor(sec / 60)}:${(sec % 60).toFixed(2).padStart(5, '0')}`;
}

/** Resolves with the event, or null after `timeout` ms / when `signal` aborts. */
const once = (el, ev, timeout = 0, signal = null) => new Promise((resolve) => {
  let timer = 0;
  const done = (e) => {
    el.removeEventListener(ev, done);
    if (signal) signal.removeEventListener('abort', abort);
    clearTimeout(timer);
    resolve(e);
  };
  const abort = () => done(null);
  el.addEventListener(ev, done);
  if (signal) signal.addEventListener('abort', abort);
  if (timeout) timer = setTimeout(() => done(null), timeout);
});

function seekTo(t) {
  const target = Math.min(Math.max(0, t), Math.max(0, (video.duration || 0) - 1e-3));
  const p = once(video, 'seeked', 2500);
  video.currentTime = target;
  return p;
}

const seekFrame = (k) => seekTo((k + 0.5) / S.fps);
const sensitivity = () => Number($('sens').value) / 100;
const even = (v) => Math.max(2, Math.round(v / 2) * 2);

function sheet(title, actions) {
  $('sheetTitle').textContent = title;
  const box = $('sheetActions');
  box.replaceChildren();
  for (const [label, fn, cls] of actions) {
    const b = document.createElement('button');
    b.className = 'btn wide ' + (cls || '');
    b.textContent = label;
    b.onclick = async () => { $('sheet').hidden = true; if (fn) await fn(); };
    box.append(b);
  }
  $('sheet').hidden = false;
}

// Display refresh rate: frame callbacks can't report more frames than the screen shows.
let displayHz = 60;
(function measureDisplay() {
  const t = [];
  const step = (now) => { t.push(now); if (t.length < 31) requestAnimationFrame(step); else {
    const d = t.slice(1).map((v, i) => v - t[i]).sort((a, b) => a - b);
    const hz = 1000 / d[Math.floor(d.length / 2)];
    if (hz > 20 && hz < 400) displayHz = hz;
  } };
  requestAnimationFrame(step);
})();

// ------------------------------------------------------------------- worker
const handlers = new Map();
let jobCounter = 0;
let worker = null;

function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => { const h = handlers.get(e.data.job); if (h) h(e.data); };
  const fail = (e) => {
    // fail every pending job so nothing waits forever; the next job gets a fresh worker
    const message = 'Tracking engine stopped: ' + ((e && e.message) || 'failed to load');
    for (const [job, h] of [...handlers]) h({ type: 'error', job, message });
    handlers.clear();
    try { worker.terminate(); } catch { /* already gone */ }
    worker = null;
  };
  worker.onerror = fail;
  worker.onmessageerror = fail;
  return worker;
}
const post = (msg, transfer) => getWorker().postMessage(msg, transfer || []);

const grab = document.createElement('canvas');
const gctx = grab.getContext('2d', { willReadFrequently: true });
function grabFrame(w, h) {
  if (grab.width !== w || grab.height !== h) { grab.width = w; grab.height = h; }
  gctx.drawImage(video, 0, 0, w, h);
  return gctx.getImageData(0, 0, w, h).data.buffer;
}

// ------------------------------------------------------------ busy overlay
let cancelFn = null;
let continueFn = null;
function setBusy(text, cancellable = false) {
  S.busy = !!text;
  $('busy').hidden = !text;
  if (text) { $('busyText').textContent = text; $('busyBar').style.width = '0%'; }
  $('btnCancel').hidden = !cancellable;
  $('btnContinue').hidden = true;
  if (!text) { cancelFn = null; continueFn = null; }
  updateEnabled();
}
const setProgress = (p) => { $('busyBar').style.width = `${Math.min(100, Math.max(0, p * 100)).toFixed(1)}%`; };
$('btnCancel').onclick = () => cancelFn && cancelFn();
$('btnContinue').onclick = () => continueFn && continueFn();

/** play() for long operations. If the browser blocks it (iOS low-power mode, page in the
 *  background) a Continue button appears: a tap is a user gesture, which play() allows. */
function keepPlaying(onBlocked) {
  return video.play().then(() => { $('btnContinue').hidden = true; }).catch(() => {
    continueFn = () => keepPlaying(onBlocked);
    $('btnContinue').hidden = false;
    if (onBlocked) onBlocked();
  });
}

// ------------------------------------------------------------ layout + draw
const narrow = () => window.matchMedia('(max-width: 899px)').matches;
function layoutStage() {
  const wrap = $('stageWrap');
  const aw = wrap.clientWidth;
  const ar = S.vw && S.vh ? S.vw / S.vh : 16 / 9;
  // phones: the video takes at most ~55% of the screen and the controls get the rest
  const ah = narrow() ? window.innerHeight * 0.55 : wrap.clientHeight - $('transport').offsetHeight;
  let w = aw, h = aw / ar;
  if (!S.vw && narrow()) h = Math.max(h, Math.min(ah, 300)); // room for the welcome text
  if (h > ah) { h = ah; w = S.vw ? ah * ar : aw; }
  stage.style.width = `${Math.floor(w)}px`;
  stage.style.height = `${Math.floor(h)}px`;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  overlay.width = Math.max(1, Math.round(w * dpr));
  overlay.height = Math.max(1, Math.round(h * dpr));
  redraw();
}
new ResizeObserver(layoutStage).observe($('stageWrap'));
window.addEventListener('resize', layoutStage);

function redraw() {
  const W = overlay.width, H = overlay.height;
  octx.clearRect(0, 0, W, H);
  if (!S.vw) return;
  const sc = W / S.vw;
  if (S.camera) {
    if (S.livePath) drawTracer(octx, S.livePath, S.livePath.length, S.style, sc, H, S.liveIdx - S.livePath.end < 10);
    return;
  }
  const f = S.frame;
  if (S.path) drawTracer(octx, S.path, S.path.until(f), S.style, sc, H, f <= S.path.end);
  if (S.debug && S.cands) {
    const c = S.cands[f];
    if (c) drawPoints(octx, c, sc, '#00e5ff', 8, true);
    if (S.result && S.result.found) {
      const xy = [];
      S.result.frames.forEach((fr, i) => { if (fr <= f) xy.push(S.result.xs[i], S.result.ys[i]); });
      drawPoints(octx, xy, sc, '#ffd400', 2.5);
    }
  }
  if (S.seed && (!S.path || f < S.path.start)) drawMarker(octx, S.seed[1] * sc, S.seed[2] * sc, 'ball');
  if (S.mode === 'manual') S.manual.forEach((p, i) => drawMarker(octx, p[0] * sc, p[1] * sc, ['start', 'apex', 'land'][i]));
}

let scrubbing = false;
function updateTime() {
  const d = video.duration;
  $('time').textContent = fmt(video.currentTime);
  if (isFinite(d) && d > 0 && !scrubbing) $('scrub').value = String(Math.round((video.currentTime / d) * 1000));
}

// ------------------------------------------------------ frame-accurate sync
let sink = null; // per-frame consumer while analysing / exporting
let rvfcId = 0;
function onVideoFrame(now, meta) {
  rvfcId = video.requestVideoFrameCallback(onVideoFrame);
  if (S.camera) { cameraFrame(meta); return; }
  S.frame = Math.round(meta.mediaTime * S.fps);
  if (sink) sink(meta);
  updateTime();
  redraw();
}
function armFrameLoop() {
  if (!hasRVFC) return;
  if (rvfcId) video.cancelVideoFrameCallback(rvfcId);
  rvfcId = video.requestVideoFrameCallback(onVideoFrame);
}
if (!hasRVFC) {
  const sync = () => { if (!S.camera) { S.frame = Math.floor(video.currentTime * S.fps + 1e-3); updateTime(); redraw(); } };
  video.addEventListener('timeupdate', sync);
  video.addEventListener('seeked', sync);
}

// --------------------------------------------------------------- loading
async function ensureDuration(signal) {
  // MediaRecorder WebM files have no duration until the browser scans to the end
  if (isFinite(video.duration)) return;
  video.currentTime = 1e101;
  await once(video, 'durationchange', 3000, signal);
  await seekTo(0);
}

/** Sample frame timestamps while playing at `rate`; returns seconds per source frame. */
function sampleFrameStep(rate, count, signal) {
  return new Promise((resolve) => {
    const samples = [];
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      video.pause();
      const dts = [];
      for (let i = 1; i < samples.length; i++) {
        const df = samples[i][1] - samples[i - 1][1], dt = samples[i][0] - samples[i - 1][0];
        if (df >= 1 && dt > 0) dts.push(dt / df);
      }
      dts.sort((a, b) => a - b);
      resolve(dts.length ? dts[Math.floor(dts.length / 4)] : 0); // low quartile: skipped frames inflate steps
    };
    const cb = (now, meta) => {
      if (finished) return;
      samples.push([meta.mediaTime, meta.presentedFrames]);
      if (samples.length < count && !video.ended) video.requestVideoFrameCallback(cb); else finish();
    };
    if (signal) signal.addEventListener('abort', finish);
    video.requestVideoFrameCallback(cb);
    video.playbackRate = rate;
    video.play().catch(finish);
    setTimeout(finish, 2500 / rate);
  });
}

async function measureFps(signal) {
  if (!hasRVFC) return 30;
  await seekTo(0);
  let step = await sampleFrameStep(1, 14, signal);
  // the screen caps how many frames are shown: a 120/240 fps clip at 1x looks like 60.
  // Measure again in slow motion so every source frame is presented.
  if (!signal.aborted && step > 0 && 1 / step > 0.8 * displayHz) {
    await seekTo(0);
    const slow = await sampleFrameStep(0.2, 12, signal);
    if (slow > 0) step = Math.min(step, slow);
  }
  let fps = step > 0 ? 1 / step : 30;
  const common = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 90, 100, 119.88, 120, 240];
  const near = common.reduce((a, c) => (Math.abs(c - fps) < Math.abs(a - fps) ? c : a));
  fps = Math.abs(near - fps) / near < 0.03 ? near : Math.round(fps);
  return fps > 1 && fps < 1000 ? fps : 30;
}

async function loadVideo(url, name) {
  if (S.busy) {
    if (url.startsWith('blob:')) URL.revokeObjectURL(url);
    toast('Please wait for the current task to finish.');
    return false;
  }
  if (S.camera) stopCamera(false);
  const gen = ++S.gen;
  const ac = new AbortController();
  setBusy('Opening video…', true);
  cancelFn = () => ac.abort();
  if (S.url && S.url.startsWith('blob:') && S.url !== url) URL.revokeObjectURL(S.url);
  Object.assign(S, { url, name, loaded: false, result: null, path: null, pathKind: '', cands: null, seed: null, manual: [], frame: 0 });
  setMode('');
  clearStats();
  video.srcObject = null;
  video.src = url;
  const ok = await Promise.race([
    once(video, 'loadeddata', 20000, ac.signal).then((e) => !!e),
    once(video, 'error', 0, ac.signal).then(() => false),
  ]);
  const giveUp = (message) => {
    setBusy(null);
    video.removeAttribute('src');
    video.load();
    S.url = null; S.vw = 0; S.vh = 0;
    $('empty').hidden = false;
    layoutStage();
    toast(message, 6000);
    return false;
  };
  if (gen !== S.gen) return false;
  if (ac.signal.aborted) return giveUp('Opening cancelled');
  if (!ok || !video.videoWidth) return giveUp("Can't play this video on this device. Try an MP4 (H.264) file.");
  await ensureDuration(ac.signal);
  S.vw = video.videoWidth;
  S.vh = video.videoHeight;
  $('empty').hidden = true;
  layoutStage();
  armFrameLoop();
  S.fps = await measureFps(ac.signal);
  await seekFrame(0);
  if (gen !== S.gen) return false;
  if (ac.signal.aborted) return giveUp('Opening cancelled');
  S.loaded = true;
  setBusy(null);
  video.playbackRate = Number($('speed').value);
  updateSeedUI();
  toast(`${name} · ${S.vw}×${S.vh} · ${Math.round(S.fps)} fps. Pick the ball color, then Auto Track.`, 5000);
  return true;
}

$('fileInput').addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  e.target.value = '';
  if (f) loadVideo(URL.createObjectURL(f), f.name);
});
$('btnOpen').addEventListener('keydown', (e) => { if ((e.key === 'Enter' || e.key === ' ') && !S.busy) $('fileInput').click(); });
$('btnEmptyOpen').onclick = () => { if (!S.busy) $('fileInput').click(); };
$('btnDemo').onclick = async () => {
  if (S.busy) return;
  selectBall('yellow');
  await loadVideo('samples/demo.mp4', 'demo.mp4');
};

// -------------------------------------------------------------- playback
function togglePlay() {
  if (!S.loaded || S.busy || S.camera) return;
  if (video.paused) {
    if (video.ended || video.currentTime >= video.duration - 1e-3) video.currentTime = 0;
    video.playbackRate = Number($('speed').value);
    video.play().catch(() => {});
  } else video.pause();
}
video.addEventListener('play', () => { $('playIcon').innerHTML = '<path d="M7 5h3v14H7zM14 5h3v14h-3z"/>'; });
video.addEventListener('pause', () => { $('playIcon').innerHTML = '<path d="M8 5l11 7-11 7z"/>'; });
$('btnPlay').onclick = togglePlay;
$('btnPrev').onclick = () => { if (S.loaded && !S.busy) { video.pause(); seekFrame(S.frame - 1); } };
$('btnNext').onclick = () => { if (S.loaded && !S.busy) { video.pause(); seekFrame(S.frame + 1); } };
$('speed').onchange = () => { if (!S.busy) video.playbackRate = Number($('speed').value); };
$('scrub').addEventListener('input', () => {
  if (!S.loaded || S.busy) return;
  scrubbing = true;
  video.pause();
  video.currentTime = (Number($('scrub').value) / 1000) * video.duration;
});
$('scrub').addEventListener('change', () => { scrubbing = false; });

// ----------------------------------------------------------------- taps
function videoPoint(e) {
  const r = overlay.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * S.vw, ((e.clientY - r.top) / r.height) * S.vh];
}

function frameImage(maxSide = 1280) {
  const s = Math.min(1, maxSide / Math.max(S.vw, S.vh));
  const w = Math.round(S.vw * s), h = Math.round(S.vh * s);
  if (grab.width !== w || grab.height !== h) { grab.width = w; grab.height = h; }
  gctx.drawImage(video, 0, 0, w, h);
  return { data: gctx.getImageData(0, 0, w, h).data, w, h, s };
}

overlay.addEventListener('click', (e) => {
  if (S.camera || !S.loaded || S.busy) return;
  if (!S.mode) { togglePlay(); return; }
  const [x, y] = videoPoint(e);
  if (S.mode === 'pick') {
    const img = frameImage();
    const rgb = sampleColor(img.data, img.w, img.h, x * img.s, y * img.s, 2);
    S.custom = colorFromSample(rgb);
    store.set('custom', rgbToHex(rgb));
    renderBallSwatches();
    selectBall('custom');
    toast(`Ball color picked (${rgbToHex(rgb)}). Now press Auto Track Shot.`);
    setMode('');
  } else if (S.mode === 'seed') {
    S.seed = [S.frame, x, y];
    const img = frameImage();
    const rgb = sampleColor(img.data, img.w, img.h, x * img.s, y * img.s, 2);
    if (!matches(S.ball, 0.6, ...rgb)) toast(`Heads-up: that doesn't look like a ${S.ball.name.toLowerCase()} ball. Pick the color if tracking fails.`, 5000);
    setMode('');
    updateSeedUI();
  } else if (S.mode === 'manual') {
    if (!S.manual.length) S.manualLaunch = S.frame;
    S.manual.push([x, y]);
    if (S.manual.length === 3) {
      buildManual();
      setMode('');
      seekFrame(Math.max(0, S.manualLaunch - Math.round(S.fps * 0.5))).then(() => {
        if (S.busy || S.camera) return;
        video.playbackRate = Number($('speed').value);
        video.play().catch(() => {});
      });
    } else { updateHint(); redraw(); }
  }
});

function setMode(mode) {
  if (mode && (!S.loaded || S.camera || S.busy)) mode = '';
  S.mode = mode;
  if (mode) video.pause();
  if (mode === 'manual') S.manual = [];
  $('btnPick').classList.toggle('on', mode === 'pick');
  $('btnSeed').classList.toggle('on', mode === 'seed');
  $('btnManual').classList.toggle('on', mode === 'manual');
  stage.classList.toggle('picking', !!mode);
  updateHint();
  redraw();
}

function updateHint() {
  const hints = {
    pick: 'Tap the ball to sample its color',
    seed: 'Tap the ball sitting at address (before the swing)',
    manual: ['1/3  Tap the ball at impact', '2/3  Tap the highest point', '3/3  Tap where the ball lands'][Math.min(S.manual.length, 2)],
  };
  const text = S.camera ? 'LIVE · swing away – the tracer appears automatically' : hints[S.mode] || '';
  $('hint').textContent = text;
  $('hint').hidden = !text;
}

$('btnPick').onclick = () => setMode(S.mode === 'pick' ? '' : 'pick');
$('btnSeed').onclick = () => setMode(S.mode === 'seed' ? '' : 'seed');
$('btnManual').onclick = () => setMode(S.mode === 'manual' ? '' : 'manual');
$('btnClearSeed').onclick = () => { S.seed = null; updateSeedUI(); redraw(); };
function updateSeedUI() {
  $('btnSeed').textContent = S.seed ? `⛳ Ball marked at ${fmt(S.seed[0] / S.fps)}` : '⛳ Mark ball at address';
  $('btnClearSeed').hidden = !S.seed;
}

// ----------------------------------------------------------- ball colour
function renderBallSwatches() {
  const box = $('ballSwatches');
  box.replaceChildren();
  const items = [...BALL_COLORS.map((c) => [c.key, c.swatch, c.name]), ['custom', S.custom ? S.custom.swatch : null, 'Custom (picked from video)']];
  for (const [key, hex, name] of items) {
    const b = document.createElement('button');
    b.className = 'swatch' + (key === 'custom' ? ' custom' : '') + (S.ball.key === key ? ' on' : '');
    b.title = name;
    b.setAttribute('aria-label', name);
    const i = document.createElement('i');
    if (hex) i.style.background = hex; else { i.style.background = '#2a303a'; i.textContent = '+'; }
    b.append(i);
    b.onclick = () => selectBall(key);
    box.append(b);
  }
  $('ballLabel').innerHTML = `Tracking a <b>${S.ball.name.toLowerCase()}</b> ball`;
}

function selectBall(key) {
  if (key === 'custom' && !S.custom) { setMode('pick'); if (!S.loaded) toast('Open a video, then tap the ball'); return; }
  S.ball = key === 'custom' ? S.custom : BALL_COLORS.find((c) => c.key === key) || BALL_COLORS[0];
  store.set('ball', S.ball.key);
  renderBallSwatches();
  if (S.camera) startLiveJob();
}

$('sens').oninput = () => { $('sensVal').textContent = $('sens').value; store.set('sens', Number($('sens').value)); };
$('sens').onchange = () => { if (S.camera) startLiveJob(); };

// ----------------------------------------------------------------- tracer
function renderTracerControls() {
  const box = $('tracerSwatches');
  box.replaceChildren();
  for (const [name, hex] of TRACER_COLORS) {
    const b = document.createElement('button');
    b.className = 'swatch' + (S.style.color === hex ? ' on' : '');
    b.title = name;
    b.setAttribute('aria-label', name);
    const i = document.createElement('i');
    i.style.background = hex;
    b.append(i);
    b.onclick = () => { S.style.color = hex; store.set('tracerColor', hex); renderTracerControls(); redraw(); };
    box.append(b);
  }
  const seg = $('styleSeg');
  seg.replaceChildren();
  for (const st of STYLES) {
    const b = document.createElement('button');
    b.textContent = st;
    b.className = S.style.style === st ? 'on' : '';
    b.onclick = () => { S.style.style = st; store.set('tracerStyle', st); renderTracerControls(); redraw(); };
    seg.append(b);
  }
}
$('thick').oninput = () => { S.style.thickness = Number($('thick').value); $('thickVal').textContent = $('thick').value; store.set('thick', S.style.thickness); redraw(); };
$('extend').oninput = () => {
  S.extend = Number($('extend').value);
  $('extendVal').textContent = $('extend').value;
  if (!S.busy && S.pathKind === 'auto' && S.result) { S.path = S.result.buildPath(S.extend); redraw(); }
};
$('chkMarker').onchange = () => { S.style.marker = $('chkMarker').checked; redraw(); };
$('chkDebug').onchange = () => { S.debug = $('chkDebug').checked; redraw(); };

// ----------------------------------------------------------------- manual
function buildManual() {
  const secs = Number($('duration').value) / 10;
  S.path = manualPath(...S.manual, S.manualLaunch, Math.round(secs * S.fps));
  S.pathKind = 'manual';
  setStats(`${secs.toFixed(1)} s`, fmt(S.manualLaunch / S.fps), 'manual', '–');
  updateEnabled();
  redraw();
}
$('duration').oninput = () => {
  $('durationVal').textContent = `${(Number($('duration').value) / 10).toFixed(1)} s`;
  if (!S.busy && S.loaded && S.pathKind === 'manual' && S.manual.length === 3) buildManual();
};
$('btnClearTracer').onclick = () => {
  if (S.busy) return;
  S.path = null; S.pathKind = ''; S.manual = []; clearStats(); updateEnabled(); redraw();
};

// ------------------------------------------------------------------ stats
function setStats(hang, launch, points, conf) {
  $('stHang').textContent = hang; $('stLaunch').textContent = launch;
  $('stPoints').textContent = points; $('stConf').textContent = conf;
}
const clearStats = () => setStats('–', '–', '–', '–');

// --------------------------------------------------------------- analysis
async function analyze() {
  if (!S.loaded || S.busy || S.camera) return;
  setBusy(`Tracking ${S.ball.name.toLowerCase()} ball…`, true); // claim before any await
  setMode('');
  video.pause();
  const gen = S.gen;
  const ac = new AbortController();
  const [pw, ph] = procSize(S.vw, S.vh, 640);
  const job = ++jobCounter;
  let inflight = 0, last = -1, sent = 0, gaps = 0, done = false, waiting = false;
  // never play faster than the screen can present frames, or frames are silently skipped
  const maxRate = Math.min(1, (0.85 * displayHz) / S.fps);
  let rate = Math.min(0.5, maxRate);
  let settle = null;
  const result = new Promise((resolve, reject) => { settle = { resolve, reject }; });
  const stop = () => { done = true; sink = null; ac.abort(); video.pause(); };
  cancelFn = () => { stop(); post({ type: 'cancel' }); handlers.delete(job); settle.resolve(null); };
  const finish = () => {
    if (done) return;
    stop();
    post({ type: 'finish', job });
  };
  handlers.set(job, (m) => {
    if (m.type === 'ack') {
      inflight--;
      if (waiting && inflight <= 1 && !done) { waiting = false; keepPlaying(); }
      else if (inflight === 0 && rate < maxRate && !done && !waiting) { rate = Math.min(maxRate, rate * 1.1); video.playbackRate = rate; }
    } else if (m.type === 'result') { handlers.delete(job); settle.resolve(m); }
    else if (m.type === 'error') { stop(); handlers.delete(job); settle.reject(new Error(m.message)); }
  });
  post({
    type: 'start', job, mode: 'analyze',
    opts: { color: S.ball, sensitivity: sensitivity(), fps: S.fps, frameSize: [S.vw, S.vh], procSize: [pw, ph], seed: S.seed },
  });
  const t0 = performance.now();
  const send = (idx) => {
    const buffer = grabFrame(pw, ph);
    post({ type: 'frame', job, idx, buffer }, [buffer]);
    inflight++; sent++; last = idx;
  };

  await seekTo(0);
  if (!done) {
    if (hasRVFC) {
      sink = (meta) => {
        if (done) return;
        const idx = Math.round(meta.mediaTime * S.fps);
        if (idx <= last) return;
        if (idx > last + 1) { // frames went by unseen: slow down so it doesn't happen again
          gaps += idx - last - 1;
          rate = Math.max(0.1, rate * 0.75);
          video.playbackRate = rate;
        }
        send(idx);
        setProgress(meta.mediaTime / video.duration);
        // the worker is falling behind: hold playback until it catches up, and play slower
        if (inflight >= 3 && !video.paused) {
          waiting = true;
          rate = Math.max(0.1, rate * 0.85);
          video.playbackRate = rate;
          video.pause();
        }
      };
      video.addEventListener('ended', finish, { once: true, signal: ac.signal });
      // the page went to the background (the OS pauses video): resume when it's back
      document.addEventListener('visibilitychange', () => {
        if (!document.hidden && !done && !waiting && video.paused) keepPlaying();
      }, { signal: ac.signal });
      send(0); // the frame already on screen; playback callbacks start with the next one
      video.playbackRate = rate;
      keepPlaying();
    } else {
      (async () => { // no frame callbacks: step through the clip frame by frame
        const n = Math.floor(video.duration * S.fps);
        for (let k = 0; k < n && !done; k++) {
          await seekFrame(k);
          while (inflight >= 3 && !done) await new Promise((r) => setTimeout(r, 5));
          if (!done) send(k);
          setProgress(k / n);
        }
        finish();
      })();
    }
  }

  let msg;
  try {
    msg = await result;
  } catch (err) {
    video.playbackRate = Number($('speed').value);
    setBusy(null);
    toast('Tracking failed: ' + err.message, 6000);
    return;
  }
  video.playbackRate = Number($('speed').value);
  if (!msg || gen !== S.gen) { setBusy(null); if (!msg) toast('Tracking cancelled'); return; }
  const secs = (performance.now() - t0) / 1000;
  $('perf').textContent = `Analysed ${sent} frames in ${secs.toFixed(1)} s (${(sent / secs).toFixed(0)} fps on this device${gaps ? `, ${gaps} not seen` : ''}).`;
  S.result = new ShotResult(msg.result);
  S.cands = msg.candidates;
  if (!S.result.found) {
    if (S.pathKind === 'auto') { S.path = null; S.pathKind = ''; clearStats(); }
    setBusy(null);
    redraw();
    sheet("Couldn't find the ball's flight", [
      ['Pick the ball color from the video', () => setMode('pick')],
      ['Mark the ball at address', () => setMode('seed')],
      ['Draw the tracer manually', () => { showTab('manual'); setMode('manual'); }],
      ['Close'],
    ]);
    return;
  }
  S.path = S.result.buildPath(S.extend);
  S.pathKind = 'auto';
  setStats(`${S.result.hangTime.toFixed(2)} s`, fmt(S.result.launchFrame / S.fps), String(S.result.frames.length), S.result.confidence);
  toast(`Ball tracked for ${S.result.hangTime.toFixed(2)} s (${S.result.confidence.toLowerCase()} confidence)`);
  await seekFrame(Math.max(0, S.result.launchFrame - Math.round(S.fps * 0.6)));
  setBusy(null); // released only after the replay seek, so nothing else can interleave
  if (gen === S.gen && !S.busy) video.play().catch(() => {});
}
$('btnTrack').onclick = analyze;

// ------------------------------------------------------------------ export
async function exportVideo() {
  if (!S.path || S.busy || S.camera) return;
  const mime = ['video/mp4;codecs=avc1.42E01F', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
    .find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m));
  if (!mime || !HTMLCanvasElement.prototype.captureStream) { toast('This browser cannot record video.'); return; }
  setBusy('Exporting video…', true);
  setMode('');
  video.pause();
  const path = S.path;            // the tracer being exported, even if the UI changes meanwhile
  const style = { ...S.style };
  const ac = new AbortController();
  const s = Math.min(1, 1920 / Math.max(S.vw, S.vh));
  const W = even(S.vw * s), H = even(S.vh * s);
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const cx = cv.getContext('2d');
  let rec;
  try {
    rec = new MediaRecorder(cv.captureStream(Math.round(S.fps)), { mimeType: mime, videoBitsPerSecond: 10e6 });
  } catch (err) {
    setBusy(null);
    toast('Recording failed to start: ' + err.message, 6000);
    return;
  }
  const chunks = [];
  let failed = null, cancelled = false;
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  const stopped = new Promise((r) => { rec.onstop = r; }); // set up before start: it may stop early
  rec.onerror = (e) => { failed = (e && e.error && e.error.message) || 'encoder error'; };
  const draw = (idx) => {
    cx.drawImage(video, 0, 0, W, H);
    drawTracer(cx, path, path.until(idx), style, W / S.vw, H, idx <= path.end);
  };
  await seekFrame(0);
  draw(0);
  rec.start(250);
  await new Promise((resolve) => {
    cancelFn = () => { cancelled = true; resolve(); };
    stopped.then(resolve); // the recorder gave up on its own
    sink = (meta) => { draw(Math.round(meta.mediaTime * S.fps)); setProgress(meta.mediaTime / video.duration); };
    video.addEventListener('ended', resolve, { once: true, signal: ac.signal });
    video.playbackRate = 1;
    keepPlaying();
  });
  sink = null;
  ac.abort();
  video.pause();
  if (rec.state !== 'inactive') rec.stop();
  await Promise.race([stopped, new Promise((r) => setTimeout(r, 5000))]);
  video.playbackRate = Number($('speed').value);
  setBusy(null);
  if (cancelled) { toast('Export cancelled'); return; }
  if (failed || !chunks.length) { toast('Export failed: ' + (failed || 'no video data was recorded'), 6000); return; }
  const type = mime.split(';')[0];
  const file = new File(chunks, `${S.name.replace(/\.[^.]+$/, '')}_tracer.${type === 'video/mp4' ? 'mp4' : 'webm'}`, { type });
  const download = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(file);
    a.download = file.name;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
  };
  const actions = [];
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    actions.push(['Share / Save to Photos', () => navigator.share({ files: [file], title: 'Golf shot tracer' }).catch(() => {}), 'primary']);
  }
  actions.push(['Download', download, actions.length ? '' : 'primary'], ['Close']);
  sheet(`Video ready · ${(file.size / 1e6).toFixed(1)} MB`, actions);
}
$('btnExport').onclick = exportVideo;

// ------------------------------------------------------------------ camera
function pickRecorderMime() {
  return ['video/mp4;codecs=avc1.42E01F', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm']
    .find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m));
}

async function startCamera() {
  if (S.busy || S.camera) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast('Camera not available here (the page must be opened over HTTPS).', 5000);
    return;
  }
  setBusy('Starting camera…'); // also stops a second tap from opening a second stream
  const gen = ++S.gen;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } },
    });
  } catch (err) {
    setBusy(null);
    toast('Camera permission was denied or no camera found.', 5000);
    return;
  }
  if (gen !== S.gen) { stream.getTracks().forEach((t) => t.stop()); setBusy(null); return; }
  setMode('');
  video.pause();
  video.removeAttribute('src');
  video.srcObject = stream;
  await video.play().catch(() => {});
  const settings = stream.getVideoTracks()[0].getSettings();
  S.camera = { stream, job: 0, inflight: 0, recording: null, fps: settings.frameRate || 30, done: 0, t0: performance.now() };
  S.vw = video.videoWidth || settings.width || 1280;
  S.vh = video.videoHeight || settings.height || 720;
  S.livePath = null;
  $('empty').hidden = true;
  $('btnRec').hidden = false;
  $('btnRec').disabled = false;
  $('btnCamera').classList.add('on');
  setBusy(null);
  layoutStage();
  armFrameLoop();
  updateHint();
  startLiveJob();
}

// phone rotated (or camera switched resolution): frame size changed
video.addEventListener('resize', () => {
  if (!S.camera || !video.videoWidth) return;
  if (video.videoWidth === S.vw && video.videoHeight === S.vh) return;
  S.vw = video.videoWidth;
  S.vh = video.videoHeight;
  layoutStage();
  startLiveJob();
});

function startLiveJob() {
  const cam = S.camera;
  if (!cam) return;
  if (cam.job) handlers.delete(cam.job);
  const [pw, ph] = procSize(S.vw, S.vh, 480);
  cam.job = ++jobCounter;
  cam.pw = pw; cam.ph = ph; cam.inflight = 0;
  S.livePath = null;
  const job = cam.job;
  handlers.set(job, (m) => {
    if (!S.camera || S.camera.job !== job) return;
    if (m.type === 'live') {
      S.camera.inflight--;
      S.camera.done++;
      S.livePath = TracerPath.fromJSON(m.path);
      redraw();
    } else if (m.type === 'ack') {
      S.camera.inflight--;
    } else if (m.type === 'error') {
      S.camera.inflight = 0;
      toast(m.message, 5000);
      setTimeout(startLiveJob, 500); // the worker is recreated on the next post
    }
  });
  post({
    type: 'start', job, mode: 'live',
    opts: { color: S.ball, sensitivity: sensitivity(), fps: cam.fps, frameSize: [S.vw, S.vh], procSize: [pw, ph] },
  });
}

function cameraFrame(meta) {
  const cam = S.camera;
  S.liveIdx = meta.presentedFrames;
  // the worker always gets the newest frame; frames arriving while it is busy are skipped
  if (cam.inflight < 2 && cam.job) {
    const buffer = grabFrame(cam.pw, cam.ph);
    post({ type: 'frame', job: cam.job, idx: meta.presentedFrames, buffer }, [buffer]);
    cam.inflight++;
  }
  const secs = (performance.now() - cam.t0) / 1000;
  if (secs > 1) $('perf').textContent = `Live tracking ${(cam.done / secs).toFixed(0)} fps · camera ${Math.round(cam.fps)} fps`;
  redraw();
}

/** Turn the camera off. restore: go back to the previously opened video. */
function stopCamera(restore = true) {
  const cam = S.camera;
  if (!cam) return;
  cam.stream.getTracks().forEach((t) => t.stop());
  handlers.delete(cam.job);
  post({ type: 'cancel' });
  S.camera = null;
  S.livePath = null;
  const gen = ++S.gen;
  video.srcObject = null;
  $('btnRec').hidden = true;
  $('btnRec').classList.remove('on');
  $('btnRec').textContent = 'REC';
  $('btnCamera').classList.remove('on');
  updateHint();
  if (restore && S.loaded && S.url) {
    video.src = S.url;
    once(video, 'loadeddata', 20000).then(() => {
      if (gen !== S.gen) return; // something else was loaded meanwhile
      S.vw = video.videoWidth; S.vh = video.videoHeight;
      layoutStage(); armFrameLoop(); seekFrame(S.frame);
    });
  } else {
    S.loaded = false;
    S.vw = 0; S.vh = 0;
    $('empty').hidden = false;
    layoutStage();
  }
  updateEnabled();
}

function startRecording() {
  const cam = S.camera;
  const mime = pickRecorderMime();
  if (!mime) { toast('Recording is not supported in this browser.'); return; }
  const chunks = [];  // per recording, so a quick restart can't mix two clips
  let rec;
  try {
    rec = new MediaRecorder(cam.stream, { mimeType: mime, videoBitsPerSecond: 12e6 });
  } catch (err) { toast('Recording failed to start: ' + err.message, 5000); return; }
  cam.recording = rec;
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  rec.onerror = () => toast('Recording failed.', 5000);
  rec.onstop = async () => {
    const type = mime.split(';')[0];
    if (!chunks.length) { toast('Nothing was recorded.'); stopCamera(); return; }
    const blob = new Blob(chunks, { type });
    stopCamera(false); // the recording replaces the current video
    if (await loadVideo(URL.createObjectURL(blob), `swing.${type === 'video/mp4' ? 'mp4' : 'webm'}`)) analyze();
  };
  rec.start(500);
  $('btnRec').classList.add('on');
  $('btnRec').textContent = 'STOP';
}

function stopRecording() {
  const cam = S.camera;
  if (!cam || !cam.recording || cam.recording.state === 'inactive') return false;
  $('btnRec').disabled = true; // until onstop has handed the clip over
  $('btnRec').textContent = '…';
  cam.recording.stop();
  return true;
}

$('btnCamera').onclick = () => {
  if (!S.camera) { startCamera(); return; }
  if (!stopRecording()) stopCamera(); // a running recording is kept and analysed (see onstop)
};
$('btnRec').onclick = () => {
  const cam = S.camera;
  if (!cam || $('btnRec').disabled) return;
  if (cam.recording && cam.recording.state !== 'inactive') stopRecording();
  else startRecording();
};

// ------------------------------------------------------------------- tabs
function showTab(name) {
  document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === name));
  document.querySelectorAll('.tab').forEach((t) => { t.hidden = t.dataset.tab !== name; });
}
document.querySelectorAll('.tabs button').forEach((b) => { b.onclick = () => showTab(b.dataset.tab); });

function updateEnabled() {
  const ready = S.loaded && !S.busy && !S.camera;
  $('btnTrack').disabled = !ready;
  $('btnExport').disabled = !(ready && S.path);
  for (const id of ['btnPick', 'btnSeed', 'btnManual', 'btnClearTracer', 'btnClearSeed', 'duration', 'extend']) $(id).disabled = !ready;
  for (const id of ['btnPlay', 'btnPrev', 'btnNext', 'scrub', 'speed']) $(id).disabled = !S.loaded || !!S.camera;
  $('btnCamera').disabled = S.busy;
  $('btnOpen').classList.toggle('disabled', S.busy);
  $('btnDemo').disabled = S.busy;
  $('btnEmptyOpen').disabled = S.busy;
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'ArrowLeft') $('btnPrev').click();
  else if (e.key === 'ArrowRight') $('btnNext').click();
  else if (e.key === 'Escape') setMode('');
});

// ------------------------------------------------------------------- init
(function init() {
  const custom = store.get('custom', null);
  if (custom) S.custom = colorFromSample(hexToRgb(custom));
  const key = store.get('ball', 'white');
  S.ball = key === 'custom' && S.custom ? S.custom : BALL_COLORS.find((c) => c.key === key) || BALL_COLORS[0];
  S.style.color = store.get('tracerColor', S.style.color);
  S.style.style = store.get('tracerStyle', S.style.style);
  S.style.thickness = store.get('thick', 6);
  $('thick').value = S.style.thickness; $('thickVal').textContent = S.style.thickness;
  $('sens').value = store.get('sens', 50); $('sensVal').textContent = $('sens').value;
  renderBallSwatches();
  renderTracerControls();
  updateEnabled();
  layoutStage();
  getWorker();
  if ('serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost')) {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  }
  window.__shottracer = S; // handy for debugging / automated tests
})();
