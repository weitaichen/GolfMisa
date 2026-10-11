// Ball tracking runs here, off the UI thread, so the page stays smooth on a phone.
//
// main -> worker  {type:'start', job, mode:'analyze'|'live', opts}
//                 {type:'frame', job, idx, buffer}      RGBA at opts.procSize (transferred)
//                 {type:'finish', job} | {type:'cancel'}
// worker -> main  {type:'ack', job, buffer}             frame processed (buffer returned for reuse)
//                 {type:'live', job, idx, path, buffer} live tracer for that frame
//                 {type:'result', job, result, candidates}
//                 {type:'error', job, message}

import { Analyzer, LiveTracker } from './analysis.js';

let job = null;
let jobId = -1;
let mode = '';

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'start') {
      jobId = m.job;
      mode = m.mode;
      job = mode === 'live' ? new LiveTracker(m.opts) : new Analyzer(m.opts);
    } else if (m.type === 'cancel') {
      job = null;
      jobId = -1;
    } else if (m.type === 'frame') {
      if (m.job !== jobId || !job) { self.postMessage({ type: 'ack', job: m.job, buffer: m.buffer }, [m.buffer]); return; }
      const rgba = new Uint8ClampedArray(m.buffer);
      if (mode === 'live') {
        const path = job.process(m.idx, rgba);
        self.postMessage({ type: 'live', job: m.job, idx: m.idx, path: path && path.toJSON(), buffer: m.buffer }, [m.buffer]);
      } else {
        job.push(m.idx, rgba);
        self.postMessage({ type: 'ack', job: m.job, idx: m.idx, buffer: m.buffer }, [m.buffer]);
      }
    } else if (m.type === 'finish') {
      if (m.job !== jobId || !job) return;
      const r = job.finish();
      const candidates = {};
      for (const [f, xy] of job.candidates()) candidates[f] = Array.from(xy);
      self.postMessage({
        type: 'result', job: m.job, candidates,
        result: { fps: r.fps, frameSize: r.frameSize, frames: r.frames, xs: r.xs, ys: r.ys, address: r.address, score: r.score },
      });
      job = null;
    }
  } catch (err) {
    self.postMessage({ type: 'error', job: m.job, message: String(err && err.message || err) });
  }
};
