# ShotTracer – golf shot tracer for any ball colour

A desktop app inspired by **SmoothSwing**: load (or record) a swing video, and it finds the
golf ball after impact and draws a glowing shot-tracer line over its flight. Unlike tracers that
only look for a white ball, you can track **white, yellow, orange, red, pink, lime, blue** balls –
or **pick any colour straight from the video** with the eyedropper.

## Install & run

```bash
pip install -r requirements.txt
python main.py                 # or: python main.py my_swing.mp4
```

`ffmpeg` on the PATH is optional – when present, exports are H.264 with the original audio.

## Using it

1. **Open Video** (or drag & drop a file, or use **Live Camera → Record**).
2. **Ball color** – choose the ball's colour. For an unusual ball, press
   *Pick ball color from video*, pause on a frame where the ball is visible and click it
   (a zoom loupe helps hit small balls).
3. *(optional)* **Mark ball at address** – pause before the swing and click the ball. This anchors
   the tracer to the tee and helps when other moving things share the ball colour.
4. **Auto Track Shot** – the tracer appears and the shot replays automatically.
5. Style it: tracer colour, **Glow / Solid / Comet**, thickness, *Extend tail* (extrapolate past
   the last frame the ball was visible), ball marker, slow-motion playback (0.1×–2×).
6. If tracking can't find the ball, use **Draw tracer manually** – click the ball, the apex and
   the landing spot; adjust *Flight time*.
7. **Export** an MP4 with the tracer burned in.

**Live Camera** draws the tracer in real time; **Record** saves the clip to
`~/Videos/ShotTracer/` and analyses it automatically when you stop.

Shortcuts: `Space` play/pause · `←/→` step frame · `Ctrl+O` open · `Ctrl+E` export · `Esc` cancel tool.

## How the tracking works

| Stage | File | What it does |
|---|---|---|
| Colour profiles | `golf_tracer/colors.py` | HSV ranges per ball colour (red wraps the hue circle); custom profile built from a sampled pixel. *Sensitivity* widens the ranges so motion-blurred, washed-out balls still match. |
| Detection | `golf_tracer/detector.py` | A ball in flight is the only thing that is **both moving and ball-coloured**. Motion = pixel differs from frames *t±k* (k = 1, 2, 4 so a far, slow ball is still seen), compared against the 3×3 min/max of the other frame so 1-px edge flicker from camera shake isn't motion. Hand-held shake is compensated by phase correlation. Motion ∩ colour mask → blobs → candidates. |
| Linking | `golf_tracer/tracker.py` | Multi-hypothesis tracker: candidates join tracks inside a prediction gate (quadratic motion model) with speed/turn limits – a ball never stops dead or turns 90° in one frame. |
| Picking the ball | `tracker.py` | Tracks are scored on length, distance travelled, smooth turning, low jitter, detection coverage, small size and a **clutter map** (spots that keep firing – leaves, cloud/tree edges – are penalised). The ball sitting at address is split off and used as the launch point. |
| Re-growing | `tracker.py` | The winning fragment is re-traced through the raw detections. Dropouts (e.g. a white ball crossing a white cloud) are bridged; a re-acquired candidate must be confirmed by the following frames, so noise can't hijack the line. |
| Smoothing | `golf_tracer/path.py` | Robust (bisquare) polynomial fit, falling back to robust LOESS; leave-one-out outlier removal. Launch is joined to the address with a straight segment. |
| Rendering | `golf_tracer/render.py` | Glow / Solid / Comet tracers drawn progressively as the ball flies. |

## Tests

`tools/make_test_video.py` generates synthetic down-the-line shots with ground truth, including a
white glove swinging past, white clouds, camera shake, sensor noise and motion blur.

```bash
python tests/test_tracking.py     # accuracy table for every ball colour at 60 and 30 fps
python -m pytest tests -q
```

Current results: every colour is found, launch frame within ±1 frame, 96–100 % of the flight
covered, mean error 0.8–2.8 px; using the wrong colour profile finds nothing.

Tips for real footage: film from behind (down-the-line) on a tripod if possible, use 60 fps or
higher, and keep the ball's background contrasting with the ball colour (a white ball against a
white sky is invisible to any tracker – use a coloured ball or the manual tracer).
