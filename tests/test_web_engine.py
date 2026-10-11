"""Accuracy of the browser (JavaScript) tracking engine on the synthetic shots.

Frames are decoded and scaled here, then fed to web/js through Node.

    python -m pytest tests/test_web_engine.py -q     or     python tests/test_web_engine.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_tracking import _video  # noqa: E402
from tools.make_test_video import BALL_BGR  # noqa: E402

COLORS = ["white", "yellow", "orange", "red", "pink", "green", "blue"]
NODE = shutil.which("node")
_TMP = Path(tempfile.gettempdir()) / "golf_tracer_tests"


def proc_size(w, h, long_side=640):
    s = min(1.0, long_side / max(w, h))
    return max(2, round(w * s / 2) * 2), max(2, round(h * s / 2) * 2)


def dump(video: str) -> Path:
    meta = Path(video).with_suffix(".frames.json")
    if meta.exists():
        return meta
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pw, ph = proc_size(w, h)
    raw = Path(video).with_suffix(".rgba")
    n = 0
    with open(raw, "wb") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
            f.write(cv2.cvtColor(small, cv2.COLOR_BGR2RGBA).tobytes())
            n += 1
    cap.release()
    meta.write_text(json.dumps({"frames": str(raw), "count": n, "fps": fps, "size": [w, h], "proc_size": [pw, ph]}))
    return meta


def evaluate(color, profile=None, fps=60, shake=1.5):
    video, truth = _video(color, fps, shake)
    out = subprocess.run([NODE, str(ROOT / "tests" / "web_engine.mjs"), str(dump(video)), profile or color],
                         capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    gt = {int(f): (x, y) for f, x, y in truth["frames"]}
    if not r["found"]:
        return dict(found=False, coverage=0.0, err=np.inf, r=r)
    p = r["path"]
    pts = {f: (x, y) for f, x, y in zip(p["frames"], p["xs"], p["ys"])}
    errs = [np.hypot(pts[f][0] - gt[f][0], pts[f][1] - gt[f][1]) for f in pts if f in gt] or [np.inf]
    flight = [f for f in gt if f - truth["launch"] <= len(gt) * 0.8]
    coverage = float(np.mean([p["frames"][0] <= f <= p["frames"][-1] for f in flight]))
    return dict(found=True, err=float(np.mean(errs)), max_err=float(np.max(errs)), coverage=coverage,
                launch_err=r["launch"] - truth["launch"], score=r["score"], ms=r["ms_per_frame"], r=r)


needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


@needs_node
@pytest.mark.parametrize("color", COLORS)
def test_web_engine_tracks_every_color(color):
    r = evaluate(color)
    assert r["found"], color
    assert r["err"] < 6, (color, r["err"])
    assert r["coverage"] > 0.8, (color, r["coverage"])
    assert abs(r["launch_err"]) <= 3, (color, r["launch_err"])


@needs_node
def test_web_engine_custom_color_and_30fps():
    r = evaluate("pink", profile=",".join(str(c) for c in BALL_BGR["pink"][::-1]), fps=30, shake=0.0)
    assert r["found"] and r["err"] < 6 and r["coverage"] > 0.8


@needs_node
def test_web_engine_wrong_color_finds_nothing():
    r = evaluate("red", profile="blue")
    assert not r["found"] or r["coverage"] < 0.3


if __name__ == "__main__":
    print(f"{'ball':8} {'fps':>4} {'found':6} {'mean err':>9} {'max err':>8} {'cover':>6} {'launch':>7} {'score':>6} {'ms/fr':>6}")
    for c in COLORS:
        for fps, shake in ((60, 1.5), (30, 0.0)):
            r = evaluate(c, fps=fps, shake=shake)
            if r["found"]:
                print(f"{c:8} {fps:>4} {'yes':6} {r['err']:9.2f} {r['max_err']:8.2f} {r['coverage']:6.2f} "
                      f"{r['launch_err']:7d} {r['score']:6.2f} {r['ms']:6.1f}")
            else:
                print(f"{c:8} {fps:>4} NOT FOUND (score {r['r']['score']:.2f})")
    r = evaluate("red", profile="blue")
    print("red ball w/ blue profile -> found:", r["found"], "coverage:", r["coverage"])
