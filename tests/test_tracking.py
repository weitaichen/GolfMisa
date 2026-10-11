"""End-to-end tracking accuracy on synthetic shots of different ball colours.

    python -m pytest tests -q        or        python tests/test_tracking.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from golf_tracer.analysis import AnalysisSettings, analyze_video  # noqa: E402
from golf_tracer.colors import BALL_COLOR_BY_KEY, color_from_sample  # noqa: E402
from tools.make_test_video import BALL_BGR, make_video  # noqa: E402

COLORS = ["white", "yellow", "orange", "red", "pink", "green", "blue"]
_TMP = Path(tempfile.gettempdir()) / "golf_tracer_tests"


def _video(color, fps=60, shake=1.5, seed=0):
    _TMP.mkdir(exist_ok=True)
    p = _TMP / f"{color}_{fps}_{shake}_{seed}.mp4"
    if p.exists() and p.with_suffix(".json").exists():
        return str(p), json.loads(p.with_suffix(".json").read_text())
    truth = make_video(str(p), color, fps=fps, shake=shake, seed=seed)
    return str(p), truth


def evaluate(color, profile=None, fps=60, shake=1.5, seed=0):
    path, truth = _video(color, fps, shake, seed)
    profile = profile or BALL_COLOR_BY_KEY[color]
    res = analyze_video(path, AnalysisSettings(profile))
    gt = {int(f): (x, y) for f, x, y in truth["frames"]}
    if not res.found:
        return dict(found=False, err=np.inf, coverage=0.0, launch_err=None, res=res)
    tp = res.build_path()
    errs = [np.hypot(*(tp.pts[f - tp.start] - gt[f])) for f in tp.frames if f in gt] or [np.inf]
    flight = [f for f in gt if f - truth["launch"] <= len(gt) * 0.8]   # far end may be sub-pixel
    coverage = np.mean([tp.start <= f <= tp.end for f in flight])
    addr_err = None if res.address is None else float(np.hypot(*(np.array(res.address) - truth["address"])))
    return dict(found=True, err=float(np.mean(errs)), max_err=float(np.max(errs)), coverage=float(coverage),
                launch_err=res.launch_frame - truth["launch"], addr_err=addr_err, score=res.score, res=res)


def test_all_colors_tracked():
    for c in COLORS:
        r = evaluate(c)
        assert r["found"], c
        assert r["err"] < 6, (c, r["err"])
        assert r["coverage"] > 0.8, (c, r["coverage"])
        assert abs(r["launch_err"]) <= 3, (c, r["launch_err"])


def test_30fps_no_shake():
    r = evaluate("yellow", fps=30, shake=0)
    assert r["found"] and r["err"] < 6 and r["coverage"] > 0.8


def test_custom_sampled_color():
    r = evaluate("pink", profile=color_from_sample(BALL_BGR["pink"]))
    assert r["found"] and r["err"] < 6


def test_wrong_color_finds_nothing_good():
    r = evaluate("red", profile=BALL_COLOR_BY_KEY["blue"])
    assert not r["found"] or r["coverage"] < 0.3


if __name__ == "__main__":
    print(f"{'ball':8} {'found':6} {'mean err':>9} {'max err':>8} {'cover':>6} {'launch':>7} {'addr':>6} {'score':>6}")
    for c in COLORS:
        for fps, shake in ((60, 1.5), (30, 0.0)):
            r = evaluate(c, fps=fps, shake=shake)
            if r["found"]:
                print(f"{c:8} {fps:>3}fps {r['err']:9.2f} {r['max_err']:8.2f} {r['coverage']:6.2f} "
                      f"{r['launch_err']:7d} {r['addr_err'] if r['addr_err'] is None else round(r['addr_err'], 1)!s:>6} "
                      f"{r['score']:6.2f}")
            else:
                print(f"{c:8} {fps:>3}fps NOT FOUND")
    r = evaluate("red", profile=BALL_COLOR_BY_KEY["blue"])
    print("red ball w/ blue profile -> found:", r["found"], "coverage:", r["coverage"])
