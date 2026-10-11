"""Generate a synthetic down-the-line golf shot video with known ball positions.

Includes realistic distractors: a swinging golfer with a white glove, static white
clouds, sensor noise, hand-held camera shake and motion blur on the ball.

    python tools/make_test_video.py --color yellow --out samples/yellow.mp4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

BALL_BGR = {
    "white": (245, 245, 245),
    "yellow": (40, 230, 250),
    "orange": (30, 140, 255),
    "red": (40, 40, 225),
    "pink": (200, 90, 255),
    "green": (60, 255, 150),
    "blue": (255, 140, 40),
}


def _background(w, h, rng):
    horizon = int(h * 0.42)
    img = np.zeros((h, w, 3), np.float32)
    for y in range(horizon):                       # sky gradient
        t = y / horizon
        img[y] = (235 - 40 * t, 190 - 10 * t, 120 + 40 * t)
    for _ in range(5):                             # static clouds
        cx, cy = rng.integers(0, w), rng.integers(10, horizon - 40)
        for _ in range(6):
            cv2.ellipse(img, (int(cx + rng.integers(-60, 60)), int(cy + rng.integers(-12, 12))),
                        (int(rng.integers(30, 70)), int(rng.integers(12, 24))), 0, 0, 360,
                        (240, 240, 240), -1, cv2.LINE_AA)
    img[:horizon] = cv2.GaussianBlur(img[:horizon], (0, 0), 3)
    tree = int(h * 0.06)                           # tree line
    for x in range(0, w, 6):
        top = horizon - tree + int(10 * math.sin(x * 0.05) + rng.integers(-6, 6))
        img[top:horizon + 4, x:x + 6] = (40, 80 + rng.integers(0, 20), 45)
    grass = np.zeros((h - horizon, w, 3), np.float32)
    grass[:] = (60, 150, 70)
    grass += rng.normal(0, 9, grass.shape).astype(np.float32)
    grass = cv2.GaussianBlur(grass, (0, 0), 1.2)
    img[horizon:] = grass
    return np.clip(img, 0, 255), horizon


def make_video(out_path: str, color: str = "white", w=1280, h=720, fps=60, seconds=3.6, launch_s=1.0,
               shake=1.5, noise=2.5, seed=0):
    rng = np.random.default_rng(seed)
    ball_bgr = BALL_BGR[color] if isinstance(color, str) else color
    bg, horizon = _background(w, h, rng)
    ax, ay = w * 0.46, h * 0.84
    r0 = 7.0
    n_frames = int(seconds * fps)
    launch = int(launch_s * fps)
    flight = max(1, int((seconds - launch_s - 0.25) * fps))

    def ball_pos(k):
        """Ball position k frames after impact."""
        u = min(1.0, k / flight)
        s = 1 - (1 - u) ** 1.7                      # decelerates in the image
        ground_y = ay - (ay - (horizon + 8)) * (1 - (1 - s) ** 2.2)
        height = h * 0.62 * math.sin(math.pi * min(1.0, s * 1.05)) * (1 - 0.55 * s)
        x = ax + w * (0.06 * s + 0.11 * s * s)      # gentle fade
        return x, ground_y - max(0.0, height), max(1.1, r0 * (1 - 0.85 * s))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    truth = {"fps": fps, "size": [w, h], "launch": launch, "address": [ax, ay], "frames": []}
    pivot = np.array([w * 0.39, h * 0.52])
    off = np.zeros(2)
    noise_bank = [rng.normal(0, noise, (h, w, 3)).astype(np.float32) for _ in range(7)]
    for f in range(n_frames):
        img = bg.copy()
        # golfer: body + swinging arms/club with a white glove
        cv2.rectangle(img, (int(w * 0.36), int(h * 0.48)), (int(w * 0.42), int(h * 0.86)), (70, 40, 30), -1)
        cv2.circle(img, (int(w * 0.39), int(h * 0.44)), int(h * 0.04), (120, 160, 200), -1, cv2.LINE_AA)
        sw = (f - launch) / fps
        ang = math.radians(np.interp(sw, [-1.0, -0.35, 0.0, 0.35, 0.8], [110, 200, 95, -20, -40]))
        hand = pivot + np.array([math.cos(ang), math.sin(ang)]) * h * 0.2
        head = pivot + np.array([math.cos(ang), math.sin(ang)]) * h * 0.42
        cv2.line(img, tuple(pivot.astype(int)), tuple(hand.astype(int)), (70, 40, 30), 12, cv2.LINE_AA)
        cv2.line(img, tuple(hand.astype(int)), tuple(head.astype(int)), (160, 160, 160), 3, cv2.LINE_AA)
        cv2.circle(img, tuple(head.astype(int)), 9, (60, 60, 60), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(hand.astype(int)), 9, (250, 250, 250), -1, cv2.LINE_AA)  # white glove

        if f < launch:
            bx, by, br = ax, ay, r0
            cv2.circle(img, (int(bx * 16), int(by * 16)), int(br * 16), ball_bgr, -1, cv2.LINE_AA, 4)
        else:
            k = f - launch
            if k <= flight:
                bx, by, br = ball_pos(k)
                # motion blur: exposure covers 40% of the frame interval centred on frame time
                amask = np.zeros(img.shape[:2], np.float32)
                subs = 6
                for i in range(subs):
                    kk = max(0.0, k + 0.4 * (i / (subs - 1) - 0.5))
                    sx, sy, _ = ball_pos(kk)
                    sub = np.zeros_like(amask)
                    cv2.circle(sub, (int(sx * 16), int(sy * 16)), int(br * 16), 1.0, -1, cv2.LINE_AA, 4)
                    amask += sub / subs
                amask = np.clip(amask, 0, 1)[..., None]
                img = img * (1 - amask) + np.float32(ball_bgr) * amask
            else:
                bx = None

        off = 0.8 * off + rng.normal(0, shake * 0.6, 2) if shake else off
        m = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        img = np.clip(img + noise_bank[f % len(noise_bank)], 0, 255).astype(np.uint8)
        writer.write(img)
        if f >= launch and bx is not None:
            truth["frames"].append([f, bx + off[0], by + off[1]])
    writer.release()
    Path(out_path).with_suffix(".json").write_text(json.dumps(truth))
    return truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", default="white", choices=sorted(BALL_BGR))
    ap.add_argument("--out", default="samples/test_shot.mp4")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--shake", type=float, default=1.5)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    make_video(a.out, a.color, fps=a.fps, shake=a.shake)
    print("wrote", a.out)
