"""Video reading with a display-frame cache, and exporting with the tracer burned in."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections import OrderedDict

import cv2
import numpy as np

from .analysis import Cancelled
from .path import TracerPath
from .render import TracerStyle, draw_tracer


def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.exists(b) and os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _count_frames(cap) -> int:
    """Count frames by decoding headers only (for containers that don't report a length)."""
    n = 0
    while cap.grab():
        n += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return n


class VideoSource:
    """Random-access frame reader; sequential reads avoid seeking."""

    def __init__(self, path: str, display_max: int = 1280, cache_mb: int = 600):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 <= fps <= 1000 else 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width <= 0 or self.height <= 0:
            self.cap.release()
            raise IOError(f"Video has no readable frames: {path}")
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.count <= 0:
            self.count = _count_frames(self.cap)
            if self.count <= 0:
                self.cap.release()
                raise IOError(f"Video has no readable frames: {path}")
        self.display_scale = min(1.0, display_max / max(self.width, self.height))
        self.display_size = (max(1, int(round(self.width * self.display_scale))),
                             max(1, int(round(self.height * self.display_scale))))
        self._next = 0
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._budget = cache_mb * 1024 * 1024
        self._frame_bytes = self.display_size[0] * self.display_size[1] * 3
        self._last_full: tuple[int, np.ndarray] | None = None

    def read_full(self, idx: int) -> np.ndarray | None:
        if idx < 0:
            return None
        if self._last_full is not None and self._last_full[0] == idx:
            return self._last_full[1]
        sequential = idx == self._next
        if not sequential:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            self._next = -1                # position unknown after a failed read
            if sequential and idx < self.count:
                self.count = idx           # container over-reported its length
            return None
        self._next = idx + 1
        if idx >= self.count:
            self.count = idx + 1           # container under-reported its length
        self._last_full = (idx, frame)
        return frame

    def read_display(self, idx: int) -> np.ndarray | None:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        frame = self.read_full(idx)
        if frame is None:
            return None
        if self.display_scale < 1:
            frame = cv2.resize(frame, self.display_size, interpolation=cv2.INTER_AREA)
        self._cache[idx] = frame
        while len(self._cache) * self._frame_bytes > self._budget:
            self._cache.popitem(last=False)
        return frame

    def release(self):
        self.cap.release()


def _even(frame: np.ndarray) -> np.ndarray:
    """Pad to even width/height (H.264 / yuv420p and many mp4 writers require it)."""
    h, w = frame.shape[:2]
    if h % 2 == 0 and w % 2 == 0:
        return frame
    return cv2.copyMakeBorder(frame, 0, h % 2, 0, w % 2, cv2.BORDER_REPLICATE)


def _remove(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


def export_video(src_path: str, path: TracerPath, style: TracerStyle, out_path: str,
                 progress=None, cancelled=None) -> str:
    """Render the clip with the tracer. Re-encodes to H.264 + original audio when ffmpeg exists."""
    if _same_file(src_path, out_path):
        raise ValueError("Choose a different file name - the export would overwrite the source video.")
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 1 <= fps <= 1000 else 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    ffmpeg = shutil.which("ffmpeg")
    target = out_path
    if ffmpeg:
        fd, target = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
    size = (w + w % 2, h + h % 2)
    writer = cv2.VideoWriter(target, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    idx = 0
    try:
        if not writer.isOpened():
            raise IOError(f"Cannot write video: {out_path}")
        while True:
            if cancelled and cancelled():
                raise Cancelled()
            ok, frame = cap.read()
            if not ok:
                break
            pts = path.until(idx)
            if len(pts):
                draw_tracer(frame, pts, style, head_active=idx <= path.end)
            writer.write(_even(frame))
            idx += 1
            if progress and idx % 10 == 0:
                progress(min(95, int(95 * idx / count)))
        writer.release()
        if idx == 0:
            raise IOError("No frames could be read from the source video.")
        if ffmpeg:
            _encode(ffmpeg, target, src_path, out_path, idx / fps, cancelled)
    except BaseException:
        cap.release()
        writer.release()                        # Windows can't delete a file still open for writing
        if target != out_path:
            _remove(target)
        _remove(out_path)                       # never leave a partial export behind
        raise
    finally:
        cap.release()
        writer.release()
    if target != out_path:
        _remove(target)
    if progress:
        progress(100)
    return out_path


def _encode(ffmpeg: str, video: str, src_path: str, out_path: str, duration: float, cancelled):
    """H.264 re-encode with the source's audio; cut to the video's length (not -shortest,
    which would truncate the video when the audio track is shorter)."""
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", video, "-i", src_path,
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-preset", "veryfast", "-c:a", "aac", "-t", f"{duration:.3f}", out_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    while True:
        try:
            _, err = proc.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if cancelled and cancelled():
                proc.kill()
                proc.communicate()
                raise Cancelled()
    if proc.returncode != 0:
        raise IOError("ffmpeg failed to encode the export: " + err.decode(errors="replace")[-400:])
