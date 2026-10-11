"""Live camera with capture and ball tracking on background threads.

    capture thread   reads every camera frame, writes it to the recording (so a
                     recording always has the camera's full frame rate) and
                     publishes it as the latest frame
    tracking thread  runs ball tracking on the newest frame; when it falls behind it
                     skips frames instead of slowing capture down
    GUI thread       calls latest() on a timer and only draws

No Qt here: the GUI polls, so no widgets are touched from another thread.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from .analysis import LiveTracker
from .colors import BallColor
from .path import TracerPath


def _rate(times) -> float:
    """Frames per second from a list of timestamps (0 when not enough data)."""
    if len(times) < 10 or times[-1] <= times[0]:
        return 0.0
    return (len(times) - 1) / (times[-1] - times[0])


class CameraSession:
    def __init__(self, color: BallColor, sensitivity: float, index: int = 0,
                 size: tuple[int, int] = (1280, 720), fps: float = 60.0):
        self.index, self.size, self.requested_fps = index, size, fps
        self.reported_fps = 30.0
        self.error: str | None = None

        self._lock = threading.Lock()                # latest frame / path / tracker settings
        self._new_frame = threading.Condition(self._lock)
        self._frame: np.ndarray | None = None
        self._seq = -1                               # capture frame number of _frame
        self._path: TracerPath | None = None
        self._cap_times: deque[float] = deque(maxlen=120)
        self._track_times: deque[float] = deque(maxlen=60)

        self._settings = (color, sensitivity)
        self._settings_changed = True

        self._rec_lock = threading.Lock()            # writer is shared with the capture thread
        self._writer = None
        self._rec_frames = 0
        self._rec_t0 = 0.0
        self.record_fps = 0.0

        self._cap = None
        self._running = False
        self._threads: list[threading.Thread] = []

    # ---------------------------------------------------------------- lifecycle
    def open(self) -> bool:
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(self.index)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.reported_fps = fps if fps and 1 <= fps <= 240 else 30.0
        self._cap = cap
        self._running = True
        self._threads = [threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True),
                         threading.Thread(target=self._track_loop, name="camera-tracking", daemon=True)]
        for t in self._threads:
            t.start()
        return True

    def close(self):
        self._running = False
        with self._lock:
            self._new_frame.notify_all()
        for t in self._threads:
            t.join(timeout=3.0)
        self.stop_recording()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def configure(self, color: BallColor, sensitivity: float):
        """New ball colour / sensitivity: the tracking thread rebuilds its tracker."""
        with self._lock:
            self._settings = (color, sensitivity)
            self._settings_changed = True
            self._path = None

    # ------------------------------------------------------------------ status
    def latest(self) -> tuple[np.ndarray | None, int, TracerPath | None]:
        with self._lock:
            return self._frame, self._seq, self._path

    @property
    def camera_fps(self) -> float:
        """Rate frames actually arrive at (cameras often don't deliver what they report)."""
        with self._lock:
            return _rate(list(self._cap_times))

    @property
    def tracking_fps(self) -> float:
        with self._lock:
            return _rate(list(self._track_times))

    @property
    def recording(self) -> bool:
        return self._writer is not None

    @property
    def recording_seconds(self) -> float:
        return time.perf_counter() - self._rec_t0 if self._writer is not None else 0.0

    # --------------------------------------------------------------- recording
    def start_recording(self, path: str) -> str | None:
        """Start writing every captured frame to `path`. Returns an error message or None."""
        frame, _, _ = self.latest()
        if frame is None:
            return "The camera is not delivering frames yet."
        fps = self.camera_fps or self.reported_fps
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            return f"Cannot write {path}"
        with self._rec_lock:
            self._writer = writer
            self._rec_frames = 0
            self._rec_t0 = time.perf_counter()
            self.record_fps = fps
        return None

    def stop_recording(self) -> tuple[int, float]:
        """Finish the file. Returns (frames written, seconds recorded)."""
        with self._rec_lock:
            writer, self._writer = self._writer, None
            frames, seconds = self._rec_frames, time.perf_counter() - self._rec_t0
        if writer is None:
            return 0, 0.0
        writer.release()
        return frames, seconds

    # ----------------------------------------------------------------- threads
    def _capture_loop(self):
        failures = 0
        while self._running:
            ok, frame = self._cap.read()           # blocks until the camera has a frame
            if not ok:
                failures += 1
                if failures > 100:
                    self.error = "The camera stopped delivering frames."
                    break
                time.sleep(0.01)
                continue
            failures = 0
            with self._rec_lock:
                if self._writer is not None:
                    self._writer.write(frame)
                    self._rec_frames += 1
            with self._lock:
                self._cap_times.append(time.perf_counter())
                self._frame = frame
                self._seq += 1
                self._new_frame.notify()
        self._running = False
        with self._lock:
            self._new_frame.notify_all()

    def _track_loop(self):
        tracker: LiveTracker | None = None
        tracker_fps = 0.0
        last_seq = -1
        while self._running:
            with self._lock:
                while self._running and self._seq == last_seq:
                    self._new_frame.wait(0.2)
                if not self._running:
                    break
                frame, seq = self._frame, self._seq     # newest frame; older ones are skipped
                changed, (color, sens) = self._settings_changed, self._settings
                self._settings_changed = False
                fps = _rate(list(self._cap_times))
            # rebuild when settings change, and once the real camera rate is known
            fps_known = fps > 0 and abs(fps - tracker_fps) / max(tracker_fps, 1e-6) > 0.15
            if tracker is None or changed or fps_known:
                tracker_fps = fps or self.reported_fps
                tracker = LiveTracker(color, sens, tracker_fps)
            path = tracker.process(frame, seq)          # capture numbers keep timing right when skipping
            last_seq = seq
            with self._lock:
                if not self._settings_changed:
                    self._path = path
                self._track_times.append(time.perf_counter())
