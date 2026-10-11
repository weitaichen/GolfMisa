from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal

from ..analysis import Cancelled


class Worker(QObject):
    """Runs fn(*args, progress=..., cancelled=..., **kw) on a background thread."""

    progress = Signal(int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            r = self.fn(*self.args, progress=self.progress.emit, cancelled=lambda: self._cancel, **self.kwargs)
            self.done.emit(r)
        except Cancelled:
            self.failed.emit("Cancelled")
        except Exception as e:  # surfaced to the user in a dialog
            traceback.print_exc()
            self.failed.emit(str(e))
        finally:
            # quit from inside the thread: a queued quit would deadlock a GUI thread
            # that is blocked in QThread.wait() (e.g. while the window is closing)
            QThread.currentThread().quit()


def start(worker: Worker, parent) -> QThread:
    th = QThread(parent)
    worker.moveToThread(th)
    th.started.connect(worker.run)
    worker.done.connect(th.quit)
    worker.failed.connect(th.quit)
    th.finished.connect(th.deleteLater)
    th.finished.connect(worker.deleteLater)
    th.start()
    return th
