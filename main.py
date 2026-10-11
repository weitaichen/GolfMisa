"""ShotTracer - golf shot tracer for any ball colour.

    python main.py [video.mp4]
"""
import sys

from PySide6.QtWidgets import QApplication

from golf_tracer.ui.main_window import MainWindow
from golf_tracer.ui.style import QSS


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ShotTracer")
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win.open_video(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
