ACCENT = "#36e27b"

QSS = f"""
* {{ font-family: "Segoe UI", "Inter", sans-serif; font-size: 13px; color: #e8eaed; }}
QMainWindow, QWidget#Root {{ background: #0d0f13; }}
QWidget#SidePanel {{ background: #14171d; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QWidget#TopBar, QWidget#Transport {{ background: #14171d; }}
QLabel#Brand {{ font-size: 17px; font-weight: 700; letter-spacing: 0.5px; }}
QLabel#BrandAccent {{ font-size: 17px; font-weight: 700; color: {ACCENT}; }}
QLabel#SectionTitle {{ color: #7d8696; font-size: 11px; font-weight: 700; letter-spacing: 1.2px; }}
QLabel#Muted {{ color: #8a93a3; font-size: 12px; }}
QLabel#Stat {{ font-size: 20px; font-weight: 700; }}
QLabel#StatLabel {{ color: #7d8696; font-size: 11px; }}
QLabel#Time {{ font-family: "Consolas", monospace; color: #c7ccd4; }}
QFrame#Card {{ background: #1a1e25; border: 1px solid #242a33; border-radius: 12px; }}
QFrame#StatCard {{ background: #1f242c; border-radius: 10px; }}

QPushButton {{
    background: #232831; border: 1px solid #2e3541; border-radius: 9px; padding: 7px 12px;
}}
QPushButton:hover {{ background: #2b313c; }}
QPushButton:pressed {{ background: #1c2028; }}
QPushButton:disabled {{ color: #5b6270; background: #1b1f26; border-color: #232831; }}
QPushButton:checked {{ border: 1px solid {ACCENT}; color: {ACCENT}; background: #1d2a23; }}
QPushButton#Primary {{
    background: {ACCENT}; color: #08110c; font-weight: 700; border: none; padding: 10px 14px; font-size: 14px;
}}
QPushButton#Primary:hover {{ background: #4cf08d; }}
QPushButton#Primary:disabled {{ background: #23422f; color: #5c7a66; }}
QPushButton#Record:checked {{ background: #e5484d; color: white; border: none; }}
QPushButton#Seg {{ border-radius: 7px; padding: 6px 4px; }}
QPushButton#Icon {{ padding: 6px; min-width: 34px; font-size: 15px; }}
QPushButton#PlayBtn {{
    background: {ACCENT}; color: #08110c; border: none; border-radius: 18px; min-width: 36px; min-height: 36px;
    font-size: 16px; font-weight: 700;
}}

QSlider::groove:horizontal {{ height: 4px; background: #2b313c; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: white; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QComboBox {{ background: #232831; border: 1px solid #2e3541; border-radius: 8px; padding: 5px 8px; }}
QComboBox QAbstractItemView {{ background: #1a1e25; selection-background-color: #2b313c; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #3a4250; background: #1a1e25; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QProgressBar {{ background: #232831; border: none; border-radius: 4px; height: 8px; text-align: center; font-size: 10px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QToolTip {{ background: #232831; color: #e8eaed; border: 1px solid #3a4250; padding: 4px; }}
QStatusBar {{ background: #0d0f13; color: #8a93a3; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: #2b313c; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""
