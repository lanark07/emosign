# Global colour palette and Qt stylesheet.
# Import colour tokens (BG, ACCENT, etc.) directly into widget files — don't
# hardcode hex strings elsewhere so a theme change only touches this file.
BG        = "#f7f7fc"
SURFACE   = "#ffffff"
CARD      = "#f0f0f8"
BORDER    = "#e2e2ee"
BORDER_HI = "#7c3aed"

ACCENT    = "#7c3aed"   # purple
ACCENT2   = "#0891b2"   # cyan
SUCCESS   = "#059669"   # green
WARNING   = "#d97706"   # amber
DANGER    = "#dc2626"   # red

TEXT      = "#1e1b4b"
TEXT_SUB  = "#4b5563"
TEXT_MUTE = "#9ca3af"

GLOBAL_STYLESHEET = f"""
QMainWindow, QDialog {{
    background: {BG};
}}
QWidget {{
    background: transparent;
    color: {TEXT};
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}}
QLabel {{
    background: transparent;
    color: {TEXT};
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    background: {SURFACE};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: {TEXT_MUTE};
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
    background: {SURFACE};
}}
QComboBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 7px 12px;
    color: {TEXT};
    min-width: 180px;
}}
QComboBox:hover {{
    border-color: {ACCENT};
}}
QComboBox:focus {{
    border-color: {ACCENT};
    outline: none;
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
    padding-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    selection-background-color: {ACCENT};
    selection-color: white;
    padding: 4px;
    outline: none;
}}
QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 8px 18px;
    color: {TEXT};
    font-size: 13px;
}}
QPushButton:hover {{
    background: {CARD};
    border-color: {ACCENT};
    color: {ACCENT};
}}
QPushButton:pressed {{
    background: {ACCENT};
    border-color: {ACCENT};
    color: white;
}}
QPushButton:disabled {{
    background: {CARD};
    border-color: {BORDER};
    color: {TEXT_MUTE};
}}
QProgressBar {{
    border-radius: 4px;
    background: {CARD};
    text-align: center;
    color: transparent;
    border: none;
}}
QScrollBar:vertical {{
    background: {CARD};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 3px;
    min-height: 20px;
}}
"""
