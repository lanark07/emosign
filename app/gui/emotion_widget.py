from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QGroupBox, QLabel, QProgressBar, QVBoxLayout, QWidget,
)

from app import config
from app.gui.theme import CARD, SURFACE, BORDER, TEXT, TEXT_SUB, TEXT_MUTE

# Per-emotion accent colours (foreground, dark variant).
# Keep in sync with the palette in theme.py if you change colours there.
_COLORS = {
    "happy":   ("#10b981", "#064e3b"),   # green
    "neutral": ("#06b6d4", "#0c4a6e"),   # cyan
    "angry":   ("#ef4444", "#7f1d1d"),   # red
    "sad":     ("#8b5cf6", "#3b0764"),   # purple
}
_ICONS = {"happy": "😊", "neutral": "😐", "angry": "😠", "sad": "😢"}


class EmotionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._bars: dict[str, QProgressBar] = {}
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Emotion Detection")
        inner = QVBoxLayout(group)
        inner.setSpacing(10)
        inner.setContentsMargins(12, 16, 12, 12)

        # Big emotion display
        self._icon_label = QLabel("—")
        self._icon_label.setAlignment(Qt.AlignCenter)
        self._icon_label.setFont(QFont("Segoe UI Emoji", 36))
        self._icon_label.setFixedHeight(56)
        self._icon_label.setStyleSheet(f"color: {TEXT}; background: transparent;")
        inner.addWidget(self._icon_label)

        self._name_label = QLabel("Detecting…")
        self._name_label.setAlignment(Qt.AlignCenter)
        self._name_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self._name_label.setStyleSheet(f"color: {TEXT_SUB}; background: transparent;")
        inner.addWidget(self._name_label)

        # Divider
        div = QWidget()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {BORDER};")
        inner.addWidget(div)

        # Per-class bars
        for cls in config.EMOTION_CLASSES:
            fg, _ = _COLORS.get(cls, ("#aaa", "#222"))
            icon  = _ICONS.get(cls, "")

            row = QWidget()
            row.setStyleSheet("background: transparent;")
            row_l = QVBoxLayout(row)
            row_l.setContentsMargins(0, 2, 0, 2)
            row_l.setSpacing(3)

            lbl = QLabel(f"{icon}  {cls.capitalize()}")
            lbl.setStyleSheet(f"color: {TEXT_MUTE}; font-size: 11px; background: transparent;")

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            bar.setStyleSheet(
                f"QProgressBar {{ border-radius: 3px; background: {SURFACE}; border: none; }}"
                f"QProgressBar::chunk {{ background: {fg}; border-radius: 3px; }}"
            )

            row_l.addWidget(lbl)
            row_l.addWidget(bar)
            inner.addWidget(row)
            self._bars[cls] = bar

        outer.addWidget(group)

    def update_emotion(self, emotion: str, scores: dict):
        fg, bg = _COLORS.get(emotion, ("#aaa", "#222"))
        icon   = _ICONS.get(emotion, "")

        self._icon_label.setText(icon)
        self._name_label.setText(emotion.capitalize())
        self._name_label.setStyleSheet(
            f"color: {fg}; font-size: 14px; font-weight: bold; background: transparent;"
        )

        for cls, bar in self._bars.items():
            bar.setValue(int(scores.get(cls, 0.0) * 100))
            bar_fg, _ = _COLORS.get(cls, ("#aaa", "#222"))
            active = (cls == emotion)
            bar.setStyleSheet(
                f"QProgressBar {{ border-radius: 3px; background: {SURFACE}; border: none; }}"
                f"QProgressBar::chunk {{ background: {bar_fg}; border-radius: 3px; "
                f"{'opacity: 1;' if active else 'opacity: 0.4;'} }}"
            )
