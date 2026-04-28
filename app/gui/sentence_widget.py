from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from app.gui.theme import (
    CARD, SURFACE, BORDER, ACCENT,
    SUCCESS, DANGER, WARNING,
    TEXT, TEXT_SUB, TEXT_MUTE,
)


def _btn(text: str, color: str, hover: str, height: int = 38) -> QPushButton:
    b = QPushButton(text)
    b.setFixedHeight(height)
    b.setStyleSheet(
        f"QPushButton {{ background: {color}; color: white; border: none;"
        f"border-radius: 7px; font-size: 13px; font-weight: 600; padding: 0 18px; }}"
        f"QPushButton:hover {{ background: {hover}; }}"
        f"QPushButton:pressed {{ background: {color}; opacity: 0.8; }}"
        f"QPushButton:disabled {{ background: {SURFACE}; color: {TEXT_MUTE}; }}"
    )
    return b


# Right-panel widget: shows the currently detected letter + confidence, lets
# the user manually add it, and displays/edits the accumulated sentence.
# All mutations are handled by MainWindow via signals — this widget holds no state.
class SentenceWidget(QWidget):
    speak_requested      = pyqtSignal(str)
    clear_requested      = pyqtSignal()
    add_letter_requested = pyqtSignal()
    backspace_requested  = pyqtSignal()
    space_requested      = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        # ── Current letter card ──────────────────────────────────────────────
        letter_group = QGroupBox("Detected Letter")
        letter_inner = QVBoxLayout(letter_group)
        letter_inner.setContentsMargins(12, 16, 12, 12)
        letter_inner.setSpacing(8)

        self._letter_label = QLabel("—")
        self._letter_label.setAlignment(Qt.AlignCenter)
        self._letter_label.setFont(QFont("Segoe UI", 52, QFont.Bold))
        self._letter_label.setMinimumHeight(80)
        self._letter_label.setWordWrap(True)
        self._letter_label.setStyleSheet(f"color: {TEXT_MUTE}; background: transparent; padding: 4px;")
        letter_inner.addWidget(self._letter_label)

        self._conf_label = QLabel("")
        self._conf_label.setAlignment(Qt.AlignCenter)
        self._conf_label.setStyleSheet(f"color: {TEXT_MUTE}; font-size: 11px; background: transparent;")
        letter_inner.addWidget(self._conf_label)

        self._add_btn = _btn("✓  Add Letter", ACCENT, "#6d28d9", height=44)
        self._add_btn.setEnabled(False)
        letter_inner.addWidget(self._add_btn)

        outer.addWidget(letter_group)

        # ── Sentence card ────────────────────────────────────────────────────
        sentence_group = QGroupBox("Sentence")
        sentence_inner = QVBoxLayout(sentence_group)
        sentence_inner.setContentsMargins(12, 16, 12, 12)
        sentence_inner.setSpacing(10)

        self._sentence_label = QLabel("…")
        self._sentence_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._sentence_label.setFont(QFont("Segoe UI", 18))
        self._sentence_label.setStyleSheet(
            f"color: {TEXT}; background: {SURFACE}; border-radius: 8px;"
            f"padding: 10px 14px; border: 1px solid {BORDER};"
        )
        self._sentence_label.setMinimumHeight(56)
        self._sentence_label.setWordWrap(True)
        sentence_inner.addWidget(self._sentence_label)

        # Edit buttons row
        edit_row = QHBoxLayout()
        edit_row.setSpacing(8)
        self._space_btn     = QPushButton("Space")
        self._backspace_btn = QPushButton("⌫  Delete")
        self._space_btn.setFixedHeight(34)
        self._backspace_btn.setFixedHeight(34)
        edit_row.addWidget(self._space_btn)
        edit_row.addWidget(self._backspace_btn)
        edit_row.addStretch()
        sentence_inner.addLayout(edit_row)

        # Action buttons row
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self._speak_btn = _btn("🔊  Speak", "#1d4ed8", "#1e40af", height=40)
        self._clear_btn = _btn("Clear", "#991b1b", "#7f1d1d", height=40)
        action_row.addWidget(self._speak_btn, stretch=1)
        action_row.addWidget(self._clear_btn)
        sentence_inner.addLayout(action_row)

        outer.addWidget(sentence_group)

        # ── Wire ────────────────────────────────────────────────────────────
        self._add_btn.clicked.connect(self.add_letter_requested)
        self._space_btn.clicked.connect(self.space_requested)
        self._backspace_btn.clicked.connect(self.backspace_requested)
        self._speak_btn.clicked.connect(self._on_speak)
        self._clear_btn.clicked.connect(self.clear_requested)

    # ── Public ───────────────────────────────────────────────────────────────

    def update_pending(self, letter: str, conf: float):
        if letter and letter not in ("nothing", "unknown", ""):
            display = letter.upper()
            font_size = 52 if len(display) == 1 else 28
            self._letter_label.setFont(QFont("Segoe UI", font_size, QFont.Bold))
            self._letter_label.setText(display)
            if conf >= 0.60:
                color = ACCENT
            elif conf >= 0.40:
                color = WARNING
            else:
                color = TEXT_MUTE
            self._letter_label.setStyleSheet(
                f"color: {color}; background: transparent; padding: 4px;"
            )
            self._conf_label.setText(f"Confidence  {conf*100:.0f}%")
            self._add_btn.setEnabled(conf >= 0.40)
        else:
            self._letter_label.setFont(QFont("Segoe UI", 52, QFont.Bold))
            self._letter_label.setText("—")
            self._letter_label.setStyleSheet(f"color: {TEXT_MUTE}; background: transparent; padding: 4px;")
            self._conf_label.setText("")
            self._add_btn.setEnabled(False)

    def update_sentence(self, text: str):
        self._sentence_label.setText(text if text else "…")

    def _on_speak(self):
        text = self._sentence_label.text()
        if text and text != "…":
            self.speak_requested.emit(text)
