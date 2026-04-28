from PyQt5.QtCore import Qt, QRunnable, QThreadPool, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow,
    QVBoxLayout, QWidget, QFrame,
)

from app import config
from app.gui.camera_widget   import CameraWidget
from app.gui.emotion_widget  import EmotionWidget
from app.gui.sentence_widget import SentenceWidget
from app.gui.model_selector  import ModelSelectorWidget
from app.gui.theme           import GLOBAL_STYLESHEET, BG, SURFACE, BORDER, ACCENT, TEXT_MUTE, CARD
from app.sentence.builder    import SentenceBuilder
from app.tts.elevenlabs_client import TTSClient
from app.workers.camera_worker  import CameraWorker
from app.workers.asl_worker     import ASLWorker
from app.workers.emotion_worker import EmotionWorker


# Runs TTS on the global QThreadPool so the GUI never blocks while audio streams.
class _SpeakRunnable(QRunnable):
    def __init__(self, client: TTSClient, text: str, emotion: str):
        super().__init__()
        self._client  = client
        self._text    = text
        self._emotion = emotion

    def run(self):
        self._client.speak(self._text, emotion=self._emotion)


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"background: {BORDER}; border: none; max-height: 1px;")
    return line


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EmoSign")
        self.setMinimumSize(720, 520)

        QApplication.instance().setStyleSheet(GLOBAL_STYLESHEET)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }}")

        self._sentence_builder = SentenceBuilder()
        self._tts_client       = TTSClient()
        self._thread_pool      = QThreadPool.globalInstance()

        self._current_letter  = ""
        self._current_conf    = 0.0
        self._current_emotion = "neutral"

        first_key  = list(config.ASL_MODELS.keys())[0]
        first_info = config.ASL_MODELS[first_key]

        self._camera_worker  = CameraWorker()
        self._asl_worker     = ASLWorker(first_info["path"], head=first_info["head"])
        # Emotion worker is optional — skip silently if the weights file is absent
        # so the app is usable before the emotion model has been trained.
        self._emotion_worker = (
            EmotionWorker(config.EMOTION_MODEL_PATH)
            if config.EMOTION_MODEL_PATH.exists() else None
        )

        self._camera_widget   = CameraWidget()
        self._emotion_widget  = EmotionWidget()
        self._sentence_widget = SentenceWidget()
        self._model_selector  = ModelSelectorWidget()

        self._build_layout()
        self._connect_signals()
        self._start_workers()

    # ── Layout ──────────────────────────────────────────────────────────────

    def _build_layout(self):
        root_widget = QWidget()
        root_widget.setStyleSheet(f"background: {BG};")
        self.setCentralWidget(root_widget)

        root = QVBoxLayout(root_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addWidget(_divider())

        content = QWidget()
        content.setStyleSheet(f"background: {BG};")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(16)

        # Camera — left, fills available space
        content_layout.addWidget(self._camera_widget, stretch=3)

        # Right panel — emotion + letter/sentence stacked, min width so it never collapses
        right_widget = QWidget()
        right_widget.setMinimumWidth(240)
        right_widget.setMaximumWidth(340)
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(12)
        right.addWidget(self._emotion_widget)
        right.addWidget(self._sentence_widget, stretch=1)
        content_layout.addWidget(right_widget, stretch=1)

        root.addWidget(content, stretch=1)
        root.addWidget(_divider())
        root.addWidget(self._build_statusbar())

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(64)
        header.setStyleSheet(f"background: {SURFACE}; border-bottom: 1px solid {BORDER};")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)

        # Brand
        brand_col = QVBoxLayout()
        brand_col.setSpacing(0)
        title = QLabel("EmoSign")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT}; background: transparent; letter-spacing: 1px;")
        sub = QLabel("Sign Language · Emotion-Aware Speech")
        sub.setStyleSheet(f"color: {TEXT_MUTE}; font-size: 11px; background: transparent;")
        brand_col.addWidget(title)
        brand_col.addWidget(sub)
        layout.addLayout(brand_col)

        layout.addStretch()
        layout.addWidget(self._model_selector)

        return header

    def _build_statusbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(28)
        bar.setStyleSheet(f"background: {CARD}; border-top: 1px solid {BORDER};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)
        tip = QLabel("Position your hand inside the frame box  ·  Hold an expression, then press Speak")
        tip.setStyleSheet(f"color: {TEXT_MUTE}; font-size: 11px; background: transparent;")
        layout.addWidget(tip)
        layout.addStretch()
        return bar

    # ── Signals ─────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self._camera_worker.frame_ready.connect(self._camera_widget.update_frame)
        self._camera_worker.frame_ready.connect(self._asl_worker.on_frame)
        if self._emotion_worker:
            self._camera_worker.frame_ready.connect(self._emotion_worker.on_frame)

        self._asl_worker.prediction.connect(self._camera_widget.set_asl_result)
        self._asl_worker.prediction.connect(self._on_asl_prediction)

        if self._emotion_worker:
            self._emotion_worker.prediction.connect(self._emotion_widget.update_emotion)
            self._emotion_worker.prediction.connect(self._on_emotion_prediction)

        self._model_selector.model_changed.connect(self._asl_worker.load_model)

        self._sentence_widget.add_letter_requested.connect(self._on_add_letter)
        self._sentence_widget.space_requested.connect(self._on_add_space)
        self._sentence_widget.backspace_requested.connect(self._on_backspace)
        self._sentence_widget.speak_requested.connect(self._on_speak)
        self._sentence_widget.clear_requested.connect(self._on_clear)

    # ── Slots ────────────────────────────────────────────────────────────────

    @pyqtSlot(str, float)
    def _on_asl_prediction(self, letter: str, conf: float):
        self._current_letter = letter
        self._current_conf   = conf
        self._sentence_widget.update_pending(letter, conf)

    @pyqtSlot()
    def _on_add_letter(self):
        # Manual "Add Letter" button bypasses the debounce — user explicitly clicked.
        letter = self._current_letter
        if not letter or letter in ("nothing", "unknown"):
            return
        if letter == "space":
            self._sentence_builder._chars.append(" ")
        elif letter == "del":
            if self._sentence_builder._chars:
                self._sentence_builder._chars.pop()
        else:
            self._sentence_builder._chars.append(letter.upper())
        self._sentence_widget.update_sentence(self._sentence_builder.get_sentence())

    @pyqtSlot()
    def _on_add_space(self):
        self._sentence_builder._chars.append(" ")
        self._sentence_widget.update_sentence(self._sentence_builder.get_sentence())

    @pyqtSlot()
    def _on_backspace(self):
        if self._sentence_builder._chars:
            self._sentence_builder._chars.pop()
        self._sentence_widget.update_sentence(self._sentence_builder.get_sentence())

    @pyqtSlot(str, dict)
    def _on_emotion_prediction(self, emotion: str, scores: dict):
        self._current_emotion = emotion

    @pyqtSlot(str)
    def _on_speak(self, text: str):
        self._thread_pool.start(_SpeakRunnable(self._tts_client, text, self._current_emotion))

    @pyqtSlot()
    def _on_clear(self):
        self._sentence_builder.clear()
        self._sentence_widget.update_sentence("")

    # ── Workers ──────────────────────────────────────────────────────────────

    def _start_workers(self):
        self._asl_worker.start()
        if self._emotion_worker:
            self._emotion_worker.start()
        self._camera_worker.start()

    def closeEvent(self, event):
        self._camera_worker.stop()
        self._asl_worker.stop()
        if self._emotion_worker:
            self._emotion_worker.stop()
        self._thread_pool.waitForDone(3000)
        event.accept()
