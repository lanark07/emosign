from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from app import config
from app.gui.theme import TEXT_MUTE, SURFACE, BORDER, ACCENT


# Dropdown in the header bar. Emits model_changed(key) which ASLWorker.load_model()
# handles live without restarting the inference thread.
class ModelSelectorWidget(QWidget):
    model_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        lbl = QLabel("Model")
        lbl.setStyleSheet(f"color: {TEXT_MUTE}; font-size: 11px; background: transparent;")
        layout.addWidget(lbl)

        self._combo = QComboBox()
        self._combo.setFixedHeight(34)
        for name, info in config.ASL_MODELS.items():
            exists  = info["path"].exists()
            display = name if exists else f"{name}  [not trained]"
            self._combo.addItem(display, userData=name)
            if not exists:
                self._combo.model().item(self._combo.count() - 1).setEnabled(False)

        self._combo.currentIndexChanged.connect(self._on_changed)
        layout.addWidget(self._combo)

    def _on_changed(self, _index: int):
        key = self._combo.currentData()
        if key:
            self.model_changed.emit(key)

    def current_model_key(self) -> str:
        return self._combo.currentData() or ""
