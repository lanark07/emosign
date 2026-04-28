import queue
import threading
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from app.inference.asl_predictor import ASLPredictor
from app import config


# Runs ASL inference on a background thread.
# maxsize=1 queue: if inference is slower than the camera, old frames are silently
# dropped instead of accumulating — keeps latency low at the cost of some frames.
class ASLWorker(QThread):
    prediction = pyqtSignal(str, float)   # letter, confidence

    def __init__(self, model_path: Path, head: str = "standard", parent=None):
        super().__init__(parent)
        self._queue      = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._lock       = threading.Lock()   # guards _predictor during hot-swap
        self._predictor  = ASLPredictor()
        self._predictor.load(model_path, head=head)

    # --- public slots ---

    def on_frame(self, frame: np.ndarray):
        """Called from camera worker signal — drops stale frames."""
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def load_model(self, model_key: str):
        """Hot-swap ASL model weights without restarting the thread.
        The lock prevents the run() loop from calling predict() mid-load."""
        info = config.ASL_MODELS.get(model_key)
        if info is None or not info["path"].exists():
            print(f"ASLWorker: model not found for key '{model_key}'")
            return
        with self._lock:
            self._predictor.load(info["path"], head=info["head"])

    # --- thread loop ---

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            h, w = frame.shape[:2]
            half = config.ROI_SIZE // 2
            cx, cy = w // 2, h // 2
            roi = frame[cy - half:cy + half, cx - half:cx + half]

            with self._lock:
                letter, conf = self._predictor.predict(roi)

            self.prediction.emit(letter, conf)

    def stop(self):
        self._stop_event.set()
        self.wait()
