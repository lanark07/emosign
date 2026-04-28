import queue
import threading
from collections import deque
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from app.inference.emotion_predictor import EmotionPredictor
from app import config


# Same queue-based pattern as ASLWorker.
# Adds a rolling average over EMOTION_SMOOTH_FRAMES to reduce jitter from
# momentary mispredictions (e.g., brief "angry" flash during a neutral expression).
class EmotionWorker(QThread):
    prediction = pyqtSignal(str, dict)   # dominant emotion, {class: score}

    def __init__(self, model_path: Path, parent=None):
        super().__init__(parent)
        self._queue      = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._predictor  = EmotionPredictor()
        self._predictor.load(model_path)
        self._history    = deque(maxlen=config.EMOTION_SMOOTH_FRAMES)

    def on_frame(self, frame: np.ndarray):
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            emotion, scores = self._predictor.predict(frame)
            self._history.append(scores)

            # Average scores over the rolling window
            smoothed = {}
            for cls in config.EMOTION_CLASSES:
                smoothed[cls] = sum(s.get(cls, 0.0) for s in self._history) / len(self._history)

            dominant = max(smoothed, key=smoothed.__getitem__)

            # Remap weak "sad" to "neutral": the model often outputs low-confidence
            # sad for neutral faces, so only report sad when clearly dominant.
            if dominant == "sad" and smoothed["sad"] < config.EMOTION_SAD_REMAP_THRESHOLD:
                dominant = "neutral"

            self.prediction.emit(dominant, smoothed)

    def stop(self):
        self._stop_event.set()
        self.wait()
