import threading

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from app import config


# Captures frames on a dedicated thread and emits each one via frame_ready.
# The signal is connected to both inference workers and the camera widget —
# each consumer decides independently whether to drop or process the frame.
class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_event = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        if not cap.isOpened():
            print("CameraWorker: could not open webcam.")
            return

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)

        cap.release()

    def stop(self):
        self._stop_event.set()
        self.wait()
