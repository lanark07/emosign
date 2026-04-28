from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from app.inference.base_predictor import BasePredictor
from app.models.emotion_model import build_emotion_net
from app import config

_FACE_CASCADE = None   # lazy-loaded on first call

_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(config.IMG_SIZE_FACE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def _get_cascade():
    # Lazy-loaded once per process — loading on import would slow startup
    # even when the emotion model is disabled.
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _FACE_CASCADE


class EmotionPredictor(BasePredictor):
    def __init__(self, num_classes: int = len(config.EMOTION_CLASSES)):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = config.EMOTION_CLASSES

    def load(self, path: Path) -> None:
        self.model = build_emotion_net(num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def predict(self, frame: np.ndarray) -> tuple[str, dict]:
        """
        Args:
            frame: full BGR frame from webcam
        Returns:
            (dominant_emotion, {class_name: score}) — falls back to full frame if no face found
        """
        roi = self._detect_face(frame)
        tensor = self._preprocess(roi)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().tolist()
        scores = {name: round(prob, 4) for name, prob in zip(self.class_names, probs)}
        dominant = max(scores, key=scores.get)
        return dominant, scores

    def _detect_face(self, frame: np.ndarray) -> np.ndarray:
        """Return face crop if detected, else centre crop of the frame."""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return frame[y:y+h, x:x+w]
        # Fallback: centre square crop
        h, w = frame.shape[:2]
        side  = min(h, w)
        y0    = (h - side) // 2
        x0    = (w - side) // 2
        return frame[y0:y0+side, x0:x0+side]

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        return _TRANSFORM(frame).unsqueeze(0).to(self.device)
