from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.inference.base_predictor import BasePredictor
from app.models.asl_model import build_classifier
from app import config


# Runs the ASL CNN on the 200×200 px ROI cropped from the centre of each frame.
class ASLPredictor(BasePredictor):
    def __init__(self, num_classes: int = len(config.ASL_CLASSES)):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = config.ASL_CLASSES
        # CLAHE normalises local contrast so predictions are stable under
        # different webcam exposures and room lighting conditions.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def load(self, path: Path, head: str = "standard") -> None:
        self.model = build_classifier(num_classes=self.num_classes, head=head).to(self.device)
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def predict(self, frame: np.ndarray) -> tuple[str, float]:
        tensor = self._preprocess(frame)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        idx = int(probs.argmax())
        return self.class_names[idx], float(probs[idx])

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # CLAHE: normalize contrast so webcam lighting matches training data
        gray = self._clahe.apply(gray)
        # Resize to training input size
        h, w = config.IMG_SIZE_ASL
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        # To float tensor [0,1], shape (1, 1, H, W)
        tensor = torch.from_numpy(gray).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)
