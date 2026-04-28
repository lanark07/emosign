# Abstract base for ASLPredictor and EmotionPredictor.
# Subclasses must implement load(), predict(), and _preprocess().
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch


class BasePredictor(ABC):
    def __init__(self):
        self.model  = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model weights from path."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> tuple:
        """Run inference on a BGR frame. Returns (label, confidence, ...)."""

    @abstractmethod
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Convert BGR frame to model-ready tensor."""
