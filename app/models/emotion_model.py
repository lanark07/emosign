# Shares the ConvBlock backbone with the ASL model.
# Input: 1×48×48 grayscale face crop. Output: 4 emotion logits.
import torch.nn as nn
from app.models.asl_model import ConvBlock

EMOTION_CLASSES = ["angry", "happy", "neutral", "sad"]
NUM_CLASSES     = len(EMOTION_CLASSES)


class EmotionNet(nn.Module):
    """Grayscale CNN for facial emotion classification (input: 1×48×48)."""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),    nn.MaxPool2d(2),
            ConvBlock(32, 64),   nn.MaxPool2d(2),
            ConvBlock(64, 128),  nn.MaxPool2d(2),
            ConvBlock(128, 256), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_emotion_net(num_classes=NUM_CLASSES, dropout=0.5) -> EmotionNet:
    return EmotionNet(num_classes=num_classes, dropout=dropout)
