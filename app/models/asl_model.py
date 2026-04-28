# 4-stage CNN for grayscale 64×64 hand-sign images.
# Architecture: ConvBlock × 4 → AdaptiveAvgPool → swappable classification head.
# The same ConvBlock is reused by EmotionNet in emotion_model.py.
import torch.nn as nn

NUM_CLASSES = 39


class ConvBlock(nn.Module):
    # Two conv layers with BN+ReLU. Double-conv before pooling increases
    # receptive field without adding a third pooling step.
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def _build_head(head: str, in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    # Three head variants allow ablation studies without changing the backbone.
    # "shallow" = direct linear (fastest, weakest), "deep" = two hidden layers (strongest).
    if head == "shallow":
        return nn.Sequential(nn.Linear(in_features, num_classes))
    if head == "standard":
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    if head == "deep":
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    raise ValueError(f"Unknown head type: {head!r}")


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5, head="standard"):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),    nn.MaxPool2d(2),
            ConvBlock(32, 64),   nn.MaxPool2d(2),
            ConvBlock(64, 128),  nn.MaxPool2d(2),
            ConvBlock(128, 256), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = _build_head(head, 256, num_classes, dropout)

    def forward(self, x):
        return self.classifier(self.features(x))


def build_classifier(num_classes=NUM_CLASSES, dropout=0.5, head="standard") -> CNNClassifier:
    return CNNClassifier(num_classes=num_classes, dropout=dropout, head=head)
