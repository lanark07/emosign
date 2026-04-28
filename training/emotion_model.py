import torch
import torch.nn as nn

from asl_model import ConvBlock

EMOTION_CLASSES = ["angry", "happy", "neutral", "sad"]
NUM_CLASSES     = len(EMOTION_CLASSES)


class EmotionNet(nn.Module):
    """Grayscale CNN for facial emotion classification (input: 1×48×48)."""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),    nn.MaxPool2d(2),   # → (B,  32, 24, 24)
            ConvBlock(32, 64),   nn.MaxPool2d(2),   # → (B,  64, 12, 12)
            ConvBlock(64, 128),  nn.MaxPool2d(2),   # → (B, 128,  6,  6)
            ConvBlock(128, 256), nn.MaxPool2d(2),   # → (B, 256,  3,  3)
            nn.AdaptiveAvgPool2d(1),                # → (B, 256,  1,  1)
            nn.Flatten(),                           # → (B, 256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_emotion_net(num_classes=NUM_CLASSES, dropout=0.5):
    return EmotionNet(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    m   = build_emotion_net()
    out = m(torch.randn(4, 1, 48, 48))
    params = sum(p.numel() for p in m.parameters())
    print(f"EmotionNet  output={out.shape}  params={params:,}")
