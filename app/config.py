# Central config — every tuneable constant lives here so nothing is hardcoded
# in worker or GUI files. Import with `from app import config`.
import os
from pathlib import Path

ROOT_DIR    = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"

ASL_MODELS = {
    "Model v1 (baseline Adam)":     {"path": WEIGHTS_DIR / "asl_v1.pth", "head": "standard"},
    "Model v2 (AdamW + cosine)":    {"path": WEIGHTS_DIR / "asl_v2.pth", "head": "standard"},
    "Model v3 (deep head + AdamW)": {"path": WEIGHTS_DIR / "asl_v3.pth", "head": "deep"},
}

EMOTION_MODEL_PATH = WEIGHTS_DIR / "emotion.pth"
EMOTION_CLASSES    = ["angry", "happy", "neutral", "sad"]

# ASL class names derived from dataset folder order (39 classes).
# Must match the order used during training (ImageFolder alphabetical sort).
ASL_CLASSES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'nothing','space','unknown',
]

CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
ROI_SIZE      = 200       # pixels, square ROI centred on frame
IMG_SIZE_ASL  = (64, 64)
IMG_SIZE_FACE = (48, 48)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_23d8ed55223eca71d6b071f888561709d4aff77df37d4d55")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Sarah

ASL_CONF_THRESHOLD  = 0.60   # minimum softmax confidence to accept a letter
ASL_DEBOUNCE_FRAMES = 8      # consecutive frames with same letter before accepting

EMOTION_SMOOTH_FRAMES = 15   # rolling average window for emotion scores
EMOTION_SAD_REMAP_THRESHOLD = 0.60  # if sad < this, show neutral instead
