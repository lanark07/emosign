# EmoSign

Real-time ASL (American Sign Language) fingerspelling recognition with facial emotion detection and emotion-aware text-to-speech.

## What it does

- **ASL recognition** — detects hand signs (A–Z, 0–9, space, delete) from your webcam using a custom-trained CNN.
- **Emotion detection** — reads your facial expression (happy / neutral / angry / sad) in parallel.
- **Sentence builder** — accumulates recognized letters into a sentence with debounce logic to avoid accidental repeats.
- **Emotion-aware TTS** — speaks the sentence aloud via ElevenLabs, adjusting voice speed, stability and style to match your detected emotion.
- **3 model variants** — switch between ASL models (v1 baseline, v2 AdamW+cosine, v3 deep head) from the UI dropdown.


- Python 3.12
- Webcam
- ElevenLabs API key (optional — without it the sentence is printed to console instead of spoken)
- CUDA GPU (optional — CPU works, just slower)

## Setup

See [REPRODUCE.md](REPRODUCE.md) for full step-by-step instructions.

Quick start:

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Install PyTorch (CUDA 12.4 example — adjust for your system)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. (Optional) Set ElevenLabs API key
set ELEVENLABS_API_KEY=your_key_here   # Windows
# export ELEVENLABS_API_KEY=your_key_here  # Linux/macOS

# 5. Run
python la.py
```

## Project structure

```
la.py                        → entry point
app/
  config.py                  → all constants (paths, thresholds, camera settings)
  main.py                    → QApplication bootstrap
  gui/
    main_window.py           → top-level window, wires all workers + widgets
    camera_widget.py         → live camera feed with ROI overlay and ASL badge
    emotion_widget.py        → emotion icon + per-class confidence bars
    sentence_widget.py       → built sentence display with Add/Space/Delete/Speak
    model_selector.py        → dropdown to hot-swap ASL model
    theme.py                 → global colour palette + Qt stylesheet
  workers/
    camera_worker.py         → QThread: cv2 capture loop, emits frames
    asl_worker.py            → QThread: runs ASL inference on maxsize=1 queue
    emotion_worker.py        → QThread: runs emotion inference + rolling average
  inference/
    base_predictor.py        → abstract base class for predictors
    asl_predictor.py         → CLAHE preprocessing + CNN forward pass
    emotion_predictor.py     → Haar face detection + emotion CNN
  models/
    asl_model.py             → CNNClassifier with swappable head (shallow/standard/deep)
    emotion_model.py         → EmotionNet (same conv backbone, 4-class head)
  sentence/
    builder.py               → debounce logic: N consecutive frames → commit letter
  tts/
    elevenlabs_client.py     → ElevenLabs SDK + pygame playback, per-emotion voice settings
training/
  train_asl.py               → training loop for ASL model (CLI flags for all hyperparams)
  train_emotion.py           → training loop for emotion model
  asl_data_prep.py           → dataset loader + augmentation for ASL
  emotion_data_prep.py       → dataset loader + augmentation for emotion
  evaluate_asl.py            → per-class accuracy report
weights/
  asl_v1.pth                 → baseline Adam model
  asl_v2.pth                 → AdamW + cosine schedule
  asl_v3.pth                 → deep head + AdamW
  emotion.pth                → 4-class facial emotion model
```

## Training your own models

The `asl-numbers-alphabet-dataset/` folder (not committed — too large for git) must be present. See [REPRODUCE.md](REPRODUCE.md) for dataset setup.

```bash
# ASL variants
python training/train_asl.py --lr 1e-3 --run_name v1
python training/train_asl.py --lr 3e-4 --optimizer adamw --scheduler cosine --dropout 0.4 --run_name v2
python training/train_asl.py --lr 5e-4 --optimizer adamw --head deep --dropout 0.3 --weight_decay 1e-4 --run_name v3

# Emotion model
python training/train_emotion.py

# Evaluate
python training/evaluate_asl.py --weights weights/asl_v1.pth
```

Trained weights are saved to `weights/asl_<run_name>.pth`.

## Configuration

Edit `app/config.py` to change:

| Constant | Default | Purpose |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam index |
| `ASL_CONF_THRESHOLD` | `0.60` | Min confidence to accept a letter |
| `ASL_DEBOUNCE_FRAMES` | `8` | Consecutive frames before committing a letter |
| `EMOTION_SMOOTH_FRAMES` | `15` | Rolling average window for emotion scores |
| `ELEVENLABS_VOICE_ID` | Sarah | ElevenLabs voice — change via env var |

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ELEVENLABS_API_KEY` | No | Without this, TTS prints to console |
| `ELEVENLABS_VOICE_ID` | No | Override the default voice |
