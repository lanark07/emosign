# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.12.10 via `.venv/` (Windows)
- Activate: `.venv\Scripts\activate`
- Run app: `python la.py` (from project root)

## Architecture

Desktop app: ASL sign recognition + facial emotion detection + sentence builder + ElevenLabs TTS.

```
la.py                    → entry shim → app.main.run()
app/
  config.py              → all constants (paths, thresholds, API keys)
  main.py                → QApplication bootstrap
  gui/main_window.py     → owns all workers + widgets, wires all signals
  gui/camera_widget.py   → CameraWidget(QLabel) — displays frames + ROI + ASL overlay
  gui/emotion_widget.py  → EmotionWidget — emotion label + per-class progress bars
  gui/sentence_widget.py → SentenceWidget — built sentence display + Speak/Clear buttons
  gui/model_selector.py  → ModelSelectorWidget — QComboBox for 3 ASL model variants
  workers/camera_worker.py  → CameraWorker(QThread) — cv2 capture loop, emits frame_ready
  workers/asl_worker.py     → ASLWorker(QThread) — queue(maxsize=1), runs ASLPredictor
  workers/emotion_worker.py → EmotionWorker(QThread) — same pattern, runs EmotionPredictor
  inference/asl_predictor.py     → ASLPredictor(BasePredictor) — preprocesses ROI, top-1 softmax
  inference/emotion_predictor.py → EmotionPredictor — Haar face detect → crop → classify
  sentence/builder.py    → SentenceBuilder — debounce (8 frames), confidence threshold (0.85)
  tts/elevenlabs_client.py → TTSClient — ElevenLabs SDK + pygame.mixer playback
training/                → standalone training scripts, run from project root
  asl_model.py / emotion_model.py    → model architectures (CNNClassifier, EmotionNet)
  asl_data_prep.py / emotion_data_prep.py → dataset loaders
  train_asl.py / train_emotion.py    → training loops with early stopping
weights/
  asl_v1.pth, asl_v2.pth, asl_v3.pth  → 3 ASL model variants (39-class CNN)
  emotion.pth                          → 4-class emotion model (angry/happy/neutral/sad)
```

## Key Design Rules

- **Workers never touch GUI.** All updates go via Qt signals into the main thread.
- **Frame queues are maxsize=1.** Stale frames are dropped, not accumulated.
- **Model hot-swap** happens inside `ASLWorker.load_model()` under a lock — no thread restart needed.
- **Emotion worker** is skipped at startup if `weights/emotion.pth` doesn't exist yet.

## Signal Flow

```
CameraWorker.frame_ready → CameraWidget.update_frame
                         → ASLWorker.on_frame → ASLWorker.prediction → MainWindow._on_asl_prediction → SentenceBuilder
                                                                      → CameraWidget.set_asl_result
                         → EmotionWorker.on_frame → EmotionWorker.prediction → EmotionWidget.update_emotion
ModelSelectorWidget.model_changed → ASLWorker.load_model
SentenceWidget.speak_requested → QThreadPool → TTSClient.speak
```

## Training Commands

Run from project root:

```bash
# Train new ASL variants
python training/train_asl.py --lr 3e-4 --optimizer adamw --scheduler cosine --dropout 0.4 --run_name v2
python training/train_asl.py --lr 5e-4 --optimizer adamw --head deep --dropout 0.3 --weight_decay 1e-4 --run_name v3

# Train emotion model
python training/train_emotion.py

# Evaluate ASL model
python training/evaluate_asl.py --weights weights/asl_v1.pth
```

Trained weights save to `weights/asl_<run_name>.pth`. Rename to `asl_v2.pth` / `asl_v3.pth` after training.

## ElevenLabs TTS

Set `ELEVENLABS_API_KEY` environment variable before running. Without it, TTS prints to console instead of speaking. Voice can be changed via `ELEVENLABS_VOICE_ID` env var.
