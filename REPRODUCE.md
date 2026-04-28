# How to Reproduce This Project

Complete guide to set up the environment, obtain the data, train the models, and run the app from scratch.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12 | Tested on 3.12.10. Use `python --version` to check. |
| Git | To clone the repo. |
| Webcam | Any USB or built-in webcam works. |
| GPU (optional) | NVIDIA GPU with CUDA 12.4+ speeds up training significantly. CPU works for inference. |

---

## 2. Clone the repository

```bash
git clone https://github.com/lanark07/emosign.git
cd emosign
```

---

## 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **Linux/macOS:** `source .venv/bin/activate`

You should see `(.venv)` in your terminal prompt.

---

## 4. Install PyTorch

PyTorch must be installed separately because the correct build depends on your hardware.

### Option A — NVIDIA GPU (CUDA 12.4)
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### Option B — CPU only
```bash
pip install torch==2.6.0 torchvision==0.21.0
```

> For other CUDA versions visit https://pytorch.org/get-started/locally/

---

## 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

This installs PyQt5, OpenCV, NumPy, pygame, ElevenLabs SDK, matplotlib, and tqdm.

---

## 6. Set up the ElevenLabs API key (optional)

Without this the app still works — it will print the sentence to the console instead of speaking it.

```bash
# Windows
set ELEVENLABS_API_KEY=your_key_here

# Linux/macOS
export ELEVENLABS_API_KEY=your_key_here
```

Get a free API key at https://elevenlabs.io (free tier is enough for demos).

---

## 7. Run the app (using pre-trained weights)

The `weights/` folder is committed to the repo and contains all three ASL models and the emotion model.

```bash
python la.py
```

The app opens a window with a live camera feed. Position your hand inside the box and hold a sign still for ~0.5 seconds to register a letter.

---

## 8. (Optional) Get the dataset and retrain

The dataset is not committed to the repo (1.1 GB). You need it only if you want to retrain the models.

### 8a. Download the ASL dataset

Download the **ASL Numbers + Alphabet Dataset** from Kaggle:
https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet

Extract it so the folder structure looks like:

```
emosign/
  asl-numbers-alphabet-dataset/
    0/
    1/
    ...
    A/
    B/
    ...
    Z/
    nothing/
    space/
    unknown/
```

### 8b. Download the emotion dataset

Download the **FER-2013** or a compatible facial emotion dataset.  
The emotion training script expects two folders:

```
training/
  emotion_data/
    train/
      angry/
      happy/
      neutral/
      sad/
    test/
      angry/
      happy/
      neutral/
      sad/
```

Check `training/emotion_data_prep.py` for the exact expected paths (`TRAIN_DIR` and `TEST_DIR`).

---

## 9. Train the models

Run all commands from the project root with the venv activated.

### ASL model variants

```bash
# v1 — baseline Adam, standard head
python training/train_asl.py --lr 1e-3 --optimizer adam --run_name v1

# v2 — AdamW + cosine schedule + lower dropout
python training/train_asl.py --lr 3e-4 --optimizer adamw --scheduler cosine --dropout 0.4 --run_name v2

# v3 — deep classification head + AdamW + weight decay
python training/train_asl.py --lr 5e-4 --optimizer adamw --head deep --dropout 0.3 --weight_decay 1e-4 --run_name v3
```

Weights are saved to `weights/asl_<run_name>.pth` automatically.

### Emotion model

```bash
python training/train_emotion.py
```

Saves to `weights/emotion.pth`.

### Evaluate

```bash
python training/evaluate_asl.py --weights weights/asl_v1.pth
```

---

## 10. Key configuration knobs

Open `app/config.py` to adjust behaviour without touching any other file:

```python
CAMERA_INDEX        = 0      # change if your webcam is not index 0
ASL_CONF_THRESHOLD  = 0.60   # raise to require higher confidence before accepting a letter
ASL_DEBOUNCE_FRAMES = 8      # raise to slow down letter registration
EMOTION_SMOOTH_FRAMES = 15   # rolling window size for emotion smoothing
```

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| Webcam doesn't open | Change `CAMERA_INDEX` in `config.py` to `1` or `2` |
| `weights/emotion.pth` not found | The emotion widget is silently skipped at startup. Train or copy the file. |
| TTS doesn't play audio | Check `ELEVENLABS_API_KEY` is set. Errors are printed to console. |
| Very slow inference | Make sure PyTorch was installed with CUDA support (`torch.cuda.is_available()` returns `True`). |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the activated venv. |
| Signs not recognized well | Ensure good lighting, keep your hand centred in the box, and try a different model variant from the dropdown. |

---

## Project was built with

- Python 3.12.10
- PyTorch 2.6.0 + CUDA 12.4
- PyQt5 5.15.11
- OpenCV 4.13
- ElevenLabs SDK 2.44.0
