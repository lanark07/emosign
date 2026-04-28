"""Evaluate ASL model on test set. Run from project root: python training/evaluate_asl.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from asl_data_prep import load_preprocessed_datasets, set_seed
from asl_model import build_classifier

ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = str(ROOT_DIR / "asl-numbers-alphabet-dataset")
WEIGHTS_DIR = ROOT_DIR / "weights"
PLOTS_DIR  = ROOT_DIR / "training" / "outputs" / "plots"
SEED       = 67


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device, non_blocking=True)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.title("ASL CNN — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_DIR / "asl_v1.pth"))
    args = parser.parse_args()

    set_seed(SEED)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, class_names = load_preprocessed_datasets(DATA_DIR, batch_size=128, seed=SEED)

    model = build_classifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))

    labels, preds = get_predictions(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    plot_confusion_matrix(labels, preds, class_names, PLOTS_DIR / "asl_confusion_matrix.png")
