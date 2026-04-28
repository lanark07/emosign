"""Train facial emotion CNN. Run from project root: python training/train_emotion.py [flags]

Expects labelled face images in training/emotion_data/train/ and training/emotion_data/test/
with one sub-folder per class (angry, happy, neutral, sad).
Saves best weights to weights/emotion.pth.
"""
import sys
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emotion_data_prep import load_emotion_datasets, set_seed, TRAIN_DIR, TEST_DIR
from emotion_model import build_emotion_net

ROOT_DIR   = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
PLOTS_DIR  = ROOT_DIR / "training" / "outputs" / "plots"

SEED              = 67
EPOCHS            = 15
LEARNING_RATE     = 1e-3
DROPOUT           = 0.5
BATCH_SIZE        = 64
WEIGHT_DECAY      = 1e-4
EARLY_STOP_PATIENCE = 5


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            n          += images.size(0)
    return total_loss / n, correct / n


def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="train"); ax1.plot(history["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(history["train_acc"], label="train"); ax2.plot(history["val_acc"], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(lr=LEARNING_RATE, dropout=DROPOUT, batch_size=BATCH_SIZE,
         weight_decay=WEIGHT_DECAY, epochs=EPOCHS, augment=True):
    set_seed(SEED)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | lr={lr:.2e} | dropout={dropout} | batch={batch_size}")

    train_loader, val_loader, _, class_names = load_emotion_datasets(
        batch_size=batch_size, seed=SEED, augment=augment
    )
    num_classes = len(class_names)
    print(f"Emotion classes ({num_classes}): {class_names}")

    model     = build_emotion_net(num_classes=num_classes, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, no_improve = float("inf"), 0
    best_val_acc, best_epoch  = 0.0, 0
    best_path = WEIGHTS_DIR / "emotion.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d} Train", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            n          += images.size(0)
        train_loss, train_acc = total_loss / n, correct / n

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d} | train {train_loss:.4f}/{train_acc:.4f} | "
              f"val {val_loss:.4f}/{val_acc:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss, best_val_acc, best_epoch = val_loss, val_acc, epoch
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break

    plot_history(history, PLOTS_DIR / "emotion_training.png")
    print(f"Done. Best model → {best_path}  (epoch {best_epoch}, val_acc={best_val_acc:.4f})")
    print(f"Classes: {class_names}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr",           type=float, default=LEARNING_RATE)
    parser.add_argument("--dropout",      type=float, default=DROPOUT)
    parser.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--no_augment",   action="store_true")
    args = parser.parse_args()
    main(lr=args.lr, dropout=args.dropout, batch_size=args.batch_size,
         weight_decay=args.weight_decay, epochs=args.epochs, augment=not args.no_augment)
