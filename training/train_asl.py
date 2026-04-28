"""Train ASL CNN classifier. Run from project root: python training/train_asl.py [flags]

All hyperparameters are exposed as CLI flags so you can run ablation experiments
without editing the file. Results are appended to training/outputs/ablation_results_asl.csv
and a loss/accuracy plot is saved to training/outputs/plots/.
"""
import sys
import csv
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asl_data_prep import load_preprocessed_datasets, set_seed
from asl_model import build_classifier

ROOT_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = str(ROOT_DIR / "asl-numbers-alphabet-dataset")
WEIGHTS_DIR   = ROOT_DIR / "weights"
PLOTS_DIR     = ROOT_DIR / "training" / "outputs" / "plots"
RESULTS_CSV   = ROOT_DIR / "training" / "outputs" / "ablation_results_asl.csv"

SEED              = 67
EPOCHS            = 25
LEARNING_RATE     = 1e-3
DROPOUT           = 0.5
BATCH_SIZE        = 64
WEIGHT_DECAY      = 0.0
LABEL_SMOOTHING   = 0.1
OPTIMIZER_NAME    = "adam"
SCHEDULER_NAME    = "plateau"
MOMENTUM          = 0.9
NUM_WORKERS       = 0
HEAD              = "standard"
EARLY_STOP_PATIENCE = 6


def validate_one_epoch(model, loader, criterion, device):
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


def build_optimizer(model, optimizer_name, lr, weight_decay, momentum):
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                               weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer, scheduler_name, epochs):
    name = scheduler_name.lower()
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def step_scheduler(scheduler, scheduler_name, val_loss):
    if scheduler is None:
        return
    if scheduler_name.lower() == "plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()


def append_results_csv(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp","run_name","head","optimizer","scheduler","lr","weight_decay",
                  "momentum","dropout","label_smoothing","batch_size","augment","num_workers",
                  "epochs_trained","best_epoch","best_val_loss","best_val_acc","final_lr","best_model_path"]
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main(
    lr=LEARNING_RATE, dropout=DROPOUT, batch_size=BATCH_SIZE,
    optimizer_name=OPTIMIZER_NAME, scheduler_name=SCHEDULER_NAME,
    weight_decay=WEIGHT_DECAY, label_smoothing=LABEL_SMOOTHING,
    augment=True, momentum=MOMENTUM, num_workers=NUM_WORKERS,
    epochs=EPOCHS, data_dir=DATA_DIR, run_name=None, log_csv=True, head=HEAD,
):
    set_seed(SEED)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Run: {run_name} | opt={optimizer_name} | sched={scheduler_name} | "
          f"lr={lr:.2e} | dropout={dropout} | batch={batch_size} | head={head}")

    train_loader, val_loader, _, class_names = load_preprocessed_datasets(
        data_dir, batch_size=batch_size, seed=SEED, augment=augment,
        num_workers=num_workers, expected_num_classes=None,
    )
    num_classes = len(class_names)
    print(f"Classes: {num_classes} — {class_names}")

    model     = build_classifier(num_classes=num_classes, dropout=dropout, head=head).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay, momentum)
    scheduler = build_scheduler(optimizer, scheduler_name, epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, no_improve = float("inf"), 0
    best_val_acc, best_epoch  = 0.0, 0
    best_path = WEIGHTS_DIR / f"asl_{run_name}.pth"

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

        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        step_scheduler(scheduler, scheduler_name, val_loss)

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

    plot_history(history, PLOTS_DIR / f"asl_{run_name}.png")
    if log_csv:
        append_results_csv(RESULTS_CSV, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name, "head": head, "optimizer": optimizer_name,
            "scheduler": scheduler_name, "lr": lr, "weight_decay": weight_decay,
            "momentum": momentum, "dropout": dropout, "label_smoothing": label_smoothing,
            "batch_size": batch_size, "augment": augment, "num_workers": num_workers,
            "epochs_trained": len(history["train_loss"]), "best_epoch": best_epoch,
            "best_val_loss": best_val_loss, "best_val_acc": best_val_acc,
            "final_lr": optimizer.param_groups[0]["lr"], "best_model_path": str(best_path),
        })

    print(f"Done. Best model → {best_path}  (epoch {best_epoch}, val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr",              type=float, default=LEARNING_RATE)
    parser.add_argument("--dropout",         type=float, default=DROPOUT)
    parser.add_argument("--batch_size",      type=int,   default=BATCH_SIZE)
    parser.add_argument("--optimizer",       type=str,   choices=["adam","adamw","sgd"], default=OPTIMIZER_NAME)
    parser.add_argument("--scheduler",       type=str,   choices=["plateau","cosine","none"], default=SCHEDULER_NAME)
    parser.add_argument("--weight_decay",    type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--momentum",        type=float, default=MOMENTUM)
    parser.add_argument("--num_workers",     type=int,   default=NUM_WORKERS)
    parser.add_argument("--epochs",          type=int,   default=EPOCHS)
    parser.add_argument("--data_dir",        type=str,   default=DATA_DIR)
    parser.add_argument("--augment",         action="store_true", default=True)
    parser.add_argument("--run_name",        type=str,   default=None)
    parser.add_argument("--no_csv",          action="store_true")
    parser.add_argument("--head",            type=str,   choices=["shallow","standard","deep"], default=HEAD)
    args = parser.parse_args()
    main(
        lr=args.lr, dropout=args.dropout, batch_size=args.batch_size,
        optimizer_name=args.optimizer, scheduler_name=args.scheduler,
        weight_decay=args.weight_decay, label_smoothing=args.label_smoothing,
        augment=args.augment, momentum=args.momentum, num_workers=args.num_workers,
        epochs=args.epochs, data_dir=args.data_dir, run_name=args.run_name,
        log_csv=not args.no_csv, head=args.head,
    )
