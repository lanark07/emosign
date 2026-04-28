"""
Generate report figures for sections 3.1 – 3.5.
Run from project root:  python training/visualize_report.py
Outputs saved to:       training/outputs/report_fig*.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import csv

matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#fafafa",
    "axes.facecolor":   "#fafafa",
    "axes.grid":        True,
    "grid.color":       "#e5e5f0",
    "grid.linewidth":   0.6,
    "axes.labelcolor":  "#1e1b4b",
    "xtick.color":      "#4b5563",
    "ytick.color":      "#4b5563",
    "text.color":       "#1e1b4b",
})

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "training" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_PURPLE = "#7c3aed"
C_CYAN   = "#06b6d4"
C_GREEN  = "#059669"
C_AMBER  = "#d97706"
C_RED    = "#dc2626"
C_LIGHT  = "#ede9fe"


# ─── helpers ────────────────────────────────────────────────────────────────

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  saved: {path}")
    plt.close(fig)


def section_title(ax, text):
    ax.set_title(text, fontsize=13, fontweight="bold", color="#1e1b4b", pad=10)


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — Dataset (3.1)
# ════════════════════════════════════════════════════════════════════════════
def fig_dataset():
    classes = (
        [str(i) for i in range(10)] +
        [chr(c) for c in range(ord("A"), ord("Z") + 1)] +
        ["nothing", "space", "unknown"]
    )
    counts = [1500] * 10 + [3000] * 26 + [3000, 3000, 1500]
    total  = sum(counts)  # 100 500

    colors = (
        [C_CYAN]   * 10 +
        [C_PURPLE] * 26 +
        [C_AMBER, C_GREEN, C_RED]
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("3.1  Dataset a príprava dát", fontsize=15, fontweight="bold",
                 color="#1e1b4b", y=1.02)

    # ── bar chart ────────────────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(classes, counts, color=colors, width=0.7, zorder=3)
    ax.set_xlabel("Trieda", fontsize=11)
    ax.set_ylabel("Počet vzoriek", fontsize=11)
    section_title(ax, f"Distribúcia vzoriek podľa tried  (celkom {total:,})")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8, rotation=45, ha="right")
    ax.set_ylim(0, 3600)

    legend = [
        mpatches.Patch(color=C_CYAN,   label="Číslice  0–9  (1 500 / trieda)"),
        mpatches.Patch(color=C_PURPLE, label="Písmená A–Z  (3 000 / trieda)"),
        mpatches.Patch(color=C_AMBER,  label="nothing / space  (3 000 / trieda)"),
        mpatches.Patch(color=C_RED,    label="unknown  (1 500 / trieda)"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="upper right", framealpha=0.9)

    # ── pie chart ────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_aspect("equal")
    ax2.axis("off")
    splits = [0.70 * total, 0.10 * total, 0.20 * total]
    labels = [f"Train\n{splits[0]:,.0f}", f"Val\n{splits[1]:,.0f}", f"Test\n{splits[2]:,.0f}"]
    wedge_colors = [C_PURPLE, C_CYAN, C_GREEN]
    wedges, texts, autotexts = ax2.pie(
        splits, labels=labels, colors=wedge_colors,
        autopct="%1.0f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    section_title(ax2, "Rozdelenie datasetu")

    fig.tight_layout()
    save(fig, "report_fig1_dataset.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — CNN Architecture (3.2)
# ════════════════════════════════════════════════════════════════════════════
def fig_architecture():
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("3.2  Architektúra CNN modelu", fontsize=15, fontweight="bold",
                 color="#1e1b4b", y=1.01)

    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.12)
    ax_arch = fig.add_subplot(gs[0])
    ax_heads = fig.add_subplot(gs[1])

    for ax in (ax_arch, ax_heads):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # ── Architecture flow ────────────────────────────────────────────────────
    section_title(ax_arch, "Tok vstupného obrazca sieťou")

    blocks = [
        ("Vstup\n1 × 64 × 64",         "#e0e7ff", C_PURPLE, 8.5),
        ("ConvBlock\n1 → 32 kanálov\n[Conv → BN → ReLU] × 2\n+ MaxPool2d ↓½",  "#ede9fe", C_PURPLE, 7.0),
        ("ConvBlock\n32 → 64 kanálov\n[Conv → BN → ReLU] × 2\n+ MaxPool2d ↓½", "#ede9fe", C_PURPLE, 5.4),
        ("ConvBlock\n64 → 128 kanálov\n[Conv → BN → ReLU] × 2\n+ MaxPool2d ↓½","#ede9fe", C_PURPLE, 3.8),
        ("ConvBlock\n128 → 256 kanálov\n[Conv → BN → ReLU] × 2\n+ MaxPool2d ↓½","#ede9fe", C_PURPLE, 2.2),
        ("AdaptiveAvgPool → Flatten\n256-dimenzionálny vektor", "#d1fae5", C_GREEN, 0.9),
    ]

    for label, bg, border, y in blocks:
        box = FancyBboxPatch((1.0, y - 0.55), 8.0, 1.0,
                             boxstyle="round,pad=0.08",
                             facecolor=bg, edgecolor=border, linewidth=1.5)
        ax_arch.add_patch(box)
        ax_arch.text(5.0, y - 0.05, label,
                     ha="center", va="center", fontsize=8.5,
                     color="#1e1b4b", multialignment="center")

    # arrows
    arrow_ys = [(b[3] - 0.55) for b in blocks[1:]]
    prev_ys  = [(b[3] + 0.45) for b in blocks[:-1]]
    for y0, y1 in zip(prev_ys, arrow_ys):
        ax_arch.annotate("", xy=(5, y1 + 0.02), xytext=(5, y0 - 0.02),
                         arrowprops={"arrowstyle": "->", "color": "#9ca3af", "lw": 1.2})

    ax_arch.text(5.0, 0.25, "→  Classifier Head  →  39 logits",
                 ha="center", va="center", fontsize=9,
                 color=C_PURPLE, fontweight="bold")

    # ── Head comparison table ────────────────────────────────────────────────
    section_title(ax_heads, "Varianty Classifier Head")

    rows = [
        ("shallow",  "256 → 39",               "—",     "—"),
        ("standard", "256 → 256 → 39",         "ReLU",  "Dropout"),
        ("deep",     "256 → 512 → 256 → 39",   "ReLU",  "BN + Dropout ×2"),
    ]
    col_headers = ["Typ", "Vrstvy", "Aktivácia", "Regulácia"]
    col_x = [0.5, 2.8, 6.2, 8.2]
    row_start = 8.8
    row_h = 1.1

    # header
    for cx, ch in zip(col_x, col_headers):
        ax_heads.text(cx, row_start, ch, fontsize=9, fontweight="bold",
                      color="#6b7280", ha="left")
    ax_heads.axhline(row_start - 0.25, color=C_PURPLE, linewidth=1.2, xmin=0.0, xmax=1.0)

    highlight = ["#f5f3ff", "#ede9fe", "#ddd6fe"]
    used_in = ["v1", "v2", "v3"]
    for i, (row, bg, model) in enumerate(zip(rows, highlight, used_in)):
        y = row_start - 0.55 - i * row_h
        rect = FancyBboxPatch((0.2, y - 0.35), 9.5, 0.85,
                              boxstyle="round,pad=0.05",
                              facecolor=bg, edgecolor="none")
        ax_heads.add_patch(rect)
        for cx, val in zip(col_x, row):
            ax_heads.text(cx, y, val, fontsize=8.5, ha="left", va="center",
                          color="#1e1b4b")
        ax_heads.text(9.75, y, f"← {model}", fontsize=8, color=C_PURPLE,
                      va="center", fontweight="bold")

    fig.tight_layout()
    save(fig, "report_fig2_architecture.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Hyperparameters (3.3)
# ════════════════════════════════════════════════════════════════════════════
def fig_hyperparameters():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("3.3  Hyperparametre a varianty modelov", fontsize=15,
                 fontweight="bold", color="#1e1b4b", y=1.02)

    # ── Table ────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.axis("off")
    section_title(ax, "Porovnanie hyperparametrov")

    params = [
        ("Optimizer",       "Adam",       "AdamW",      "AdamW"),
        ("Learning rate",   "1e-3",       "3e-4",       "5e-4"),
        ("Scheduler",       "Plateau",    "Cosine",     "Plateau"),
        ("Dropout",         "0.5",        "0.4",        "0.3"),
        ("Weight decay",    "0.0",        "0.0",        "1e-4"),
        ("Label smoothing", "0.1",        "0.1",        "0.1"),
        ("Batch size",      "64",         "64",         "64"),
        ("Head",            "standard",   "standard",   "deep"),
        ("Max epochs",      "25",         "25",         "25"),
        ("Early stop pat.", "6",          "6",          "6"),
        ("Augmentation",    "✓",          "✓",          "✓"),
    ]

    col_labels = ["Hyperparameter", "v1 (baseline)", "v2 (AdamW+cos)", "v3 (deep head)"]
    col_x      = [0.0, 0.42, 0.63, 0.82]
    col_align  = ["left", "center", "center", "center"]
    row_h      = 0.073
    y0         = 0.97

    header_colors = ["#1e1b4b"] + [C_PURPLE] * 3
    for cx, ch, cl, ca in zip(col_x, col_labels, header_colors, col_align):
        ax.text(cx, y0, ch, transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", color=cl, ha=ca, va="top")
    ax.plot([0, 1], [y0 - 0.04, y0 - 0.04], color=C_PURPLE,
            linewidth=1.2, transform=ax.transAxes, clip_on=False)

    row_bgs = ["#f5f3ff", "#fafafa"]
    for i, row in enumerate(params):
        y = y0 - 0.08 - i * row_h
        rect = plt.Rectangle((0, y - 0.025), 1, row_h - 0.008,
                              transform=ax.transAxes,
                              facecolor=row_bgs[i % 2], edgecolor="none")
        ax.add_patch(rect)
        for cx, val, ca in zip(col_x, row, col_align):
            ax.text(cx, y, val, transform=ax.transAxes,
                    fontsize=9, ha=ca, va="center", color="#1e1b4b")

    # ── Grouped bar: key numeric hyperparameters ─────────────────────────────
    ax2 = axes[1]
    section_title(ax2, "Vizuálne porovnanie kľúčových parametrov")

    params_plot = {
        "Learning\nrate (×10³)":  [1.0,  0.3,  0.5],
        "Dropout":                 [0.5,  0.4,  0.3],
        "Weight\ndecay (×10³)":   [0.0,  0.0,  0.1],
    }
    x      = np.arange(len(params_plot))
    width  = 0.22
    colors = [C_PURPLE, C_CYAN, C_GREEN]
    labels = ["v1", "v2", "v3"]

    for i, (color, label) in enumerate(zip(colors, labels)):
        vals = [list(v)[i] for v in params_plot.values()]
        bars = ax2.bar(x + (i - 1) * width, vals, width,
                       label=label, color=color, alpha=0.88, zorder=3)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.008,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8,
                     color=color, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(list(params_plot.keys()), fontsize=10)
    ax2.set_ylabel("Hodnota", fontsize=11)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    save(fig, "report_fig3_hyperparameters.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — Training curves (3.4)
# ════════════════════════════════════════════════════════════════════════════
def fig_training():
    plot_paths = [
        OUT_DIR / "plots" / "asl_v1.png",
        OUT_DIR / "plots" / "asl_v2.png",
        OUT_DIR / "plots" / "asl_v3.png",
    ]
    titles = [
        "v1 — Adam, LR=1e-3, dropout=0.5, standard head",
        "v2 — AdamW, LR=3e-4, cosine scheduler, dropout=0.4, standard head",
        "v3 — AdamW, LR=5e-4, plateau, dropout=0.3, deep head",
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.suptitle("3.4  Priebeh trénovania — loss a accuracy po epochách",
                 fontsize=15, fontweight="bold", color="#1e1b4b", y=1.01)

    for ax, path, title in zip(axes, plot_paths, titles):
        ax.axis("off")
        if path.exists():
            img = np.array(Image.open(path))
            ax.imshow(img, aspect="auto")
        ax.set_title(title, fontsize=10.5, color="#1e1b4b", pad=6,
                     fontweight="bold")

    fig.tight_layout(h_pad=2.0)
    save(fig, "report_fig4_training_curves.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — Evaluation (3.5)
# ════════════════════════════════════════════════════════════════════════════
def fig_evaluation():
    # Parse CSV
    csv_path = OUT_DIR / "ablation_results_asl.csv"
    runs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            runs.append({
                "name":     f"v{row['run_name']}",
                "aug":      row["augment"] == "True",
                "acc":      float(row["best_val_acc"]) * 100,
                "loss":     float(row["best_val_loss"]),
                "epoch":    int(row["best_epoch"]),
                "trained":  int(row["epochs_trained"]),
                "opt":      row["optimizer"],
                "sched":    row["scheduler"],
                "head":     row["head"],
            })

    no_aug = [r for r in runs if not r["aug"]]
    with_aug = [r for r in runs if r["aug"]]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("3.5  Vyhodnotenie modelu", fontsize=15, fontweight="bold",
                 color="#1e1b4b", y=1.01)
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # ── 1. Accuracy comparison bar chart ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    section_title(ax1, "Validačná presnosť — bez vs. s augmentáciou")

    names   = ["v1", "v2", "v3"]
    acc_no  = [r["acc"] for r in no_aug]
    acc_aug = [r["acc"] for r in with_aug]
    x = np.arange(3)
    w = 0.32

    b1 = ax1.bar(x - w/2, acc_no,  w, label="Bez augmentácie", color=C_CYAN,   alpha=0.85, zorder=3)
    b2 = ax1.bar(x + w/2, acc_aug, w, label="S augmentáciou",  color=C_PURPLE, alpha=0.85, zorder=3)

    for bars in (b1, b2):
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() - 0.12,
                     f"{bar.get_height():.2f}%",
                     ha="center", va="top", fontsize=8, color="white", fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11)
    ax1.set_ylabel("Val. accuracy (%)", fontsize=10)
    ax1.set_ylim(99.0, 100.05)
    ax1.legend(fontsize=9)

    # ── 2. Best epoch comparison ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    section_title(ax2, "Epocha najlepšieho modelu (early stopping)")

    ep_no  = [r["epoch"] for r in no_aug]
    ep_aug = [r["epoch"] for r in with_aug]
    b3 = ax2.bar(x - w/2, ep_no,  w, label="Bez augmentácie", color=C_CYAN,   alpha=0.85, zorder=3)
    b4 = ax2.bar(x + w/2, ep_aug, w, label="S augmentáciou",  color=C_PURPLE, alpha=0.85, zorder=3)
    for bars in (b3, b4):
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     str(int(bar.get_height())),
                     ha="center", va="bottom", fontsize=9, fontweight="bold",
                     color=C_PURPLE if bars is b4 else C_CYAN)

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=11)
    ax2.set_ylabel("Epocha", fontsize=10)
    ax2.legend(fontsize=9)

    # ── 3. Val loss comparison ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    section_title(ax3, "Validačný loss (nižší = lepší)")

    loss_no  = [r["loss"] for r in no_aug]
    loss_aug = [r["loss"] for r in with_aug]
    b5 = ax3.bar(x - w/2, loss_no,  w, label="Bez augmentácie", color=C_CYAN,   alpha=0.85, zorder=3)
    b6 = ax3.bar(x + w/2, loss_aug, w, label="S augmentáciou",  color=C_PURPLE, alpha=0.85, zorder=3)
    for bars in (b5, b6):
        for bar in bars:
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                     color=C_PURPLE if bars is b6 else C_CYAN)

    ax3.set_xticks(x)
    ax3.set_xticklabels(names, fontsize=11)
    ax3.set_ylabel("Cross-entropy loss", fontsize=10)
    ax3.legend(fontsize=9)

    # ── 4. Summary results table ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    section_title(ax4, "Súhrnné výsledky všetkých behov")

    headers = ["Model", "Aug", "Optimizer", "Head", "Best epoch", "Val acc"]
    col_x   = [0.0, 0.12, 0.22, 0.38, 0.58, 0.76]
    col_al  = ["left"] * 6
    y0      = 0.95
    row_h   = 0.115

    for cx, h in zip(col_x, headers):
        ax4.text(cx, y0, h, transform=ax4.transAxes,
                 fontsize=8.5, fontweight="bold", color="#6b7280", ha="left")
    ax4.plot([0, 1], [y0 - 0.04, y0 - 0.04], color=C_PURPLE,
             linewidth=1.2, transform=ax4.transAxes, clip_on=False)

    row_bgs = ["#f5f3ff", "#fafafa"]
    for i, r in enumerate(runs):
        y = y0 - 0.1 - i * row_h
        rect = plt.Rectangle((0, y - 0.04), 1, row_h - 0.01,
                              transform=ax4.transAxes,
                              facecolor=row_bgs[i % 2], edgecolor="none")
        ax4.add_patch(rect)
        best = (r["acc"] == max(rr["acc"] for rr in runs))
        vals = [r["name"], "✓" if r["aug"] else "✗",
                r["opt"], r["head"], str(r["epoch"]), f"{r['acc']:.4f}%"]
        for cx, val in zip(col_x, vals):
            color = C_GREEN if (best and val == vals[-1]) else "#1e1b4b"
            weight = "bold" if best and val == vals[-1] else "normal"
            ax4.text(cx, y, val, transform=ax4.transAxes,
                     fontsize=8.5, ha="left", va="center",
                     color=color, fontweight=weight)

    fig.tight_layout()
    save(fig, "report_fig5_evaluation.png")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating report figures…")
    fig_dataset()
    fig_architecture()
    fig_hyperparameters()
    fig_training()
    fig_evaluation()
    print("\nDone. All figures saved to training/outputs/")
