"""
Generate a sample images grid from the ASL dataset.
Run from project root: python training/visualize_samples.py
"""
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "asl-numbers-alphabet-dataset"
OUT     = ROOT / "training" / "outputs" / "report_fig0_samples.png"

random.seed(42)

CLASSES = (
    [str(i) for i in range(10)] +
    [chr(c) for c in range(ord("A"), ord("Z") + 1)] +
    ["nothing", "space", "unknown"]
)

# color tag per group
GROUP_COLOR = (
    ["#06b6d4"] * 10 +
    ["#7c3aed"] * 26 +
    ["#d97706", "#059669", "#dc2626"]
)

# pick one random image per class
images, labels, colors = [], [], []
for cls, col in zip(CLASSES, GROUP_COLOR):
    folder = DATA / cls
    files  = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    if not files:
        continue
    img = Image.open(random.choice(files)).convert("L")
    images.append(np.array(img))
    labels.append(cls)
    colors.append(col)

# layout: 5 rows × 8 cols = 40 slots (we have 39 classes, last slot blank)
COLS, ROWS = 8, 5
fig, axes = plt.subplots(ROWS, COLS, figsize=(16, 11))
fig.patch.set_facecolor("#fafafa")
fig.suptitle("3.1  Ukážkové snímky z ASL datasetu  (1 náhodný obrázok / trieda)",
             fontsize=14, fontweight="bold", color="#1e1b4b", y=0.98)

legend_handles = [
    mpatches.Patch(color="#06b6d4", label="Číslice 0–9"),
    mpatches.Patch(color="#7c3aed", label="Písmená A–Z"),
    mpatches.Patch(color="#d97706", label="nothing"),
    mpatches.Patch(color="#059669", label="space"),
    mpatches.Patch(color="#dc2626", label="unknown"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=10, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01))

for idx, ax in enumerate(axes.flat):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if idx >= len(images):
        ax.set_facecolor("#fafafa")
        continue

    ax.imshow(images[idx], cmap="gray", vmin=0, vmax=255)

    # colored border matching the group
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(colors[idx])
        spine.set_linewidth(2.5)

    ax.set_title(labels[idx], fontsize=10, fontweight="bold",
                 color=colors[idx], pad=3)

fig.tight_layout(rect=[0, 0.06, 1, 0.96], h_pad=0.8, w_pad=0.5)
fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"saved: {OUT}")
plt.close(fig)
