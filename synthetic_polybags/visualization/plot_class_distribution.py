#!/usr/bin/env python3
"""Class distribution bar chart across train / val / test splits."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_NAMES = [
    "pink\npolybag",
    "blue\npolybag",
    "yellow\npolybag",
    "grey\npolybag",
    "green\npolybag",
    "red\npolybag",
    "teal\npolybag",
]

# RGB tuples matching the actual bag colours
BAG_COLORS = [
    "#E060C0",   # 0 pink  — magenta/warm pink
    "#6080E8",   # 1 blue  — periwinkle/mid-blue
    "#F0C020",   # 2 yellow — orange-yellow
    "#B0B0B0",   # 3 grey
    "#80CC30",   # 4 green — lime/yellow-green
    "#E03030",   # 5 red   — orange-red
    "#30B0A0",   # 6 teal  — cyan-green
]

# counts[split][class_id]
counts = {
    "Train (mcmot)": [3968, 2023, 6447, 1911, 1686, 2000, 3965],
    "Val":           [1004, 1004, 2540, 1028, 1004, 1004, 1004],
    "Test":          [1004,  942, 1994,   76, 1004, 1004, 1004],
}

SPLIT_COLORS   = ["#4C72B0", "#DD8452", "#55A868"]   # blue, orange, green
SPLIT_HATCHES  = ["", "//", ".."]
SPLIT_LABELS   = list(counts.keys())
N_CLASSES      = 7
N_SPLITS       = 3
BAR_W          = 0.22
GROUP_GAP      = 0.08

x = np.arange(N_CLASSES)

fig, ax = plt.subplots(figsize=(11, 5))

for si, (label, color, hatch) in enumerate(
        zip(SPLIT_LABELS, SPLIT_COLORS, SPLIT_HATCHES)):
    offsets = x + (si - 1) * (BAR_W + 0.02)
    vals    = counts[label]
    bars    = ax.bar(offsets, vals, width=BAR_W,
                     color=color, hatch=hatch,
                     edgecolor="white", linewidth=0.6,
                     alpha=0.88, label=label, zorder=3)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                    f"{v:,}", ha="center", va="bottom", fontsize=7.5,
                    color="#333333")

# Coloured class-name tick labels
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=9)
for tick_label, col in zip(ax.get_xticklabels(), BAG_COLORS):
    tick_label.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=col,
                             edgecolor="none", alpha=0.88))
    tick_label.set_color("white")
    tick_label.set_fontweight("bold")

ax.set_xlabel("")
ax.set_ylabel("Instance count", fontsize=11)
ax.set_title("Class distribution across Train / Val / Test splits",
             fontsize=13, fontweight="bold", pad=14)

ax.set_xlim(-0.6, N_CLASSES - 0.4)
ax.set_ylim(0, 8000)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{int(v):,}"))
ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

totals = {k: sum(v) for k, v in counts.items()}
legend_labels = [f"{lbl}  (total {totals[lbl]:,})" for lbl in SPLIT_LABELS]
handles = [mpatches.Patch(facecolor=c, hatch=h, edgecolor="white",
                           alpha=0.88, label=l)
           for c, h, l in zip(SPLIT_COLORS, SPLIT_HATCHES, legend_labels)]
ax.legend(handles=handles, loc="upper right", fontsize=10,
          framealpha=0.9, edgecolor="#cccccc")

out = "/Users/awthura/OVGU/AMS/synthetic_polybags/class_distribution.png"
plt.tight_layout()
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved → {out}")
