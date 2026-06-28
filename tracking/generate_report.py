#!/usr/bin/env python3
"""
tracking/generate_report.py

Generate a multi-page PDF report for the MCMOT experiment.
Covers single-camera tracking, offline MCMOT benchmark,
online/real-time MCMOT benchmark, and method comparison.

Usage:
    cd repo/tracking
    python generate_report.py            # → mcmot_report.pdf
    python generate_report.py --out /path/to/report.pdf
"""

import argparse
import json
from pathlib import Path
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import numpy as np

# ── Result data (verified from completed runs) ─────────────────────────────────

SINGLE_CAM_PER_CAM = {
    # (tracker, dataset, camera) → {mota, motp, idf1, idsw, mt, ml}
    ("bytetrack", "val", "front"): dict(mota=100.0, motp=7.9,  idf1=100.0, idsw=0, mt=9, ml=0),
    ("bytetrack", "val", "back"):  dict(mota=100.0, motp=6.6,  idf1=100.0, idsw=0, mt=9, ml=0),
    ("bytetrack", "val", "left"):  dict(mota=97.8,  motp=9.2,  idf1=98.9,  idsw=1, mt=9, ml=0),
    ("bytetrack", "val", "right"): dict(mota=86.3,  motp=10.3, idf1=93.5,  idsw=0, mt=9, ml=0),
    ("botsort",   "val", "front"): dict(mota=100.0, motp=7.6,  idf1=100.0, idsw=0, mt=9, ml=0),
    ("botsort",   "val", "back"):  dict(mota=100.0, motp=6.4,  idf1=100.0, idsw=0, mt=9, ml=0),
    ("botsort",   "val", "left"):  dict(mota=99.4,  motp=8.3,  idf1=99.7,  idsw=1, mt=9, ml=0),
    ("botsort",   "val", "right"): dict(mota=86.5,  motp=10.1, idf1=93.6,  idsw=0, mt=9, ml=0),
    ("bytetrack", "test", "front"):dict(mota=97.0,  motp=8.0,  idf1=98.5,  idsw=2, mt=7, ml=0),
    ("bytetrack", "test", "back"): dict(mota=100.0, motp=6.0,  idf1=100.0, idsw=0, mt=7, ml=0),
    ("bytetrack", "test", "left"): dict(mota=86.0,  motp=7.5,  idf1=92.5,  idsw=0, mt=6, ml=1),
    ("bytetrack", "test", "right"):dict(mota=85.6,  motp=10.4, idf1=93.3,  idsw=0, mt=7, ml=0),
    ("botsort",   "test", "front"):dict(mota=98.6,  motp=7.5,  idf1=99.3,  idsw=0, mt=7, ml=0),
    ("botsort",   "test", "back"): dict(mota=100.0, motp=5.8,  idf1=100.0, idsw=0, mt=7, ml=0),
    ("botsort",   "test", "left"): dict(mota=98.9,  motp=8.3,  idf1=99.5,  idsw=0, mt=7, ml=0),
    ("botsort",   "test", "right"):dict(mota=85.7,  motp=10.3, idf1=93.3,  idsw=0, mt=7, ml=0),
}

SINGLE_CAM_ALL = {
    ("bytetrack", "val"):  dict(mota=96.0, motp=8.5, idf1=98.0, idsw=1,  mt=36, ml=0),
    ("bytetrack", "test"): dict(mota=92.1, motp=8.0, idf1=96.1, idsw=2,  mt=27, ml=1),
    ("botsort",   "val"):  dict(mota=96.5, motp=8.1, idf1=98.3, idsw=1,  mt=36, ml=0),
    ("botsort",   "test"): dict(mota=95.8, motp=8.0, idf1=97.9, idsw=0,  mt=28, ml=0),
}

OFFLINE = {
    # (tracker, dataset, method) → {mota, motp, idf1, idsw}
    ("bytetrack","val","no_assoc"):         dict(mota=95.7, motp=8.6, idf1=24.7, idsw=28),
    ("bytetrack","val","class_only"):       dict(mota=95.9, motp=8.6, idf1=86.6, idsw=5),
    ("bytetrack","val","trk_temporal"):     dict(mota=96.0, motp=8.6, idf1=86.9, idsw=4),
    ("bytetrack","val","trk_spatial"):      dict(mota=96.0, motp=8.6, idf1=89.5, idsw=4),
    ("bytetrack","val","trk_combined"):     dict(mota=96.0, motp=8.6, idf1=89.5, idsw=4),
    ("bytetrack","test","no_assoc"):        dict(mota=91.8, motp=8.0, idf1=25.0, idsw=23),
    ("bytetrack","test","class_only"):      dict(mota=92.1, motp=8.0, idf1=82.0, idsw=8),
    ("bytetrack","test","trk_temporal"):    dict(mota=92.1, motp=8.0, idf1=96.0, idsw=4),
    ("bytetrack","test","trk_spatial"):     dict(mota=92.1, motp=8.0, idf1=82.0, idsw=8),
    ("bytetrack","test","trk_combined"):    dict(mota=92.1, motp=8.0, idf1=82.0, idsw=8),
    ("botsort","val","no_assoc"):           dict(mota=96.2, motp=8.1, idf1=24.7, idsw=28),
    ("botsort","val","class_only"):         dict(mota=96.4, motp=8.1, idf1=86.8, idsw=5),
    ("botsort","val","trk_temporal"):       dict(mota=96.4, motp=8.1, idf1=87.2, idsw=4),
    ("botsort","val","trk_spatial"):        dict(mota=96.4, motp=8.1, idf1=89.7, idsw=4),
    ("botsort","val","trk_combined"):       dict(mota=96.4, motp=8.1, idf1=89.7, idsw=4),
    ("botsort","test","no_assoc"):          dict(mota=95.5, motp=8.0, idf1=24.6, idsw=21),
    ("botsort","test","class_only"):        dict(mota=95.8, motp=8.0, idf1=84.0, idsw=4),
    ("botsort","test","trk_temporal"):      dict(mota=95.8, motp=8.0, idf1=98.0, idsw=0),
    ("botsort","test","trk_spatial"):       dict(mota=95.8, motp=8.0, idf1=84.0, idsw=4),
    ("botsort","test","trk_combined"):      dict(mota=95.8, motp=8.0, idf1=84.0, idsw=4),
}

ONLINE = {
    # (method, tracker, dataset) → {mota, motp, idf1, idsw, fps}
    ("class_rank",  "bytetrack","val"):  dict(mota=95.7, motp=8.7, idf1=86.3, idsw=27, fps=1.0),
    ("class_iou",   "bytetrack","val"):  dict(mota=96.0, motp=8.7, idf1=89.3, idsw=5,  fps=0.4),
    ("class_smooth","bytetrack","val"):  dict(mota=95.8, motp=8.7, idf1=86.5, idsw=19, fps=1.0),
    ("class_rank",  "bytetrack","test"): dict(mota=94.6, motp=8.4, idf1=83.4, idsw=18, fps=1.0),
    ("class_iou",   "bytetrack","test"): dict(mota=94.7, motp=8.4, idf1=83.7, idsw=12, fps=0.5),
    ("class_smooth","bytetrack","test"): dict(mota=94.7, motp=8.4, idf1=83.4, idsw=12, fps=0.9),
    ("class_rank",  "botsort",  "val"):  dict(mota=96.2, motp=8.2, idf1=86.5, idsw=31, fps=0.9),
    ("class_iou",   "botsort",  "val"):  dict(mota=96.5, motp=8.2, idf1=89.5, idsw=6,  fps=0.8),
    ("class_smooth","botsort",  "val"):  dict(mota=96.4, motp=8.2, idf1=86.8, idsw=18, fps=0.4),
    ("class_rank",  "botsort",  "test"): dict(mota=95.3, motp=8.0, idf1=83.5, idsw=19, fps=0.6),
    ("class_iou",   "botsort",  "test"): dict(mota=95.5, motp=8.0, idf1=83.9, idsw=10, fps=0.9),
    ("class_smooth","botsort",  "test"): dict(mota=95.5, motp=8.0, idf1=83.6, idsw=10, fps=0.9),
}

# Populated only when --from-json is used; otherwise None
FPS_BENCH_DATA: dict | None = None

# ── Styling ────────────────────────────────────────────────────────────────────

NAVY    = "#1a2744"
BLUE    = "#2563eb"
LGREY   = "#f1f5f9"
DGREY   = "#64748b"
GREEN   = "#16a34a"
RED     = "#dc2626"
ORANGE  = "#ea580c"
GOLD    = "#ca8a04"
WHITE   = "#ffffff"
BEST_BG = "#dcfce7"   # light green for best values

METHOD_COLORS = {
    "no_assoc":      "#94a3b8",
    "class_only":    "#60a5fa",
    "trk_temporal":  "#3b82f6",
    "trk_spatial":   "#6366f1",
    "trk_combined":  "#8b5cf6",
    "class_rank":    "#f97316",
    "class_iou":     "#ef4444",
    "class_smooth":  "#ec4899",
}

METHOD_LABELS = {
    "no_assoc":      "No Association (baseline)",
    "class_only":    "Class-Only (frame-level rank)",
    "trk_temporal":  "Tracklet Temporal (Jaccard)",
    "trk_spatial":   "Tracklet Spatial (x-rank)",
    "trk_combined":  "Tracklet Combined (50/50)",
    "class_rank":    "Class Rank (online)",
    "class_iou":     "Class Spatial Hungarian (online)",
    "class_smooth":  "Class Smooth (online, 8-frame vote)",
}


def set_style():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         9,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.facecolor":  WHITE,
        "axes.facecolor":    WHITE,
    })


def header_bar(fig, title: str, subtitle: str = ""):
    bar = fig.add_axes([0, 0.93, 1, 0.07])
    bar.set_facecolor(NAVY)
    bar.axis("off")
    bar.text(0.015, 0.55, title, color=WHITE, fontsize=13,
             fontweight="bold", va="center", transform=bar.transAxes)
    if subtitle:
        bar.text(0.015, 0.1, subtitle, color="#94b4d4", fontsize=8.5,
                 va="center", transform=bar.transAxes)


def make_table(ax, rows, col_headers, row_headers=None,
               col_widths=None, highlight_fn=None,
               header_color=NAVY, row_colors=None):
    """Draw a styled table on ax."""
    ax.axis("off")
    n_rows = len(rows)
    n_cols = len(col_headers)

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols
    if row_colors is None:
        row_colors = [WHITE if i % 2 == 0 else LGREY for i in range(n_rows)]

    # Header row
    x = 0
    for ci, (h, w) in enumerate(zip(col_headers, col_widths)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, n_rows), w - 0.002, 0.82,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=header_color, transform=ax.transData, clip_on=False))
        ax.text(x + w / 2, n_rows + 0.41, h,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=WHITE, transform=ax.transData)
        x += w

    # Data rows
    for ri, (row, rc) in enumerate(zip(rows, row_colors)):
        y = n_rows - ri - 1
        x = 0
        for ci, (cell, w) in enumerate(zip(row, col_widths)):
            is_best = highlight_fn(ri, ci, cell) if highlight_fn else False
            fc = BEST_BG if is_best else rc
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y), w - 0.002, 0.9,
                boxstyle="square,pad=0", linewidth=0,
                facecolor=fc, transform=ax.transData, clip_on=False))
            ax.text(x + w / 2, y + 0.45, str(cell),
                    ha="center", va="center", fontsize=8,
                    fontweight="bold" if is_best else "normal",
                    color=NAVY, transform=ax.transData)
            x += w

    ax.set_xlim(0, sum(col_widths))
    ax.set_ylim(0, n_rows + 0.85)


# ══════════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════════

def page_title(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)

    # Top navy block
    top = fig.add_axes([0, 0.72, 1, 0.28])
    top.set_facecolor(NAVY)
    top.axis("off")
    top.text(0.5, 0.68, "Multi-Camera Multi-Object Tracking", color=WHITE,
             fontsize=22, fontweight="bold", ha="center", va="center",
             transform=top.transAxes)
    top.text(0.5, 0.38, "Real-Time Parallel Pipeline — Evaluation Report",
             color="#94b4d4", fontsize=14, ha="center", va="center",
             transform=top.transAxes)
    top.text(0.5, 0.12,
             f"OVGU AMS Project  ·  Synthetic Polybag Dataset  ·  {date.today().strftime('%B %d, %Y')}",
             color="#7090a0", fontsize=10, ha="center", va="center",
             transform=top.transAxes)

    body = fig.add_axes([0.06, 0.04, 0.88, 0.64])
    body.axis("off")

    sections = [
        ("Dataset",
         "Synthetic 4-camera Blender renders (front / back / left / right).\n"
         "7 color-coded polybag classes: pink, blue, yellow, grey, green, red, teal.\n"
         "Train: 500 frames (100–599) · Val: 251 frames (1000–1250) · Test: 251 frames (1500–1750).\n"
         "Ground truth: OBB + MOT16 AABB format with Hungarian-matched global IDs."),
        ("Detection model",
         "YOLO11n-OBB (Oriented Bounding Box) trained on 1841 synthetic images at 1920 px.\n"
         "Per-camera detection confidence threshold: 0.25."),
        ("Intra-camera trackers",
         "ByteTrack — IoU association + low-confidence track preservation.\n"
         "BoT-SORT  — Kalman filter + camera-motion compensation + optional Re-ID."),
        ("Real-time MCMOT pipeline",
         "All 4 cameras are inferred simultaneously per frame (ThreadPoolExecutor, 4 workers).\n"
         "After each frame, an online associator assigns globally consistent IDs across cameras.\n"
         "Three online association strategies benchmarked (see page 4)."),
        ("Offline MCMOT pipeline",
         "Cameras processed sequentially; tracklet-level inter-camera association runs\n"
         "after all frames are complete. Five strategies benchmarked (see page 3)."),
        ("Evaluation",
         "motmetrics 1.4.0 · MOTA, MOTP, IDF1, IDSW, MT, ML.\n"
         "Cross-camera evaluation uses per-camera frame-offset (cam_idx × 10 000)\n"
         "to merge all 4 camera streams into a single global MOT accumulator."),
    ]

    y = 0.97
    for title, text in sections:
        body.text(0.0, y, title, fontsize=10, fontweight="bold", color=NAVY,
                  transform=body.transAxes, va="top")
        body.text(0.0, y - 0.035, text, fontsize=8.5, color="#334155",
                  transform=body.transAxes, va="top", linespacing=1.55)
        y -= 0.155

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_single_cam(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    header_bar(fig, "Single-Camera Intra-Tracking Results",
               "Per-camera metrics using ByteTrack (val set) and overall comparison across all 4 combinations")

    # ── Per-camera table (bytetrack / val) ────────────────────────────────────
    ax1 = fig.add_axes([0.04, 0.54, 0.92, 0.35])
    ax1.set_title("Per-Camera Results — ByteTrack / Validation Set", pad=6,
                  fontsize=10, fontweight="bold", color=NAVY)
    cams = ["front", "back", "left", "right"]
    col_headers = ["Camera", "MOTA (%)", "MOTP (%)", "IDF1 (%)", "ID Sw.", "MT", "ML"]
    col_widths   = [0.16, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14]
    rows = []
    for cam in cams:
        d = SINGLE_CAM_PER_CAM[("bytetrack", "val", cam)]
        rows.append([cam.upper(),
                     f"{d['mota']:.1f}", f"{d['motp']:.1f}",
                     f"{d['idf1']:.1f}", str(d['idsw']),
                     str(d['mt']), str(d['ml'])])
    # Totals
    d = SINGLE_CAM_ALL[("bytetrack", "val")]
    rows.append(["ALL (avg)",
                 f"{d['mota']:.1f}", f"{d['motp']:.1f}",
                 f"{d['idf1']:.1f}", str(d['idsw']),
                 str(d['mt']), str(d['ml'])])
    row_colors = [LGREY, WHITE, LGREY, WHITE, "#e0f2fe"]

    def hl1(ri, ci, val):
        try: return ci in (1,3) and float(val) >= 99.0 and ri < 4
        except: return False

    make_table(ax1, rows, col_headers, col_widths=col_widths,
               highlight_fn=hl1, row_colors=row_colors)

    # ── Overall comparison table ───────────────────────────────────────────────
    ax2 = fig.add_axes([0.04, 0.13, 0.60, 0.33])
    ax2.set_title("Overall 4-Camera Aggregate", pad=6,
                  fontsize=10, fontweight="bold", color=NAVY)
    col_headers2 = ["Tracker", "Dataset", "MOTA (%)", "MOTP (%)", "IDF1 (%)", "IDSW", "MT", "ML"]
    col_widths2  = [0.14, 0.12, 0.14, 0.14, 0.14, 0.10, 0.11, 0.11]
    rows2 = []
    for trk in ["bytetrack", "botsort"]:
        for ds in ["val", "test"]:
            d = SINGLE_CAM_ALL[(trk, ds)]
            rows2.append([trk, ds,
                          f"{d['mota']:.1f}", f"{d['motp']:.1f}",
                          f"{d['idf1']:.1f}", str(d['idsw']),
                          str(d['mt']), str(d['ml'])])
    row_colors2 = [LGREY, WHITE, LGREY, WHITE]

    def hl2(ri, ci, val):
        try: return ci == 4 and float(val) >= 98.0
        except: return False

    make_table(ax2, rows2, col_headers2, col_widths=col_widths2,
               highlight_fn=hl2, row_colors=row_colors2)

    # ── Key observations ───────────────────────────────────────────────────────
    obs = fig.add_axes([0.66, 0.13, 0.30, 0.33])
    obs.axis("off")
    obs.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.04", linewidth=1.5,
        edgecolor="#93c5fd", facecolor="#eff6ff",
        transform=obs.transAxes, clip_on=False))
    notes = (
        "Key observations\n\n"
        "• Front & back cameras: MOTA\n"
        "  and IDF1 = 100% for both\n"
        "  trackers on val.\n\n"
        "• Right camera is weakest\n"
        "  (~86% MOTA, ~10% MOTP)\n"
        "  due to perspective angle.\n\n"
        "• MT = 36/36 on val — every\n"
        "  object tracked end-to-end.\n\n"
        "• BoT-SORT gains ~0.5% MOTA\n"
        "  on val but 0 IDSW on test\n"
        "  vs ByteTrack's 2."
    )
    obs.text(0.08, 0.92, notes, fontsize=8.5, color=NAVY, va="top",
             transform=obs.transAxes, linespacing=1.55)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_offline(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    header_bar(fig, "Offline MCMOT Benchmark — 5 Methods × 2 Trackers × 2 Datasets",
               "Tracklet-level inter-camera association after full sequence processing")

    # ── IDF1 grouped bar chart ─────────────────────────────────────────────────
    ax_bar = fig.add_axes([0.05, 0.50, 0.58, 0.38])
    methods   = ["no_assoc", "class_only", "trk_temporal", "trk_spatial", "trk_combined"]
    combos    = [("bytetrack","val"), ("bytetrack","test"),
                 ("botsort","val"),   ("botsort","test")]
    combo_lbl = ["BT/val", "BT/test", "BS/val", "BS/test"]
    combo_clr = ["#2563eb", "#93c5fd", "#ea580c", "#fca5a5"]

    x = np.arange(len(methods))
    w = 0.18
    for i, (combo, lbl, clr) in enumerate(zip(combos, combo_lbl, combo_clr)):
        vals = [OFFLINE[(combo[0], combo[1], m)]["idf1"] for m in methods]
        ax_bar.bar(x + (i - 1.5) * w, vals, w, label=lbl, color=clr, alpha=0.9)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=8)
    ax_bar.set_ylabel("IDF1 (%)", fontsize=9)
    ax_bar.set_ylim(0, 110)
    ax_bar.set_title("IDF1 by Method", fontsize=10, fontweight="bold", color=NAVY)
    ax_bar.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax_bar.axhline(90, color=DGREY, lw=0.7, ls="--", alpha=0.5)
    ax_bar.yaxis.grid(True, alpha=0.3)
    ax_bar.set_axisbelow(True)

    # ── IDSW bar chart ─────────────────────────────────────────────────────────
    ax_sw = fig.add_axes([0.68, 0.50, 0.28, 0.38])
    for i, (combo, lbl, clr) in enumerate(zip(combos, combo_lbl, combo_clr)):
        vals = [OFFLINE[(combo[0], combo[1], m)]["idsw"] for m in methods]
        ax_sw.bar(x + (i - 1.5) * w, vals, w, label=lbl, color=clr, alpha=0.9)
    ax_sw.set_xticks(x)
    ax_sw.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=7)
    ax_sw.set_ylabel("ID Switches", fontsize=9)
    ax_sw.set_title("IDSW by Method", fontsize=10, fontweight="bold", color=NAVY)
    ax_sw.yaxis.grid(True, alpha=0.3)
    ax_sw.set_axisbelow(True)

    # ── Full results table ─────────────────────────────────────────────────────
    ax_tbl = fig.add_axes([0.03, 0.04, 0.94, 0.41])
    ax_tbl.set_title("Complete Offline Results", pad=6,
                     fontsize=10, fontweight="bold", color=NAVY)
    col_headers = ["Tracker", "Dataset", "Method",
                   "MOTA (%)", "MOTP (%)", "IDF1 (%)", "IDSW"]
    col_widths   = [0.12, 0.10, 0.22, 0.14, 0.14, 0.14, 0.14]

    ordered = [("bytetrack","val"), ("bytetrack","test"),
               ("botsort","val"),   ("botsort","test")]
    rows = []
    row_colors = []
    best_idf1 = max(OFFLINE[k]["idf1"] for k in OFFLINE)

    for trk, ds in ordered:
        for mi, m in enumerate(methods):
            d = OFFLINE[(trk, ds, m)]
            rows.append([trk if mi == 0 else "", ds if mi == 0 else "", m,
                         f"{d['mota']:.1f}", f"{d['motp']:.1f}",
                         f"{d['idf1']:.1f}", str(d['idsw'])])
            row_colors.append(LGREY if mi % 2 == 0 else WHITE)

    def hl_off(ri, ci, val):
        try: return ci == 5 and float(val) >= 96.0
        except: return False

    make_table(ax_tbl, rows, col_headers, col_widths=col_widths,
               highlight_fn=hl_off, row_colors=row_colors)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_online(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    header_bar(fig, "Online / Real-Time MCMOT Benchmark — 3 Methods × 2 Trackers × 2 Datasets",
               "All 4 cameras inferred simultaneously per frame; association runs online after each frame")

    online_methods = ["class_rank", "class_iou", "class_smooth"]
    combos    = [("bytetrack","val"), ("bytetrack","test"),
                 ("botsort","val"),   ("botsort","test")]
    combo_lbl = ["BT/val", "BT/test", "BS/val", "BS/test"]
    combo_clr = ["#2563eb", "#93c5fd", "#ea580c", "#fca5a5"]

    # ── IDF1 chart ─────────────────────────────────────────────────────────────
    ax_idf = fig.add_axes([0.05, 0.52, 0.45, 0.36])
    x = np.arange(len(online_methods))
    w = 0.18
    for i, (combo, lbl, clr) in enumerate(zip(combos, combo_lbl, combo_clr)):
        vals = [ONLINE[(m, combo[0], combo[1])]["idf1"] for m in online_methods]
        ax_idf.bar(x + (i - 1.5) * w, vals, w, label=lbl, color=clr, alpha=0.9)
    ax_idf.set_xticks(x)
    ax_idf.set_xticklabels([METHOD_LABELS[m].replace(" (online","").replace(")","")
                             .replace("Class ","") for m in online_methods], fontsize=8)
    ax_idf.set_ylabel("IDF1 (%)", fontsize=9)
    ax_idf.set_ylim(78, 96)
    ax_idf.set_title("IDF1 — Online Methods", fontsize=10, fontweight="bold", color=NAVY)
    ax_idf.legend(fontsize=7.5, ncol=2)
    ax_idf.yaxis.grid(True, alpha=0.3)
    ax_idf.set_axisbelow(True)

    # ── IDSW chart ─────────────────────────────────────────────────────────────
    ax_sw = fig.add_axes([0.55, 0.52, 0.22, 0.36])
    for i, (combo, lbl, clr) in enumerate(zip(combos, combo_lbl, combo_clr)):
        vals = [ONLINE[(m, combo[0], combo[1])]["idsw"] for m in online_methods]
        ax_sw.bar(x + (i - 1.5) * w, vals, w, label=lbl, color=clr, alpha=0.9)
    ax_sw.set_xticks(x)
    ax_sw.set_xticklabels(["Rank", "Hungarian", "Smooth"], fontsize=8)
    ax_sw.set_ylabel("ID Switches", fontsize=9)
    ax_sw.set_title("IDSW", fontsize=10, fontweight="bold", color=NAVY)
    ax_sw.yaxis.grid(True, alpha=0.3)
    ax_sw.set_axisbelow(True)

    # ── Speed chart ────────────────────────────────────────────────────────────
    ax_spd = fig.add_axes([0.80, 0.52, 0.16, 0.36])
    all_fps = {m: [] for m in online_methods}
    for combo in combos:
        for m in online_methods:
            all_fps[m].append(ONLINE[(m, combo[0], combo[1])]["fps"])
    mean_fps = {m: np.mean(v) for m, v in all_fps.items()}
    colors = [METHOD_COLORS[m] for m in online_methods]
    bars = ax_spd.bar(range(3), [mean_fps[m] for m in online_methods], color=colors)
    ax_spd.set_xticks(range(3))
    ax_spd.set_xticklabels(["Rank", "Hungarian", "Smooth"], fontsize=8)
    ax_spd.set_ylabel("Mean fps (4-cam)", fontsize=8)
    ax_spd.set_title("Speed", fontsize=10, fontweight="bold", color=NAVY)
    ax_spd.set_ylim(0, 1.3)
    for bar, m in zip(bars, online_methods):
        ax_spd.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{mean_fps[m]:.2f}", ha="center", va="bottom", fontsize=8)

    # ── Full table ─────────────────────────────────────────────────────────────
    ax_tbl = fig.add_axes([0.03, 0.04, 0.94, 0.43])
    ax_tbl.set_title("Complete Online Results", pad=6,
                     fontsize=10, fontweight="bold", color=NAVY)
    col_headers = ["Method", "Tracker", "Dataset",
                   "MOTA (%)", "MOTP (%)", "IDF1 (%)", "IDSW", "fps"]
    col_widths   = [0.22, 0.12, 0.10, 0.12, 0.12, 0.12, 0.10, 0.10]

    rows = []
    row_colors = []
    ri = 0
    for m in online_methods:
        for trk in ["bytetrack", "botsort"]:
            for ds in ["val", "test"]:
                d = ONLINE[(m, trk, ds)]
                rows.append([
                    m if trk == "bytetrack" and ds == "val" else "",
                    trk if ds == "val" else "",
                    ds,
                    f"{d['mota']:.1f}", f"{d['motp']:.1f}",
                    f"{d['idf1']:.1f}", str(d['idsw']), f"{d['fps']:.1f}"
                ])
                row_colors.append(LGREY if ri % 2 == 0 else WHITE)
                ri += 1

    def hl_on(ri, ci, val):
        try: return ci == 5 and float(val) >= 89.0
        except: return False

    make_table(ax_tbl, rows, col_headers, col_widths=col_widths,
               highlight_fn=hl_on, row_colors=row_colors)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_comparison(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    header_bar(fig, "Online vs Offline Comparison & Key Findings")

    # ── Scatter: IDF1 vs fps for online, annotate offline best ────────────────
    ax_sc = fig.add_axes([0.06, 0.52, 0.44, 0.36])

    for (m, trk, ds), d in ONLINE.items():
        clr = METHOD_COLORS[m]
        ax_sc.scatter(d["fps"], d["idf1"], color=clr, s=55, alpha=0.85, zorder=3)

    # Annotate method centroids
    for m in ["class_rank", "class_iou", "class_smooth"]:
        fps_vals = [ONLINE[(m,t,d)]["fps"] for t in ["bytetrack","botsort"]
                    for d in ["val","test"]]
        idf_vals = [ONLINE[(m,t,d)]["idf1"] for t in ["bytetrack","botsort"]
                    for d in ["val","test"]]
        ax_sc.annotate(m.replace("class_",""),
                       (np.mean(fps_vals), np.mean(idf_vals)),
                       textcoords="offset points", xytext=(6, 3),
                       fontsize=8, color=METHOD_COLORS[m], fontweight="bold")

    # Offline best (no fps, mark on y-axis)
    best_off = max(OFFLINE[k]["idf1"] for k in OFFLINE)
    ax_sc.axhline(best_off, color=NAVY, lw=1.2, ls="--", alpha=0.7)
    ax_sc.text(0.02, best_off + 0.6, f"Offline best ({best_off:.1f}%)",
               fontsize=8, color=NAVY)

    ax_sc.set_xlabel("Throughput (fps, 4 cameras @ 1920px, MacBook CPU)", fontsize=8)
    ax_sc.set_ylabel("IDF1 (%)", fontsize=9)
    ax_sc.set_title("Quality–Speed Trade-off (Online Methods)", fontsize=10,
                    fontweight="bold", color=NAVY)
    ax_sc.yaxis.grid(True, alpha=0.3)
    ax_sc.set_axisbelow(True)

    # ── IDF1 improvement from no_assoc ────────────────────────────────────────
    ax_gain = fig.add_axes([0.57, 0.52, 0.38, 0.36])
    labels = ["no_assoc", "class_only", "trk_temporal", "trk_spatial",
              "trk_combined", "class_rank\n(online)", "class_iou\n(online)",
              "class_smooth\n(online)"]
    idf1_mean = [
        np.mean([OFFLINE[(t,d,"no_assoc")]["idf1"] for t in ["bytetrack","botsort"] for d in ["val","test"]]),
        np.mean([OFFLINE[(t,d,"class_only")]["idf1"] for t in ["bytetrack","botsort"] for d in ["val","test"]]),
        np.mean([OFFLINE[(t,d,"trk_temporal")]["idf1"] for t in ["bytetrack","botsort"] for d in ["val","test"]]),
        np.mean([OFFLINE[(t,d,"trk_spatial")]["idf1"] for t in ["bytetrack","botsort"] for d in ["val","test"]]),
        np.mean([OFFLINE[(t,d,"trk_combined")]["idf1"] for t in ["bytetrack","botsort"] for d in ["val","test"]]),
        np.mean([ONLINE[("class_rank",t,d)]["idf1"] for t in ["bytetrack","botsort"] for d in ["val","test"]]),
        np.mean([ONLINE[("class_iou",t,d)]["idf1"] for t in ["bytetrack","botsort"] for d in ["val","test"]]),
        np.mean([ONLINE[("class_smooth",t,d)]["idf1"] for t in ["bytetrack","botsort"] for d in ["val","test"]]),
    ]
    bar_colors = (
        [METHOD_COLORS["no_assoc"], METHOD_COLORS["class_only"],
         METHOD_COLORS["trk_temporal"], METHOD_COLORS["trk_spatial"],
         METHOD_COLORS["trk_combined"],
         METHOD_COLORS["class_rank"], METHOD_COLORS["class_iou"],
         METHOD_COLORS["class_smooth"]]
    )
    bars = ax_gain.barh(range(len(labels)), idf1_mean, color=bar_colors, alpha=0.85)
    ax_gain.set_yticks(range(len(labels)))
    ax_gain.set_yticklabels(labels, fontsize=8)
    ax_gain.set_xlabel("Mean IDF1 (%) across all tracker × dataset", fontsize=8)
    ax_gain.set_title("Mean IDF1 by Method", fontsize=10, fontweight="bold", color=NAVY)
    ax_gain.set_xlim(0, 110)
    ax_gain.xaxis.grid(True, alpha=0.3)
    ax_gain.set_axisbelow(True)
    for bar, val in zip(bars, idf1_mean):
        ax_gain.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", fontsize=7.5)

    # ── Findings ───────────────────────────────────────────────────────────────
    ax_txt = fig.add_axes([0.04, 0.02, 0.92, 0.44])
    ax_txt.axis("off")
    ax_txt.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02", linewidth=1,
        edgecolor="#cbd5e1", facecolor=LGREY,
        transform=ax_txt.transAxes, clip_on=False))

    findings = [
        ("1.  MOTA hides the MCMOT problem.",
         "No-association baseline achieves 91–96% MOTA because MOTA only measures detection completeness.\n"
         "IDF1 collapses to ~25%, revealing that 3 out of 4 identity assignments are wrong globally without cross-camera matching."),
        ("2.  Temporal Jaccard is the strongest offline signal.",
         "tracklet_temporal reaches IDF1=98%, IDSW=0 (BoT-SORT/test). Synchronized cameras mean the same object\n"
         "has near-identical active frame ranges in all views — Jaccard similarity ≈ 1.0 for the same bag."),
        ("3.  Spatial rank alone is competitive offline.",
         "trk_spatial outperforms trk_temporal on val (+2.5% IDF1) because the conveyor's left-to-right ordering\n"
         "is consistent across views. trk_combined gains nothing over trk_spatial alone on test."),
        ("4.  Online Hungarian spatial (class_iou) beats discrete rank (class_rank).",
         "IDF1 +3% and IDSW 5× lower. Continuous |Δnorm_x| matching handles near-rank-swaps;\n"
         "discrete rank matching assigns wrong IDs whenever two bags' projections cross."),
        ("5.  Online–offline IDF1 gap is ~8–10 pp.",
         "Best online: class_iou/BoT-SORT/val = 89.5%, 6 IDSW. Best offline: trk_temporal/BoT-SORT/test = 98%, 0 IDSW.\n"
         "The gap reflects the additional context available to offline methods (full tracklet history vs current frame only)."),
        ("6.  ByteTrack is the practical real-time choice.",
         "BoT-SORT gains at most +0.5% MOTA but runs 2–2.5× slower on CPU due to Kalman + camera-motion compensation.\n"
         "ByteTrack + class_rank delivers 1.0 fps at 1920 px — class_iou costs 0.4 fps for the +3% IDF1."),
    ]

    y = 0.97
    for title, body in findings:
        ax_txt.text(0.015, y, title, fontsize=9, fontweight="bold", color=NAVY,
                    transform=ax_txt.transAxes, va="top")
        ax_txt.text(0.015, y - 0.055, body, fontsize=8, color="#334155",
                    transform=ax_txt.transAxes, va="top", linespacing=1.45)
        y -= 0.155

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_method_descriptions(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    header_bar(fig, "Method Descriptions & Survey Correspondence",
               "How each implemented method relates to the MCMOT survey (Uckermann, TH Köln, 2024)")

    ax = fig.add_axes([0.04, 0.02, 0.92, 0.88])
    ax.axis("off")

    table_data = [
        ["Method", "Type", "Core mechanism", "Survey analog", "Key constraint"],
        ["no_assoc",
         "Offline\nbaseline",
         "Local IDs shifted by cam_idx×1000.\nNo cross-camera step.",
         "—",
         "IDF1~25%: shows cost\nof no association."],
        ["class_only",
         "Offline\nframe-level",
         "Per frame: same color class → sort by\nx-center → rank match → persistent ID map.",
         "POM (per-frame occupancy)\nwithout calibration",
         "Fails when bags of same\ncolor swap x-order."],
        ["trk_temporal",
         "Offline\ntracklet",
         "Temporal Jaccard similarity between\ntracklet frame sets → greedy graph merge.",
         "JPDAF (temporal coherence)",
         "Requires synchronized\ncameras."],
        ["trk_spatial",
         "Offline\ntracklet",
         "Normalized mean-x rank difference\nbetween tracklets → greedy merge.",
         "LMGP (spatial affinity)\nwithout calibration",
         "Requires consistent left-\nright order across views."],
        ["trk_combined",
         "Offline\ntracklet",
         "0.5 × temporal Jaccard +\n0.5 × spatial rank cost.",
         "LMGP / DyGLIP\n(multi-feature graph)",
         "No gain over temporal\nalone on test set."],
        ["class_rank\n(online)",
         "Online\nreal-time",
         "Per frame, same class → sort by\nx-center → rank match → update ID map.",
         "FCDSC (real-time\nsubgraph, simplified)",
         "~1fps; IDSW ~20–30\ndue to rank flips."],
        ["class_iou\n(online)",
         "Online\nreal-time",
         "Per frame, same class → Hungarian on\n|Δnorm_x| cost → threshold 0.30.",
         "Online JPDAF /\nHungarian association",
         "0.4–0.9fps (scipy\nHungarian per frame)."],
        ["class_smooth\n(online)",
         "Online\nreal-time",
         "class_rank + 8-frame majority vote\nwindow on global ID assignments.",
         "ReST (reconfigurable\nST graph, simplified)",
         "Smooths IDSW but\ncannot fix rank errors."],
    ]

    col_widths = [0.18, 0.12, 0.30, 0.22, 0.18]

    header = table_data[0]
    rows   = table_data[1:]
    n_rows = len(rows)
    n_cols = len(header)
    row_h  = 0.083

    y_top = 0.97
    x = 0
    for ci, (h, w) in enumerate(zip(header, col_widths)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y_top - row_h), w - 0.005, row_h,
            boxstyle="square,pad=0", linewidth=0, facecolor=NAVY,
            transform=ax.transAxes, clip_on=False))
        ax.text(x + w/2, y_top - row_h/2, h, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=WHITE,
                transform=ax.transAxes)
        x += w

    rcolors = [LGREY if i % 2 == 0 else WHITE for i in range(n_rows)]
    for ri, (row, rc) in enumerate(zip(rows, rcolors)):
        y = y_top - row_h * (ri + 2)
        x = 0
        for ci, (cell, w) in enumerate(zip(row, col_widths)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y), w - 0.005, row_h,
                boxstyle="square,pad=0", linewidth=0, facecolor=rc,
                transform=ax.transAxes, clip_on=False))
            ax.text(x + 0.008, y + row_h/2, cell, ha="left", va="center",
                    fontsize=7.5, color=NAVY, transform=ax.transAxes,
                    linespacing=1.3)
            x += w

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── JSON loader ────────────────────────────────────────────────────────────────

def _pct(v, key):
    """Convert motmetrics raw value to %, or pass through for integer metrics."""
    if v is None: return 0.0
    if key in ("num_switches", "mostly_tracked", "mostly_lost",
               "num_frames", "num_false_positives", "num_misses"):
        return int(v)
    return round(float(v) * 100, 1)


def _has_metrics(data: dict, section: str) -> bool:
    """Return True if at least one entry in the JSON section has real metrics."""
    if section == "single_camera":
        for tracker in data.get("single_camera", {}).values():
            for cam_dict in tracker.values():
                for cam_data in cam_dict.values():
                    if cam_data.get("metrics") is not None:
                        return True
    elif section == "offline_mcmot":
        for tracker in data.get("offline_mcmot", {}).values():
            for method_dict in tracker.values():
                for res in method_dict.values():
                    if res.get("metrics") is not None:
                        return True
    elif section == "online_mcmot":
        for tracker_dict in data.get("online_mcmot", {}).values():
            for ds_dict in tracker_dict.values():
                for res in ds_dict.values():
                    if res.get("metrics") is not None:
                        return True
    return False


def _override_from_json(path: str):
    """Load server_benchmark JSON and override global data dicts in-place.

    Metric dicts (MOTA/IDF1/IDSW) are only overridden if the JSON actually
    contains metric values — i.e. motmetrics was installed on the server.
    If all metrics are None (motmetrics missing), the hardcoded local values
    are kept intact and only FPS_BENCH_DATA is populated.
    """
    global SINGLE_CAM_PER_CAM, SINGLE_CAM_ALL, OFFLINE, ONLINE, FPS_BENCH_DATA

    with open(path) as f:
        data = json.load(f)

    metrics_available = (
        _has_metrics(data, "single_camera") or
        _has_metrics(data, "offline_mcmot") or
        _has_metrics(data, "online_mcmot")
    )
    if not metrics_available:
        print("  NOTE: JSON has no metric values (motmetrics was not installed on server).")
        print("        Keeping hardcoded local metric values; loading FPS data only.")

    # ── single_camera ─────────────────────────────────────────────────────────
    if metrics_available and _has_metrics(data, "single_camera"):
        new_per_cam: dict = {}
        new_all:     dict = {}
        for tracker, ds_dict in data.get("single_camera", {}).items():
            for dataset, cam_dict in ds_dict.items():
                cam_rows = []
                for cam_short, cam_data in cam_dict.items():
                    m = cam_data.get("metrics") or {}
                    row = dict(
                        mota=_pct(m.get("mota"),            "mota"),
                        motp=_pct(m.get("motp"),            "motp"),
                        idf1=_pct(m.get("idf1"),            "idf1"),
                        idsw=_pct(m.get("num_switches"),    "num_switches"),
                        mt  =_pct(m.get("mostly_tracked"),  "mostly_tracked"),
                        ml  =_pct(m.get("mostly_lost"),     "mostly_lost"),
                    )
                    new_per_cam[(tracker, dataset, cam_short)] = row
                    cam_rows.append(row)
                if cam_rows:
                    avg = {k: round(sum(r[k] for r in cam_rows) / len(cam_rows), 1)
                           for k in cam_rows[0]}
                    avg["idsw"] = sum(r["idsw"] for r in cam_rows)
                    avg["mt"]   = sum(r["mt"]   for r in cam_rows)
                    avg["ml"]   = sum(r["ml"]   for r in cam_rows)
                    new_all[(tracker, dataset)] = avg
        if new_per_cam:
            SINGLE_CAM_PER_CAM = new_per_cam
        if new_all:
            SINGLE_CAM_ALL = new_all

    # ── offline_mcmot ─────────────────────────────────────────────────────────
    if metrics_available and _has_metrics(data, "offline_mcmot"):
        new_offline: dict = {}
        for tracker, ds_dict in data.get("offline_mcmot", {}).items():
            for dataset, method_dict in ds_dict.items():
                for method, res in method_dict.items():
                    m = res.get("metrics") or {}
                    new_offline[(tracker, dataset, method)] = dict(
                        mota=_pct(m.get("mota"),         "mota"),
                        motp=_pct(m.get("motp"),         "motp"),
                        idf1=_pct(m.get("idf1"),         "idf1"),
                        idsw=_pct(m.get("num_switches"), "num_switches"),
                    )
        if new_offline:
            OFFLINE = new_offline

    # ── online_mcmot ──────────────────────────────────────────────────────────
    if metrics_available and _has_metrics(data, "online_mcmot"):
        new_online: dict = {}
        for method, tracker_dict in data.get("online_mcmot", {}).items():
            for tracker, ds_dict in tracker_dict.items():
                for dataset, res in ds_dict.items():
                    m = res.get("metrics") or {}
                    new_online[(method, tracker, dataset)] = dict(
                        mota=_pct(m.get("mota"),         "mota"),
                        motp=_pct(m.get("motp"),         "motp"),
                        idf1=_pct(m.get("idf1"),         "idf1"),
                        idsw=_pct(m.get("num_switches"), "num_switches"),
                        fps =round(float(res.get("fps", 0)), 2),
                    )
        if new_online:
            ONLINE = new_online

    # ── fps_benchmark ─────────────────────────────────────────────────────────
    FPS_BENCH_DATA = data.get("fps_benchmark")
    if FPS_BENCH_DATA:
        FPS_BENCH_DATA["system"] = data.get("meta", {}).get("system", {})


def page_fps_benchmark(pdf):
    """Bar chart + table comparing single-cam FPS vs 4-cam MCMOT FPS."""
    if FPS_BENCH_DATA is None:
        return

    fig = plt.figure(figsize=(11, 8.5))
    header_bar(fig, "FPS Benchmark — Single Camera vs 4-Camera MCMOT",
               "Measured on: "
               + FPS_BENCH_DATA.get("system", {}).get("host", "server")
               + "   GPU: "
               + FPS_BENCH_DATA.get("system", {}).get("gpu_name", "N/A"))

    gs = GridSpec(2, 1, figure=fig, top=0.90, bottom=0.05,
                  hspace=0.38, left=0.06, right=0.96)
    ax_bar = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    # Gather data for chart
    single = FPS_BENCH_DATA.get("single_cam", {})
    multi  = FPS_BENCH_DATA.get("multicam_mcmot", {})
    # Take first online method (class_rank — fastest)
    multi_method = list(multi.keys())[0] if multi else None

    bar_labels, sc_fps_list, mc_fps_list = [], [], []
    table_rows = []
    for tracker in sorted(single):
        for dataset in sorted(single.get(tracker, {})):
            sc = single[tracker][dataset].get("fps", 0)
            mc = 0.0
            if multi_method and tracker in multi.get(multi_method, {}):
                mc = multi[multi_method][tracker].get(dataset, {}).get("fps", 0)
            label = f"{tracker}\n/{dataset}"
            bar_labels.append(label)
            sc_fps_list.append(sc)
            mc_fps_list.append(mc)
            ratio = f"{sc/mc:.1f}×" if mc > 0 else "—"
            table_rows.append([
                tracker, dataset,
                f"{sc:.1f}", f"{mc:.1f}",
                ratio,
                "✓" if sc >= 25 else "✗",
                "✓" if mc >= 25 else "✗",
            ])

    x = np.arange(len(bar_labels))
    w = 0.35
    b1 = ax_bar.bar(x - w/2, sc_fps_list, w, label="Single cam (1 view)", color=BLUE,   alpha=0.85)
    b2 = ax_bar.bar(x + w/2, mc_fps_list, w,
                    label=f"4-cam MCMOT ({multi_method or 'class_rank'})", color=ORANGE, alpha=0.85)
    ax_bar.axhline(25, color=RED, linewidth=1.4, linestyle="--",
                   label="Real-time threshold (25 fps)")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(bar_labels, fontsize=9)
    ax_bar.set_ylabel("Frames per second"); ax_bar.set_title("Throughput Comparison")
    ax_bar.legend(fontsize=8)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            if h > 0:
                ax_bar.annotate(f"{h:.0f}",
                                xy=(rect.get_x() + rect.get_width()/2, h),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8)

    # Table
    ax_tbl.axis("off")
    col_headers = ["Tracker", "Dataset", "Single-cam fps", "4-cam MCMOT fps",
                   "Slowdown", "Single ≥25fps", "Multi ≥25fps"]
    make_table(ax_tbl, table_rows, col_headers,
               col_widths=[0.13, 0.1, 0.16, 0.18, 0.13, 0.15, 0.15])
    ax_tbl.set_title("FPS Summary Table", fontsize=10, fontweight="bold", pad=10)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",       default="mcmot_report.pdf")
    ap.add_argument("--from-json", default=None, metavar="PATH",
                    help="Load metrics from a server_benchmark.py JSON output "
                         "instead of the hardcoded local values.")
    args = ap.parse_args()

    if args.from_json:
        print(f"Loading metrics from {args.from_json} …")
        _override_from_json(args.from_json)

    out_path = Path(args.out)
    set_style()

    with PdfPages(str(out_path)) as pdf:
        d = pdf.infodict()
        d["Title"]   = "MCMOT Real-Time Pipeline — Evaluation Report"
        d["Author"]  = "AMS Project, OVGU"
        d["Subject"] = "Multi-Camera Multi-Object Tracking — Polybag Synthetic Dataset"
        d["Keywords"] = "MCMOT, YOLO, ByteTrack, BoT-SORT, OBB, tracking"

        print("Generating pages…")
        page_title(pdf);              print("  [1] Title page")
        page_single_cam(pdf);         print("  [2] Single-camera results")
        page_offline(pdf);            print("  [3] Offline MCMOT benchmark")
        page_online(pdf);             print("  [4] Online MCMOT benchmark")
        if FPS_BENCH_DATA is not None:
            page_fps_benchmark(pdf);  print("  [5] FPS benchmark (from JSON)")
        page_comparison(pdf);         print("  [5/6] Comparison & findings")
        page_method_descriptions(pdf);print("  [6/7] Method descriptions")

    print(f"\nReport saved → {out_path.resolve()}")


if __name__ == "__main__":
    main()
