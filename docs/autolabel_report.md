---
title: "MOT of Polybags"
subtitle: "Auto-Labelling Pipeline for Synthetic Dataset Annotation"
date: "May 2026"
geometry: "margin=2.2cm"
fontsize: 11pt
header-includes: |
  \usepackage{booktabs}
  \usepackage{microtype}
---

## Overview

The goal was to automatically annotate a synthetic dataset of 1,157 rendered frames showing colour-coded polybags on a dark background. Only **86 frames** were hand-annotated using LabelMe (frames 0200 to 0969, selected to span variation in scene composition). The remaining **1,071 frames** required automated labelling. Six object classes were defined: *pink, blue, yellow, grey, green,* and *red* polybags. All annotations follow the YOLO OBB format (8 normalised corner coordinates per instance).

## Iterative Development (4 Rounds)

The final pipeline was reached through four rounds of development, each addressing a specific failure of the previous approach.

**Round 1: Hardcoded HSV Colour Ranges**

The first approach used manually tuned HSV ranges per class (e.g., H in [100, 130] for blue). Each per-class mask was cleaned with morphological close/open operations and fitted with `cv2.minAreaRect`. This worked on isolated bags but produced merged bounding boxes when bags of the same colour were touching or overlapping, a common occurrence in the dataset.

**Round 2: Learned GT Colour Centroids**

Instead of fixed ranges, HSV centroids were computed from the 86 ground-truth annotations by sampling the eroded interior of each annotated polygon. Classification used a nearest-centroid rule with 3x weight on the hue channel, to prioritise dominant colour over brightness variation. This eliminated manual threshold tuning and improved cross-image colour consistency, but segmentation was still contour-based and failed to split touching same-class instances.

**Round 3: Distance Transform Seeding**

A distance transform on the foreground mask was used to generate one seed per local maximum, providing a count of expected bag instances in each blob. The seeds were used as input to a marker-based approach, but without a full watershed pass the touching-bag splitting remained unreliable on crowded scenes.

**Round 4: Watershed Segmentation (Current)**

The final pipeline (`autolabel_full.py`) combines all prior improvements with a proper watershed pass:

1. **Foreground mask**: Threshold on HSV value channel (V > 120) to separate bright polybags from the dark background (background V approx. 59). Followed by MORPH\_OPEN (3x3, 1 iteration) to remove sub-pixel noise.
2. **Distance transform and local maxima**: `cv2.distanceTransform` (L2) with a light Gaussian blur to suppress shadow-induced sub-peaks. Local maxima with minimum 12 px separation (approx. half bag diameter) provide one watershed seed per instance.
3. **Watershed**: Markers built from connected components of the peak map. `cv2.watershed` correctly splits touching bags into separate regions.
4. **Colour classification**: Median HSV sampled from each watershed region interior; classified with the GT-centroid nearest-neighbour rule from Round 2. Regions smaller than 150 px are discarded as noise.
5. **OBB fitting**: `cv2.approxPolyDP` with progressive epsilon (0.02 to 0.15) to find a convex quadrilateral. Falls back to `cv2.minAreaRect` if no clean quad is found. Corners are normalised to image dimensions and written in YOLO OBB format.

## Dataset Composition

| Split | Count | Source |
|---|---|---|
| Manually labelled | 86 | LabelMe polygon annotations (JSON) |
| Auto-labelled | 1,071 | Watershed pipeline (Round 4) |
| **Total** | **1,157** | Merged; manual labels take precedence |

The merged dataset lives in `full_dataset/` with `labels_manual/`, `labels_auto/`, and `labels/` (symlinks that always point to manual if available, auto otherwise).

## Auto-Label Benchmark

The watershed pipeline was evaluated against the 86 ground-truth frames by comparing predicted instance counts and classes to the hand annotations.

| Metric | Score |
|---|---|
| Instance recall | **93.0%** (858 / 923) |
| Instance precision | **88.8%** (858 / 966) |
| Avg. delta per frame | +0.50 (slight over-detection) |

**Per-class recall:**

| Class | Recall |
|---|---|
| red\_polybag | **100.0%** (83 / 83) |
| grey\_polybag | 98.8% (85 / 86) |
| blue\_polybag | 96.7% (87 / 90) |
| green\_polybag | 94.5% (223 / 236) |
| pink\_polybag | 91.2% (155 / 170) |
| yellow\_polybag | 87.2% (225 / 258) |

Red and grey bags achieve near-perfect recall due to their distinct hue and low saturation respectively. Yellow bags show the lowest recall (87.2%), likely because their hue overlaps with specular highlights under certain lighting conditions in the renders.

## Pseudo-Label Training (YOLO11-OBB)

On top of the classical CV pipeline, an iterative pseudo-labelling loop was set up (`pseudo_label_train.py`) to progressively improve labels using the model's own predictions:

- **Round 0**: Train YOLO11n-OBB (100 epochs, 1920 px, batch 4) on the 86 GT images only. A 15% validation split is held out from GT in every round.
- **Round N**: Run inference on the 1,071 unlabelled images with the previous round's weights. Detections with confidence >= 0.35 replace the watershed auto-labels; lower-confidence frames keep the classical label. GT labels are never replaced.

This loop continues until validation mAP plateaus.
