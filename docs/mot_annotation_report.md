---
geometry: margin=1.8cm
fontsize: 10.5pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{caption}
  - \usepackage{graphicx}
  - \captionsetup{font=small}
---

\begin{center}
{\LARGE\bfseries OBB-MOT Ground-Truth Annotation via Blender}\\[0.4em]
{\large Multi-Camera MOT of Polybags, 7 Classes}\\[0.2em]
{\small May 2026}
\end{center}
\vspace{-1em}
\noindent\rule{\textwidth}{0.4pt}
\vspace{0.5em}

## Background

Multi-Camera Multi-Object Tracking (MCMOT) benchmarking requires ground-truth annotations that carry a globally consistent object ID across every frame and every camera simultaneously. Standard MOT16 format uses axis-aligned bounding boxes and was designed for single-camera sequences. Existing annotation tools such as MOT16 Annotator require manual interaction per object per frame and produce no oriented bounding box (OBB) support. For a physics-based synthetic dataset rendered from Blender, none of these limitations apply: the simulator already holds exact 3D particle positions, and a Blender Python script can produce pixel-perfect annotations automatically.

## Comparison with Existing Approaches

| Criterion | MOT16 Annotator (ref. tool) | Our Blender approach |
|---|---|---|
| Annotation method | Manual, per object per frame | Fully automatic |
| Bounding box type | Axis-aligned (AABB) | Oriented (OBB) |
| Cross-camera ID consistency | Manual re-annotation per camera | Guaranteed by 3D centroid matching |
| 3D world coordinates | Not available | Exported per annotation |
| Scales with frame count | No | Yes |
| Human effort | Very high | Near zero |

## OBB-MOT Format

A new annotation format was defined to support oriented bounding boxes within the MOT evaluation framework. Two files are produced per camera sequence.

| File | Description |
|---|---|
| `gt_obb.txt` | Extended OBB-MOT format with 16 fields per line |
| `gt.txt` | Standard MOT16 AABB format for legacy tool compatibility |
| `seqinfo.ini` | MOTChallenge sequence descriptor |
| `FORMAT.md` | Full format specification |

**OBB-MOT field layout (`gt_obb.txt`):**

| Field | Value |
|---|---|
| frame | 1-based sequence frame index |
| id | Globally unique particle ID, consistent across all frames and cameras |
| x1, y1 ... x4, y4 | Four OBB corner coordinates in pixels |
| conf | 1 (ground truth) |
| class\_id | 1-based class label (1=pink, 2=blue, 3=yellow, 4=grey, 5=green, 6=red, 7=teal) |
| visibility | 1.0 (fully visible) |
| cx\_w, cy\_w, cz\_w | 3D world centroid of the particle in Blender units |

## ID Assignment via Hungarian Matching

Consistent particle IDs are assigned by matching 3D world centroids across consecutive frames using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). For each frame transition, a cost matrix of Euclidean distances between all current and previous centroids is computed and solved for a minimum-cost assignment. Particles within the distance threshold (0.30 Blender units) retain their ID; particles beyond it receive a new unique ID. Because all cameras process the same STL geometry for a given frame, the same matching result applies to all cameras simultaneously with no additional computation.

## Class Correction Pipeline

Initial class labels were assigned by material index (arbitrary STL separation order). These were replaced by a colour-classification pipeline that samples the rendered pixel colour at each annotation centroid, classifies by median HSV, and applies a per-track majority vote across all frames. Three co-located tracks (1, 6, 11) that were always within the isolation threshold were confirmed visually by the operator. A 7th class, **teal\_polybag**, was identified and added during this process.

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{fig_mot_mcmot_early.png}
\hfill
\includegraphics[width=0.48\textwidth]{fig_mot_mcmot_late.png}
\caption{\textbf{Train split.} Four-camera OBB-MOT overlays (2$\times$2 grid). Each colour is a unique track ID consistent across all cameras. \textbf{Left:} frame 150 --- all 11 tracks visible, particles clustered. \textbf{Right:} frame 575 --- particles spread, each camera observes a different subset.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{fig_mot_val_early.png}
\hfill
\includegraphics[width=0.48\textwidth]{fig_mot_val_late.png}
\caption{\textbf{Val split (frames 1,000-1,250).} \textbf{Left:} frame 1,050. \textbf{Right:} frame 1,225. Track IDs remain consistent with the train split; particle configuration is visibly different.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{fig_mot_test_early.png}
\hfill
\includegraphics[width=0.48\textwidth]{fig_mot_test_late.png}
\caption{\textbf{Test split (frames 1,500-1,750).} \textbf{Left:} frame 1,550. \textbf{Right:} frame 1,725. Particles are fully dispersed; the scene state is entirely unseen during training.}
\end{figure}

## Results

### Per-Split Summary

| Split | Frames | Frame range | MOT rows/camera | Total MOT rows | Track IDs | Classes |
|-------|--------|-------------|-----------------|----------------|-----------|---------|
| Train | 500 | 100-599 | 5,500 | 22,000 | 11 | 7 |
| Val | 251 | 1,000-1,250 | 2,146 | 8,584 | 11 | 7 |
| Test | 251 | 1,500-1,750 | 1,756 | 7,024 | 11 | 7 |
| **Total** | **1,002** | | | **37,608** | | |

All frames in each split are present in every camera with no gaps. Annotation accuracy is pixel-perfect with mathematically exact OBBs. Track IDs are globally consistent across all cameras and all splits via Hungarian centroid matching. Manual effort was limited to visual class confirmation for 3 co-located tracks in the train split; val and test class labels were assigned automatically using the confirmed per-track majority vote.

## Conclusion

The Blender-based OBB-MOT annotation pipeline produces globally consistent, pixel-perfect oriented bounding box ground truth for all cameras and all frames simultaneously. Three non-overlapping splits (train 100-599, val 1,000-1,250, test 1,500-1,750) provide 1,002 frames per camera and 37,608 total MOT annotations, making the dataset directly suitable for MCMOT benchmarking with standard metrics (HOTA, MOTA, IDF1) without data leakage.
