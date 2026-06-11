---
geometry: margin=1.8cm
fontsize: 10pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{caption}
  - \usepackage{graphicx}
  - \usepackage{xcolor}
  - \usepackage{titlesec}
  - \usepackage{parskip}
  - \captionsetup{font=small}
  - \setlength{\parskip}{3pt}
  - \setlength{\parindent}{0pt}
  - \titlespacing*{\section}{0pt}{8pt}{3pt}
  - \titlespacing*{\subsection}{0pt}{5pt}{2pt}
  - \definecolor{warn}{RGB}{170,0,0}
  - \definecolor{ok}{RGB}{0,120,40}
---

\begin{center}
{\LARGE\bfseries Polybag Multi-Camera Tracking Dataset}\\[0.4em]
{\large Generation, Annotation, and Evaluation Splits}\\[0.2em]
{\small Synthetic DEM Simulation \textbullet{} Blender OBB Annotation \textbullet{} OBB-MOT Ground Truth}\\[0.1em]
{\small May 2026}
\end{center}
\vspace{-0.8em}
\noindent\rule{\textwidth}{0.4pt}
\vspace{0.2em}

## 1. Background

A physics-based Discrete Element Method (DEM) simulation of polybag transport was rendered in Blender from four cameras (Front, Back, Left, Right), producing per-frame STL geometry files alongside rendered images. The goal is a fully annotated multi-camera dataset supporting both oriented bounding box (OBB) object detection training and Multi-Camera Multi-Object Tracking (MCMOT) evaluation across three non-overlapping splits: train, val, and test.

## 2. Rendering Gap Problem and Resolution

### 2.1 Original Problem

The initial render was run in two batches (frames 100-768 and 769-2000) and was interrupted mid-run each time. Each camera accumulated different gaps, producing a dataset in which the four cameras did not share a common set of frames, a fundamental obstacle for MCMOT evaluation.

| Camera | Rendered | Gaps | Coverage | Longest contiguous run |
|--------|----------|------|----------|------------------------|
| Front  | 616 | 23 | 33.2\% | 232 frames (769-1000) |
| Back   | 428 | 23 | 25.6\% | 244 frames (100-343) |
| Left   | 464 | 31 | 24.8\% | 217 frames (769-985) |
| Right  | 333 | 17 | 20.7\% | 179 frames (100-278) |

\textcolor{warn}{\textbf{Only 35 frames existed simultaneously across all four cameras}}, split across five disconnected segments with a longest run of 18 consecutive frames. Without shared frames, cross-camera re-identification collapses to four independent single-camera problems, and Kalman-filter-based trackers (SORT, ByteTrack, DeepSORT) fragment every track at each gap, making HOTA/MOTA/IDF1 scores statistically meaningless.

### 2.2 Resolution

Fresh renders of three consecutive sequences were produced using four parallel Blender instances per sequence, with no interruptions. A skip-logic mechanism in the render script ensures that already-rendered frames are not re-processed if a render is resumed after interruption.

\textcolor{ok}{\textbf{All frames in every split are present in every camera with no gaps.}}

## 3. YOLO OBB Annotation Pipeline

### 3.1 Approach Comparison

| Criterion | Manual Labelling | HSV Auto-labeller | Blender Projection (used) |
|-----------|-----------------|-------------------|--------------------------|
| Accuracy | High (if careful) | Approximate | Pixel-perfect |
| Lighting sensitivity | None | High | None |
| Occlusion handling | Manual | Fails | Geometry-based |
| Scales with frames | No | Yes | Yes |
| Manual effort | High | Tuning required | Zero |

### 3.2 Implementation

**`blender_annotate.py`** runs headlessly inside Blender for each frame. It imports the per-frame STL, separates loose parts, projects all mesh vertices through each camera matrix via `world_to_camera_view()`, fits `cv2.minAreaRect()` to the projected convex hull, and writes normalised 8-coordinate YOLO OBB labels. **`relabel_synth.py`** corrects class IDs by sampling a 13$\times$13 pixel patch at each OBB centroid in the rendered image, classifying by median HSV, and applying a per-track majority vote.

### 3.3 Class Scheme

| ID | Class | Colour |
|----|-------|--------|
| 0 | pink\_polybag | Magenta / warm pink |
| 1 | blue\_polybag | Periwinkle / mid-blue |
| 2 | yellow\_polybag | Yellow / orange-yellow |
| 3 | grey\_polybag | Neutral grey / white |
| 4 | green\_polybag | Lime / yellow-green |
| 5 | red\_polybag | Red / orange-red |
| 6 | teal\_polybag | Teal / cyan-green |

Train split class distribution: pink 3,968 \textbullet{} blue 2,023 \textbullet{} yellow 6,447 \textbullet{} grey 1,911 \textbullet{} green 1,686 \textbullet{} red 2,000 \textbullet{} teal 3,965.

## 4. OBB-MOT Annotation Pipeline

### 4.1 Approach Comparison

| Criterion | MOT16 Annotator | Blender approach (used) |
|-----------|----------------|------------------------|
| Annotation method | Manual, per object per frame | Fully automatic |
| Bounding box type | Axis-aligned (AABB) | Oriented (OBB) |
| Cross-camera ID consistency | Manual re-annotation per camera | Guaranteed by 3D centroid matching |
| 3D world coordinates | Not available | Exported per annotation |
| Scales with frame count | No | Yes |

### 4.2 OBB-MOT Format

Each camera sequence produces `gt_obb.txt` (16-field extended format) and `gt.txt` (standard MOT16 AABB for legacy tools). The `gt_obb.txt` fields are: `frame, id, x1, y1, x2, y2, x3, y3, x4, y4, conf, class_id, visibility, cx_w, cy_w, cz_w`.

### 4.3 ID Assignment

Consistent particle IDs are assigned by matching 3D world centroids across consecutive frames using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). A cost matrix of Euclidean distances is solved for minimum-cost assignment; particles within 0.30 Blender units retain their ID, others receive a new unique ID. Because all cameras share the same STL geometry per frame, the matching applies to all cameras simultaneously.

### 4.4 Class Correction

Initial labels from material index were replaced by a colour pipeline: HSV sampling at each centroid, per-track majority vote, and visual confirmation for 3 co-located tracks (IDs 1, 6, 11) that fell within the isolation threshold. This process also identified the 7th class, **teal\_polybag**.

## 5. Dataset Splits

To prevent data leakage the three splits use non-overlapping frame ranges with deliberate gaps between them, ensuring no temporal continuity across split boundaries.

| Split | Frame range | Frames | Images | YOLO ann. | MOT rows/cam | MOT rows total |
|-------|-------------|--------|--------|-----------|--------------|----------------|
| Train | 100-599 | 500 | 2,000 | 22,000 | 5,500 | 22,000 |
| Val | 1,000-1,250 | 251 | 1,004 | 7,584 | 2,146 | 8,584 |
| Test | 1,500-1,750 | 251 | 1,004 | 6,024 | 1,756 | 7,024 |
| **Total** | | **1,002** | **4,008** | **35,608** | | **37,608** |

Gaps (frames 600-999 and 1,251-1,499) eliminate any risk of appearance-based leakage. All splits share the same 11 particle IDs and 7 classes, with globally consistent track IDs across cameras and splits.

## 6. Visual Overlays

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{fig_yolo_mcmot_early.png}
\hfill
\includegraphics[width=0.48\textwidth]{fig_yolo_mcmot_late.png}
\caption{\textbf{YOLO OBB, train split.} Frame 150 (left) and frame 575 (right). Colour-coded by class.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{fig_yolo_val_early.png}
\hfill
\includegraphics[width=0.48\textwidth]{fig_yolo_val_late.png}
\caption{\textbf{YOLO OBB, val split.} Frame 1,050 (left) and frame 1,225 (right). Particle layout differs substantially from train.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{fig_yolo_test_early.png}
\hfill
\includegraphics[width=0.48\textwidth]{fig_yolo_test_late.png}
\caption{\textbf{YOLO OBB, test split.} Frame 1,550 (left) and frame 1,725 (right). Fully dispersed state unseen during training.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{fig_mot_mcmot_early.png}
\hfill
\includegraphics[width=0.48\textwidth]{fig_mot_mcmot_late.png}
\caption{\textbf{OBB-MOT, train split.} Frame 150 (left) and frame 575 (right). Colour-coded by track ID, consistent across all cameras.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{fig_mot_val_early.png}
\hfill
\includegraphics[width=0.48\textwidth]{fig_mot_val_late.png}
\caption{\textbf{OBB-MOT, val split.} Frame 1,050 (left) and frame 1,225 (right). Track IDs remain globally consistent with train.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{fig_mot_test_early.png}
\hfill
\includegraphics[width=0.48\textwidth]{fig_mot_test_late.png}
\caption{\textbf{OBB-MOT, test split.} Frame 1,550 (left) and frame 1,725 (right). Particles fully dispersed; scene state entirely unseen during training.}
\end{figure}

## 7. Conclusion

The Blender-based pipeline produces pixel-perfect OBB annotations and globally consistent OBB-MOT ground truth for all cameras and all frames with near-zero manual effort. Three non-overlapping splits (train 100-599, val 1,000-1,250, test 1,500-1,750) provide 4,008 images, 35,608 YOLO OBB annotations, and 37,608 MOT annotation rows, making the dataset directly suitable for detector training, hyperparameter tuning, and MCMOT benchmarking with HOTA, MOTA, and IDF1 metrics without data leakage.
