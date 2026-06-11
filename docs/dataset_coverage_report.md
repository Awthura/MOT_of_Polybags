---
geometry: margin=1.5cm
fontsize: 10pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{caption}
  - \usepackage{xcolor}
  - \usepackage{parskip}
  - \usepackage{titlesec}
  - \captionsetup{font=small}
  - \setlength{\parskip}{2pt}
  - \setlength{\parindent}{0pt}
  - \titlespacing*{\section}{0pt}{6pt}{2pt}
  - \definecolor{warn}{RGB}{170,0,0}
  - \definecolor{ok}{RGB}{0,120,40}
---

\begin{center}
{\LARGE\bfseries Multi-Camera Frame Coverage: Problem and Resolution}\\[0.25em]
{\large Rendering Gap Analysis for MCMOT Dataset Generation}\\[0.1em]
{\small May 2026}
\end{center}
\vspace{-0.7em}
\noindent\rule{\textwidth}{0.4pt}

## Original Problem

The initial render was run in two batches (frames 100-768 and 769-2000) and was interrupted mid-run each time. Each camera accumulated different gaps, producing a dataset in which the four cameras did not share a common set of frames, a fundamental obstacle for MCMOT evaluation, which requires every camera to observe every frame simultaneously.

## Original Per-Camera Coverage (Broken Dataset)

| Camera | Rendered | Range | Gaps | Missing | Coverage | Longest contiguous run |
|----|-----|----|---|-----|-----|------------|
| Front  | 616 | 100-1954 | 23 | 1,239 | 33.2\% | 232 frames (769-1000) |
| Back   | 428 | 100-1772 | 23 | 1,245 | 25.6\% | 244 frames (100-343) |
| Left   | 464 | 103-1972 | 31 | 1,406 | 24.8\% | 217 frames (769-985) |
| Right  | 333 | 100-1708 | 17 | 1,276 | 20.7\% | 179 frames (100-278) |
| **Total** | **1,841** | | **94** | **5,166** | **26.5\%** | |

\textcolor{warn}{\textbf{Only 35 frames existed simultaneously across all four cameras}}, split across five disconnected segments with a longest run of only 18 consecutive frames (185-202). Pairwise overlaps ranged from 65 (Left+Right) to 269 (Front+Left) frames.

## Why This Broke MCMOT Evaluation

MCMOT trackers (SORT, ByteTrack, DeepSORT) rely on a Kalman filter that fragments every track at each gap, artificially inflating ID-switch and fragmentation metrics. Cross-camera re-identification requires all cameras to observe the same time step; without shared frames it collapses to four independent single-camera problems. Standard benchmarks require 150-200 consecutive frames for stable HOTA/MOTA/IDF1 scores. An 18-frame sequence yields statistically meaningless results.

## Resolution

A fresh render of 500 consecutive frames (100-599) was produced for all four cameras simultaneously, with no interruptions. Four parallel Blender instances completed the render in approximately 20 minutes on the available machine. All downstream annotation steps ran automatically.

## Result: Clean MCMOT Dataset

\textcolor{ok}{\textbf{All 500 frames are present in every camera with no gaps.}}

| Metric | Value |
|----|----|
| Frames | 100-599 (500 consecutive) |
| Cameras | 4 (Front, Back, Left, Right) |
| Frames common to all 4 cameras | **500 (100\%)** |
| YOLO OBB annotations | 22,000 (5,500 per camera) |
| OBB-MOT annotations | 22,000 (5,500 per camera) |
| Unique track IDs | 11 |
| Object classes | 7 |
| Sequence duration at 25 fps | 20 seconds |
| MCMOT evaluation feasible | **Yes** |

## Dataset Splits

To prevent data leakage the three splits use non-overlapping frame ranges separated by gaps, so the model never trains on frames it is later evaluated on.

| Split | Frame range | Frames | Cameras | Images | YOLO annotations | MOT rows (per cam) |
|---|---|---|---|---|---|---|
| Train | 100-599 | 500 | 4 | 2,000 | 22,000 | 5,500 |
| Val | 1,000-1,250 | 251 | 4 | 1,004 | 7,584 | 2,146 |
| Test | 1,500-1,750 | 251 | 4 | 1,004 | 6,024 | 1,756 |
| **Total** | | **1,002** | | **4,008** | **35,608** | |

Gaps between splits (601-999 and 1,251-1,499) are intentional: they ensure no temporal continuity between train, val, and test sequences, eliminating any risk of appearance-based leakage across split boundaries.
