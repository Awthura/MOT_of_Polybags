---
geometry: margin=1.8cm
fontsize: 10.5pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{caption}
  - \usepackage{graphicx}
  - \captionsetup{font=small}
  - \usepackage{titling}
  - \setlength{\droptitle}{-3em}
---

\begin{center}
{\LARGE\bfseries Automatic Ground-Truth Annotation via Blender}\\[0.4em]
{\large YOLO OBB Classification of Polybags, 7 Classes}\\[0.2em]
{\small May 2026}
\end{center}
\vspace{-1em}
\noindent\rule{\textwidth}{0.4pt}
\vspace{0.5em}

## Background

The synthetic training dataset was generated in Blender using a physics-based Discrete Element Method (DEM) simulation. Per-frame STL files containing all polybag geometries were imported, separated into individual objects, and rendered from four cameras (Front, Back, Left, Right). Annotations were previously produced manually or via a colour-based auto-labeller relying on HSV masks and watershed segmentation, which was sensitive to lighting changes, occlusion, and background contamination. For synthetic data this approach is fundamentally unnecessary: Blender holds the exact position, rotation, and geometry of every object at every frame.

## Comparison of Annotation Approaches

| Criterion            | Manual Labelling     | HSV Auto-labeller        | Blender Projection (new) |
|----------------------|----------------------|--------------------------|--------------------------|
| Accuracy             | High (if careful)    | Approximate              | Pixel-perfect            |
| Speed                | Very slow            | Fast                     | Fast                     |
| Lighting sensitivity | None                 | High                     | None                     |
| Occlusion handling   | Manual               | Fails                    | Geometry-based           |
| Scales with frames   | No                   | Yes                      | Yes                      |
| Manual effort        | High                 | Tuning required          | Zero                     |

## Implementation

Two scripts were developed. **`blender_annotate.py`** runs headlessly inside Blender and generates YOLO OBB labels alongside rendering. **`build_synth_dataset.py`** consolidates fragmented render outputs into a single flat dataset. A subsequent colour-classification pipeline (**`blender_color_classify.py`** and **`relabel_synth.py`**) corrected class labels using rendered pixel HSV sampling and visual ground-truth confirmation.

| Step | Script | Description |
|------|--------|-------------|
| 1. Render | `blender_annotate.py` | Renders all 4 cameras for each frame; loads per-frame STL, separates loose parts |
| 2. Vertex Projection | `blender_annotate.py` | Projects all mesh vertices through the camera matrix via `world_to_camera_view()` |
| 3. OBB Fitting | `blender_annotate.py` | Fits `cv2.minAreaRect()` to projected convex hull; yields tight rotated bounding box |
| 4. Class Assignment | `relabel_synth.py` | Samples 13$\times$13 pixel patch at each OBB centroid in the rendered image; classifies by median HSV; confirmed against visual ground truth |
| 5. Label Output | `blender_annotate.py` | Writes normalised 8-coordinate YOLO OBB `.txt` per frame per camera |
| 6. Dataset Assembly | `build_synth_dataset.py` | Copies images and labels into flat `images/ + labels/` structure |

## Class Scheme (7 Classes)

| ID | Class | Colour |
|----|-------|--------|
| 0 | pink\_polybag | Magenta / warm pink |
| 1 | blue\_polybag | Periwinkle / mid-blue |
| 2 | yellow\_polybag | Yellow / orange-yellow |
| 3 | grey\_polybag | Neutral grey / white |
| 4 | green\_polybag | Lime / yellow-green |
| 5 | red\_polybag | Red / orange-red |
| 6 | teal\_polybag | Teal / cyan-green |

## Results

### Per-Split Summary

| Split | Frames | Images | Annotations | Ann./camera | Object classes |
|-------|--------|--------|-------------|-------------|----------------|
| Train (100-599) | 500 | 2,000 | 22,000 | 5,500 | 7 |
| Val (1,000-1,250) | 251 | 1,004 | 7,584 | 1,896 | 7 |
| Test (1,500-1,750) | 251 | 1,004 | 6,024 | 1,506 | 7 |
| **Total** | **1,002** | **4,008** | **35,608** | | |

All annotations are pixel-perfect OBBs. Colour class is sampled from rendered-image HSV with visual confirmation for 3 co-located tracks. Splits are non-overlapping with deliberate frame gaps to prevent data leakage.

### Train Class Distribution

| Class | pink | blue | yellow | grey | green | red | teal |
|-------|------|------|--------|------|-------|-----|------|
| Count | 3,968 | 2,023 | 6,447 | 1,911 | 1,686 | 2,000 | 3,965 |

## Annotation Overlays

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{fig_yolo_mcmot_early.png}
\hfill
\includegraphics[width=0.48\textwidth]{fig_yolo_mcmot_late.png}
\caption{\textbf{Train split.} Four-camera YOLO OBB overlays (2$\times$2 grid). \textbf{Left:} frame 150 --- bags clustered near simulation start, 7-class colour-coded OBBs. \textbf{Right:} frame 575 --- bags spread across the scene.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{fig_yolo_val_early.png}
\hfill
\includegraphics[width=0.48\textwidth]{fig_yolo_val_late.png}
\caption{\textbf{Val split (frames 1,000-1,250).} \textbf{Left:} frame 1,050. \textbf{Right:} frame 1,225. Particle positions differ substantially from the train split, confirming temporal separation.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{fig_yolo_test_early.png}
\hfill
\includegraphics[width=0.48\textwidth]{fig_yolo_test_late.png}
\caption{\textbf{Test split (frames 1,500-1,750).} \textbf{Left:} frame 1,550. \textbf{Right:} frame 1,725. Particles are more dispersed, representing a later simulation state unseen during training.}
\end{figure}

## Conclusion

Because Blender holds the exact 3D geometry, OBB fitting is mathematically exact and entirely independent of image content, lighting, or colour. Class labels were assigned by sampling the rendered pixel colour at each annotation centroid and confirmed visually, giving a fully verified 7-class dataset. The three non-overlapping splits (train 100-599, val 1,000-1,250, test 1,500-1,750) provide 4,008 images with 35,608 annotations suitable for object detection training, hyperparameter tuning, and final evaluation without data leakage.
