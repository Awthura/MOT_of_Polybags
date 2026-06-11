# MOT of Polybags

Multi-Object Tracking (MOT) of polybags on a conveyor belt using synthetic Blender-rendered data and real camera footage. Developed at the **Chair of Automation / Manufacturing Systems (AMS), OVGU Magdeburg**.

## Overview

The project has two parallel tracks:

1. **Synthetic pipeline** — Physics-based DEM simulation rendered in Blender from 4 cameras, with fully automatic YOLO OBB + OBB-MOT ground truth generation.
2. **Real data pipeline** — Auto-labelling of real conveyor footage using watershed segmentation, with iterative pseudo-label YOLO training.

---

## Repository Structure

```
.
├── annotation/          # Label generation scripts
├── dataset/             # Dataset assembly and preprocessing
├── training/            # YOLO pseudo-label training
├── render/              # Blender render scripts (bash)
├── visualization/       # Overlay and video generation
├── tools/               # Diagnostic and one-off utilities
├── experiments/
│   └── real_data/       # Scripts for real conveyor footage
└── docs/                # Reports and pipeline diagrams (.md, .drawio)
```

---

## Datasets

### Synthetic Dataset (`synth_dataset/`)
- **1,841 frames** rendered from 4 cameras (Front, Back, Left, Right) at 1920×1080
- **7 classes**: pink, blue, yellow, grey, green, red, teal polybags
- **YOLO OBB labels**: `synth_dataset/labels/`
- **OBB-MOT ground truth**: `synth_dataset/mot_obb/cam_0{1-4}_*/gt/gt_obb.txt` — 66,341 rows total
- Track→class mapping: `synth_dataset/track_classes.csv`

### Real Dataset (`full_dataset/`)
- **1,157 frames** (1920×1080), dark background
- **6 classes**: pink, blue, yellow, grey, green, red polybags
- **86 manually annotated** frames + **1,071 auto-labelled** via watershed
- Merged labels in `full_dataset/labels/` (manual takes precedence)

---

## Setup

```bash
pip install -r requirements.txt
```

For Blender-based scripts (`annotation/blender_annotate.py`, `annotation/blender_mot_annotate.py`), run inside Blender:
```bash
/path/to/Blender convert_stl_to_animation_multi_camera.blend \
    --background --python annotation/blender_annotate.py \
    -- --frames 100-599 --out_dir synth_dataset
```

---

## Annotation Pipeline

### Synthetic — YOLO OBB labels
```
annotation/blender_annotate.py   →  raw labels (wrong class IDs from material index)
annotation/relabel_synth.py      →  corrects class IDs via HSV centroid sampling
dataset/build_synth_dataset.py   →  flattens per-camera renders into synth_dataset/
dataset/generate_overlays.py     →  debug overlays for visual QC
```

### Synthetic — OBB-MOT ground truth
```
annotation/blender_mot_annotate.py  →  Hungarian-matched track IDs, OBB corners, 3D centroids
annotation/fix_mot_classes.py       →  maps material-index class_ids → visually confirmed classes
```

### Real data — Watershed auto-labelling
```
annotation/autolabel_full.py   →  V>120 threshold → MORPH_OPEN → distance transform
                                   → watershed → HSV nearest-centroid classifier → OBB
```

#### OBB-MOT format (`gt_obb.txt`) — 16 comma-separated fields:
```
frame, id, x1, y1, x2, y2, x3, y3, x4, y4, conf, class_id, visibility, cx_w, cy_w, cz_w
```

---

## Training

Iterative pseudo-label training with YOLO11n-OBB:

```bash
# Set up Round 0 dataset (86 GT images)
python training/pseudo_label_train.py --setup 0

# Train Round 0
yolo obb train data=pseudo_label/round_0/data.yaml \
     model=yolo11n-obb.pt epochs=100 imgsz=1920 batch=4 \
     name=round_0 project=pseudo_label/runs

# Set up Round 1 from Round 0 predictions (conf >= 0.35)
python training/pseudo_label_train.py --setup 1 \
     --weights pseudo_label/runs/round_0/weights/best.pt
```

---

## Rendering (Blender, MacBook)

| Script | Purpose | Est. time |
|--------|---------|-----------|
| `render/render_500_mcmot.sh` | Fresh 500-frame MCMOT dataset (frames 100–599) | ~67 min |
| `render/render_missing_frames.sh` | Fill gaps in existing synth_dataset | ~3–4 hrs |
| `render/render_250_test.sh` | Render test split | ~34 min |
| `render/render_250_val.sh` | Render val split | ~34 min |

All render scripts are resumable — already-rendered frame/camera pairs are skipped automatically.

---

## Class Scheme

| YOLO ID | Class | MOT class_id |
|---------|-------|--------------|
| 0 | pink_polybag | 1 |
| 1 | blue_polybag | 2 |
| 2 | yellow_polybag | 3 |
| 3 | grey_polybag | 4 |
| 4 | green_polybag | 5 |
| 5 | red_polybag | 6 |
| 6 | teal_polybag | 7 |

---

## Cluster Training (OVGU `ants` cluster)

SSH: `ssh <username>@ants.cs.ovgu.de`  
Partition: `gpu-stud` (NVIDIA A40, 46 GB VRAM)

See [cluster wiki](https://code.ovgu.de/fin-all/cluster/-/wikis/home) for SLURM job submission.
