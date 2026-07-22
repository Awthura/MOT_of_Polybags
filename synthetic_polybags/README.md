# Synthetic Polybags

Synthetic track of the MOT of Polybags project (see [top-level README](../README.md)): a physics-based DEM simulation rendered in Blender from 4 cameras, with fully automatic YOLO OBB + OBB-MOT ground truth generation, tracking, and MCMOT benchmarking. The real-camera-footage pipeline lives in the sibling directory [`../real_polybags/`](../real_polybags/README.md).

---

## Repository Structure

```
.
├── annotation/          # Label generation scripts (Blender OBB + MOT ground truth)
├── dataset/             # Dataset assembly and preprocessing
├── training/            # YOLO training configs/slurm scripts
├── render/              # Blender render scripts (bash)
├── tracking/            # MCMOT tracking, association, benchmarking
├── tracking_results/    # Tracker output + benchmark JSONs (gitignored)
├── visualization/       # Overlay and video generation
├── tools/               # Diagnostic and one-off utilities
└── docs/                # Reports and pipeline diagrams (.md, .drawio)
```

Large/generated assets live alongside the code but are gitignored: `synth_dataset_mcmot/`, `synth_dataset_test/`, `synth_dataset_val/`, `convert_stl_to_animation_multi_camera*.blend*`, `logs/`, `render_logs_*/`.

---

## Dataset (`synth_dataset*/`)
- **1,841 frames** rendered from 4 cameras (Front, Back, Left, Right) at 1920×1080
- **7 classes**: pink, blue, yellow, grey, green, red, teal polybags
- **YOLO OBB labels**: `synth_dataset/labels/`
- **OBB-MOT ground truth**: `synth_dataset/mot_obb/cam_0{1-4}_*/gt/gt_obb.txt` — 66,341 rows total
- Track→class mapping: `synth_dataset/track_classes.csv`
- Separate `synth_dataset_mcmot/`, `synth_dataset_val/`, `synth_dataset_test/` splits for MCMOT evaluation

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

#### OBB-MOT format (`gt_obb.txt`) — 16 comma-separated fields:
```
frame, id, x1, y1, x2, y2, x3, y3, x4, y4, conf, class_id, visibility, cx_w, cy_w, cz_w
```

---

## Training

Trained directly on `synth_dataset` via SLURM on the OVGU cluster (`training/train_synth.slurm`, `training/train_synth_hires.slurm`) or the Colab notebook (`training/train_colab.ipynb`). Weights: `training/weights_synth_640.pt` (imgsz=640), `training/weights_synth_hires.pt` (imgsz=1920, recommended — mAP50=0.995, mAP50-95=0.989).

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
