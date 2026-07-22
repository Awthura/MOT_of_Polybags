# Real Polybags

Real-camera-footage pipeline for polybag detection/tracking on a conveyor belt. Split out from the `synthetic_polybags` repo (which covers the Blender-synthetic track only). This directory is plain working files, not (yet) a git repo.

## Structure

```
.
├── annotation/          # real_autolabel.py, autolabel_full.py, polybag_pipeline.py — watershed auto-labelling
├── dataset/             # generate_overlays.py — debug overlays for QC
├── tools/               # review_tool.py (Flask overlay viewer), flatten_overlays.sh
├── training/            # pseudo_label_train.py — iterative pseudo-label YOLO training
├── docs/                # autolabel/watershed reports and pipeline diagrams
├── experiments/         # raw camera footage from the 2026-05-28 conveyor experiments
│                        # (Bulk_2_moving/, Bulk_rest/, Single_Polybags_moving/, etc.)
├── full_dataset/        # 1,157-frame dataset: images/, labels_manual/, labels_auto/, labels/ (merged)
├── real_data_labels/    # review-tool working state (labels/, overlays/, review/)
├── pipeline_output/     # early autolabel pipeline dev output (EDA, experiments)
├── pseudo_label/        # pseudo-label training round data
├── runs/                # YOLO training run outputs
├── review_state.json    # review_tool.py calibration/state
└── yolo11n-obb.pt       # base pretrained weights
```

## Known issue

`full_dataset/images/` are symlinks to `YOLO11_OBB_training_data/`, which no longer exists on disk — all 1,157 links are currently broken. Carried over as-is from before the reorg; needs a source-data recovery pass before `full_dataset` can be used for anything that touches the actual images (labels/annotations themselves are intact files, not symlinks).

## Dataset (`full_dataset/`)
- **1,157 frames** (1920×1080), dark background
- **6 classes**: pink, blue, yellow, grey, green, red polybags
- **86 manually annotated** frames + **1,071 auto-labelled** via watershed segmentation
- Merged labels in `full_dataset/labels/` (manual takes precedence)

## Pipeline

```
annotation/autolabel_full.py   →  V>120 threshold → MORPH_OPEN → distance transform
                                   → watershed → HSV nearest-centroid classifier → OBB
dataset/generate_overlays.py   →  debug overlays for visual QC
tools/review_tool.py           →  Flask viewer to browse/QC auto-labelled frames
```

Iterative pseudo-label training with YOLO11n-OBB:
```bash
python training/pseudo_label_train.py --setup 0
yolo obb train data=pseudo_label/round_0/data.yaml \
     model=yolo11n-obb.pt epochs=100 imgsz=1920 batch=4 \
     name=round_0 project=pseudo_label/runs
```
