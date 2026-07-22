# Real Polybags

Real-camera-footage pipeline for polybag detection/tracking on a conveyor belt. Split out from the `synthetic_polybags` repo (which covers the Blender-synthetic track only). This directory is plain working files, not (yet) a git repo.

**Status:** the original 1,157-frame dataset (`full_dataset/`) has been removed — its source images were already gone (broken symlinks) and it's being superseded by a new dataset from the supervisor. The pipeline code below is reusable once that dataset lands; the design work for it is in progress.

## Structure

```
.
├── annotation/          # real_autolabel.py, autolabel_full.py, polybag_pipeline.py — watershed auto-labelling
├── dataset/             # generate_overlays.py — debug overlays for QC
├── tools/               # review_tool.py (Flask overlay viewer), flatten_overlays.sh
├── training/            # pseudo_label_train.py — iterative pseudo-label YOLO training
├── docs/                # autolabel/watershed reports and pipeline diagrams (from the old dataset)
├── experiments/         # raw camera footage from the 2026-05-28 conveyor experiments
│                        # (Bulk_2_moving/, Bulk_rest/, Single_Polybags_moving/, etc.)
├── real_data_labels/    # review-tool working state from the old dataset (labels/, overlays/, review/) — legacy reference
├── review_state.json    # review_tool.py calibration/state (tied to the old dataset)
└── yolo11n-obb.pt       # base pretrained weights, reusable for the new dataset
```

## Pipeline (reusable once new data lands)

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
