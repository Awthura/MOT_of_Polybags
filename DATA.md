# Data and model assets (Google Drive)

GitHub does not host the large binaries this project needs to run. The repository
carries all code, configs, the committed calibration results, and the printed
calibration boards. Everything large (rendered datasets, video recordings, Blender
and mesh assets) lives on Google Drive.

> **Google Drive links** (each shared as "anyone with the link — Viewer"):
>
> - **Tier 1 — demo videos** → `real_polybags/algorithm/videos/`:
>   https://drive.google.com/drive/folders/14RQh4hhNrukqPCSlQvUqwzPbnDX8AUqD
> - **Tier 1 — weights + dashboard map** (the six files in §2):
>   https://drive.google.com/drive/folders/1V4Be_Td6aJQI9diglv0_WuFRE8KSM9kv
> - **Tier 2** (synthetic datasets) and **Tier 3** (calibration/experiment
>   recordings): _add links here once uploaded._

Download only what the task you care about needs (see [§4](#4-minimum-set-per-task)).
You do **not** need the ~13 GB of synthetic data just to run the digital twin.

---

## 1. Large assets on Drive

| Repo path (where it goes) | Size | What it is | Needed for |
|---|---|---|---|
| `synthetic_polybags/synth_dataset_mcmot/` | 6.7 GB | 500-frame 4-camera MCMOT set + OBB-MOT GT | synthetic tracking + HOTA/MOTA |
| `synthetic_polybags/synth_dataset_test/` | 3.3 GB | test split + GT | synthetic evaluation |
| `synthetic_polybags/synth_dataset_val/` | 3.3 GB | val split + GT | synthetic evaluation |
| `synthetic_polybags/convert_stl_to_animation_multi_camera.blend` + STL geometry | small + external | Blender sim scene and DEM mesh sequence | re-rendering the synthetic set |
| `real_polybags/algorithm/videos/` | 0.6 GB | curated demo clips (`static/ single/ bulk/ validation/`) | the digital twin |
| `real_polybags/calibration/calibration_videos/` | 1.8 GB | board + belt recordings | redoing calibration |
| `real_polybags/experiments/` | 1.4 GB | raw 2026-05-28 conveyor footage | re-labelling from source |
| `real_polybags/real_data/` | 0.3 GB | recording sessions + sync utilities data | recording/sync work |

## 2. Small essentials (weights + display map)

These are small (each < 20 MB) but are gitignored, and inference needs them.
**Download:** https://drive.google.com/drive/folders/1V4Be_Td6aJQI9diglv0_WuFRE8KSM9kv

| Repo path | Size | Purpose |
|---|---|---|
| `synthetic_polybags/training/weights_synth_hires.pt` | 6 MB | synthetic detector (imgsz 1920, mAP50 0.995) |
| `synthetic_polybags/training/weights_synth_640.pt` | 6 MB | synthetic detector (imgsz 640) |
| `real_polybags/training/yolo_polybag_v1/best.pt` | 18 MB | deployed real detector (twin default) |
| `real_polybags/training/yolo11seg_finetune/weights/best.pt` | 6 MB | segmentation variant (mask overlays) |
| `real_polybags/yolo11n-obb.pt` | 5 MB | base pretrained weights |
| `real_polybags/calibration/datasets/dashboard/conveyor_dashboard.png` | 2 MB | the belt map the dashboard draws on |

The per-camera calibration results (`real_polybags/calibration/results/*.json`) are
**already committed**, so the twin's geometry needs nothing extra.

---

## 3. Proposed Drive folder structure

Mirror the repo paths so placement is a straight copy:

```
MOT_of_Polybags_data/
├── synthetic_polybags/
│   ├── synth_dataset_mcmot.zip
│   ├── synth_dataset_test.zip
│   ├── synth_dataset_val.zip
│   └── blender/                # .blend + STL geometry
├── real_polybags/
│   ├── algorithm_videos.zip    # -> real_polybags/algorithm/videos/
│   ├── calibration_videos.zip
│   ├── experiments.zip
│   └── real_data.zip
└── weights/                    # the §2 essentials, if not committed
```

## 4. Minimum set per task

- **Run the digital twin:** `algorithm_videos.zip` + the weights (`best.pt`,
  `conveyor_dashboard.png`). ~0.6 GB. Calibration results are already in git.
- **Reproduce synthetic tracking / HOTA:** `synth_dataset_mcmot.zip` +
  `weights_synth_hires.pt`. ~6.7 GB.
- **Reproduce synthetic evaluation on splits:** add `synth_dataset_test.zip` /
  `synth_dataset_val.zip`.
- **Re-render the synthetic set:** the `blender/` folder (scene + STL) and Blender.
- **Redo calibration:** `calibration_videos.zip`.
- **Re-label real footage:** `experiments.zip`.

## 5. Placement

Unzip each archive so its contents land at the repo path in the §1 table. For
example:

```bash
# from the repo root
unzip ~/Downloads/synth_dataset_mcmot.zip -d synthetic_polybags/
unzip ~/Downloads/algorithm_videos.zip   -d real_polybags/algorithm/     # -> videos/
```

Optional, scriptable download with [`gdown`](https://github.com/wkentaro/gdown)
(`pip install gdown`):

```bash
gdown --folder "<DRIVE_FOLDER_URL>" -O ./_drive
# then unzip from ./_drive into the paths above
```

After placing the files, verify the layout:

```bash
ls synthetic_polybags/synth_dataset_mcmot/cam_01_front/images | head
ls real_polybags/algorithm/videos/single
ls real_polybags/training/yolo_polybag_v1/best.pt
```
