# Multi-Object Tracking of Polybags on a Conveyor

Detection, tracking, and multi-camera association of deformable polybags on a
conveyor belt. Developed in the **Autonomous Multisensor Systems (AMS)** group,
Otto von Guericke University Magdeburg, supervised by Sai Preetham Sata.

The work runs along **two tracks** that share one pipeline design:

- **`synthetic_polybags/`** — a physics-simulated conveyor rendered in Blender
  from four cameras, with fully automatic ground truth (YOLO OBB + OBB-MOT). Used
  to build and benchmark every component in a setting where the labels are exact
  and every experiment repeats. Detection reaches mAP50 0.995, and single-camera
  tracking reaches up to MOTA 97.9 / IDF1 92.6 / HOTA 86.9 (BoT-SORT).
- **`real_polybags/`** — the same pipeline carried onto the physical five-camera
  rig: a board-free calibration to a shared metric belt plane, real footage
  auto-labelled with a fine-tuned vision-language model, a trained detector, and a
  **live digital twin** that projects every detected bag onto a 2-D belt map in
  real time over MQTT, fuses detections across cameras, and counts by occupancy
  flux.

The full write-up is in `report/` (LaTeX; kept local, not committed). This README
is the guide to running the code in this repository.

---

## 1. What is in git, what is on Google Drive

GitHub does not host the large binaries this project needs to run (rendered
datasets, video recordings, model weights, Blender/mesh assets). The repository
carries **all code, configs, the calibration result files, and the printed
calibration boards** — 208 tracked files, a few MB. Everything large lives on
Google Drive.

**Before running anything, read [`DATA.md`](DATA.md).** It lists every external
asset, its size, which task needs it, and exactly where to unpack it in the tree.

Quick sense of scale:

| Asset group | Size | Needed for |
|---|---|---|
| Synthetic datasets (`synth_dataset_{mcmot,test,val}/`) | ~13 GB | synthetic training / tracking / evaluation |
| Real demo videos (`real_polybags/algorithm/videos/`) | ~0.6 GB | the digital twin |
| Real raw recordings + calibration videos | ~3.5 GB | re-calibration, re-labelling |
| Model weights (`*.pt`) | <20 MB each | all inference (small; see DATA.md) |
| Blender + STL sim assets | small + external | re-rendering the synthetic set |

---

## 2. Prerequisites

- **Python 3.10+** (3.11 recommended).
- **Mosquitto** MQTT broker — only for the digital twin (`brew install mosquitto`
  / `apt install mosquitto`).
- **Blender 3.x+** — only to re-render or re-annotate the synthetic set. The
  Blender scripts import `bpy`/`mathutils` and must run *inside* Blender, not the
  system Python.
- **GPU** recommended for training (the project used an NVIDIA A40 on the OVGU
  cluster) but not required for inference.
- Optional, only if capturing from the physical cameras: the Basler `pypylon` and
  Intel `pyrealsense2` SDKs. Nothing in the offline pipeline needs them.

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the whole offline pipeline. The heavier, task-specific
dependencies are:

| Package | Needed for |
|---|---|
| `ultralytics`, `torch`, `torchvision` | detection + tracking (YOLO11) |
| `transformers`, `peft` | LocateAnything auto-labelling and its LoRA fine-tune |
| `motmetrics` | MOTA / MOTP / IDF1 evaluation |
| `flask` | the calibration web tool and the streamer |
| `paho-mqtt`, `pyyaml` | the digital-twin streamer |
| `opencv-python`, `numpy`, `scipy`, `Pillow`, `matplotlib` | everywhere |

> **HOTA** (with DetA / AssA) is computed with [TrackEval](https://github.com/JonathonLuiten/TrackEval),
> installed separately. On NumPy 2.0 both `motmetrics` and TrackEval need small
> shims for the removed aliases — see [§6 Known issues](#6-known-issues-and-gotchas).

---

## 3. Repository layout

```
.
├── README.md              # this file
├── DATA.md                # Google Drive manifest — read before running
├── requirements.txt
├── report/                # LaTeX final report (local, gitignored)
├── demo/                  # result-video documentation for the digital twin
│
├── synthetic_polybags/    # SYNTHETIC TRACK
│   ├── annotation/          # Blender OBB + OBB-MOT ground-truth generation
│   ├── dataset/             # dataset assembly from per-camera renders
│   ├── render/              # Blender render shell scripts (resumable)
│   ├── training/            # YOLO train configs + weights_synth_*.pt
│   ├── tracking/            # single-cam tracking, association, MOT evaluation
│   ├── visualization/       # overlay + video generation
│   ├── tools/               # diagnostics
│   └── docs/                # dataset + annotation reports
│
└── real_polybags/         # REAL TRACK
    ├── calibration/         # board-free metric-belt calibration web tool
    ├── annotation/          # legacy watershed auto-labelling (superseded)
    ├── training/
    │   ├── locate_anything/   # LocateAnything-3B LoRA fine-tune + tiled auto-label
    │   ├── yolo_polybag_v1/   # the deployed detector (YOLO11s)
    │   └── yolo11seg_finetune/# segmentation variant (mask overlays)
    ├── benchmark_real/      # evaluation vs the hand-annotated human ground truth
    ├── algorithm/           # the LIVE DIGITAL TWIN (streamer + dashboard + fusion)
    ├── real_data/           # recording + sync utilities
    └── docs/                # auto-label + dataset reports
```

Each subdirectory has its own README with deeper detail; this file is the map and
the quickstart. See [§7](#7-documentation-index) for the full document index.

---

## 4. Synthetic track — how to run

The datasets (`synth_dataset_mcmot/`, `synth_dataset_test/`, `synth_dataset_val/`)
and the weights come from Google Drive (see `DATA.md`). With those in place:

**Train the detector** (OVGU cluster, SLURM):
```bash
cd synthetic_polybags
sbatch training/train_synth_hires.slurm      # imgsz 1920 -> weights_synth_hires.pt
```
The trained weights (`training/weights_synth_hires.pt`, mAP50 0.995) are provided,
so training is optional for reproducing the tracking numbers.

**Run single-camera tracking** (ByteTrack + BoT-SORT over all cameras/splits):
```bash
python tracking/run_tracking.py --dataset both --tracker both \
    --model training/weights_synth_hires.pt --imgsz 1920 --conf 0.25
```

**Evaluate MOTA / MOTP / IDF1** (py-motmetrics):
```bash
python tracking/evaluate_mot.py --all               # every tracker/cam/split
python tracking/evaluate_mot.py --tracker bytetrack --dataset test --cam front
```

**Cross-camera association** (offline and real-time associators):
```bash
python tracking/associate_cameras.py                # see script header for options
```

**Re-render the synthetic set** (Blender, resumable — needs the STL sim geometry,
see DATA.md):
```bash
bash render/render_500_mcmot.sh                     # 500-frame MCMOT set, ~67 min
/path/to/blender convert_stl_to_animation_multi_camera.blend --background \
    --python annotation/blender_annotate.py -- --frames 100-599 --out_dir synth_dataset
python annotation/relabel_synth.py                  # fix class IDs via HSV centroid
python annotation/blender_mot_annotate.py           # Hungarian-matched OBB-MOT GT
```

Full details: [`synthetic_polybags/README.md`](synthetic_polybags/README.md).

---

## 5. Real track — how to run

### 5a. Calibration (build the shared metric belt frame)

A self-contained Flask web tool solves intrinsics + board-free extrinsics from
recordings and writes one result file per camera to `calibration/results/`.

```bash
cd real_polybags/calibration
python app.py            # then open the printed local URL in a browser
```
The solved results (`results/basler_1.json`, `basler_2.json`, `lucid.json`,
`rgbd_1_color.json`) are already committed, so the digital twin runs without
redoing calibration. To redo it, follow
[`calibration/PROCEDURE.md`](real_polybags/calibration/PROCEDURE.md).

### 5b. Auto-label real footage (LocateAnything-3B + LoRA)

Real bags are a single merged class, so labels come from a fine-tuned
vision-language localiser, cleaned with belt masks and tiled inference.
```bash
cd real_polybags/training/locate_anything
bash finetune_lora.sh                 # LoRA fine-tune on the supervisor's annotations
python autolabel_tiled.py             # tiled auto-label -> labels_final_v3
```

### 5c. Train the deployed detector (YOLO11s, polybag_v1)

```bash
cd real_polybags/training/yolo_polybag_v1
bash train.sh                         # -> best.pt (the twin's default detector)
```
`best.pt` is provided on Drive, so this is optional.

### 5d. The live digital twin

Producer (streamer) → **MQTT** → decoupled browser dashboard. Runs one scenario
at a time; the demo videos come from Drive into `algorithm/videos/{scenario}/`.

```bash
cd real_polybags/algorithm
pip install -r requirements.txt        # adds paho-mqtt, pyyaml on top of the base set

# 1. broker (1883 tcp for the streamer, 9001 websockets for the browser)
mosquitto -c mosquitto.conf

# 2. streamer — pick a scenario and detector
python streamer/app.py --scenario single --model detect
#   scenarios: static | single | bulk | validation
#   models:    detect (yolo_polybag_v1, higher recall) | seg (masks)

# 3. open the combined view
open http://127.0.0.1:5002/
```
Verify the geometry alone, no browser needed:
```bash
python smoke_test.py                   # each camera must reproduce its stored fit RMS
mosquitto_sub -t /polybags -v          # watch the bus
```
Details, message schema, and the fusion/counting stage:
[`algorithm/README.md`](real_polybags/algorithm/README.md).

### 5e. Evaluate against human ground truth

The `benchmark_real/` set holds the supervisor's hand-annotated validation
recording. It yields the in-domain detection mAP and the tracking HOTA/DetA/AssA
reported in the paper (detector mean mAP50 0.963 on the Basler cameras; greedy
tracker HOTA ~66, association-limited).
```bash
cd real_polybags/benchmark_real
python build_benchmark.py              # assemble GT + predictions in MOTChallenge layout
```

Full details: [`real_polybags/README.md`](real_polybags/README.md).

---

## 6. Known issues and gotchas

These cost real debugging time and are documented so they do not have to be
rediscovered.

- **Sequential multi-camera detection degradation.** Running all four synthetic
  cameras through one long-lived Python process quietly degrades detection on the
  later cameras. Run **one isolated subprocess per camera** for faithful numbers
  (this also frees memory). See `project_mcmot_benchmark_bugs`.
- **NumPy 2.0 removed aliases.** `motmetrics` needs `np.asfarray`, and TrackEval
  needs `np.float/int/bool/object/str`. Add shims at the top of any eval script:
  ```python
  import numpy as np
  for n, p in (("float",float),("int",int),("bool",bool),("object",object),("str",str)):
      if not hasattr(np, n): setattr(np, n, p)
  if not hasattr(np, "asfarray"):
      np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
  ```
- **MQTT on macOS binds IPv6 only.** Binding the broker to `localhost` listens on
  IPv6 alone, and the browser's IPv4 websocket is silently refused. Bind to
  `127.0.0.1` explicitly (already done in `mosquitto.conf`).
- **Recorded AVIs report an unreliable frame count and refuse to seek.** The
  streamer only reads forward and loops at end-of-file.
- **Not all cameras are metric.** Only the two Basler cameras are fully metric;
  `lucid` and `rgbd_1_color` are homography-only (accuracy falls off toward frame
  edges, drawn as dashed rings), and `rgbd_2_color` is uncalibrated. So the twin
  shows four cameras, not five.

---

## 7. Documentation index

| Document | Covers |
|---|---|
| [`DATA.md`](DATA.md) | **Google Drive assets and where to put them** |
| [`synthetic_polybags/README.md`](synthetic_polybags/README.md) | synthetic pipeline, dataset, class scheme, cluster training |
| `synthetic_polybags/docs/` | dataset + annotation reports (Blender OBB, MOT, coverage) |
| [`real_polybags/README.md`](real_polybags/README.md) | real pipeline overview and status |
| [`real_polybags/calibration/README.md`](real_polybags/calibration/README.md) | calibration method, plus `PROCEDURE`, `USER_MANUAL`, `VALIDATION_PLAN` |
| [`real_polybags/algorithm/README.md`](real_polybags/algorithm/README.md) | the digital twin: streamer, dashboard, MQTT schema, fusion, counting |
| [`real_polybags/AUTOLABEL_REPORT.md`](real_polybags/AUTOLABEL_REPORT.md) | the LocateAnything auto-labelling pipeline and results |
| `demo/documentation/` | write-up of the result video (method, dataset, twin integration) |

---

## Credits

AMS group, OVGU Magdeburg. Supervisor: Sai Preetham Sata. Repository:
[github.com/Awthura/MOT_of_Polybags](https://github.com/Awthura/MOT_of_Polybags).
