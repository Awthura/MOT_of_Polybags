# Real Polybags — real-rig track

The physical-conveyor track of the project (see the [top-level README](../README.md)).
It carries the synthetic pipeline onto a real five-camera rig: two Basler GigE, one
Lucid Triton GigE, and two Intel RealSense D435 (colour + depth), recording a
one-directional, 580 mm-wide belt at 1280×720.

Two facts make the real data far harder than the simulation and shape the whole
design: the bags are a **single merged class** (no colour cue), and the moving
recordings have **no timestamps** (the cameras are not truly synchronised). Both
break the calibration-free association cues used on the synthetic track, which is
why the real track associates by **geometry** instead: every camera is mapped onto
one shared metric belt plane, and association becomes a distance in millimetres.

The five themes below match Section 5.2 of the report.

---

## 1. Calibration → a shared metric belt frame (`calibration/`)

A purpose-built Flask web tool solves per-camera intrinsics and **board-free**
extrinsics (each camera registered independently to a surveyed metric map of the
belt, so non-overlapping views are fine). Results are one JSON per camera in
`calibration/results/` and are **committed** — the twin runs without redoing this.

```bash
cd calibration && python app.py       # open the printed local URL
```

| Camera | Status |
|---|---|
| `basler_1` | metric, clean — trusted reference |
| `basler_2` | metric, weaker (~11 mm mean vs basler_1) |
| `lucid` | homography-only (across-belt up to ~236 mm off) |
| `rgbd_1_color` | homography-only (needs RealSense factory intrinsics) |
| `rgbd_2_color` | not calibrated |

See `calibration/README.md`, `PROCEDURE.md`, `USER_MANUAL.md`, `VALIDATION_PLAN.md`.

## 2. The metric belt map (`calibration/datasets/dashboard/`)

The shared frame is a top-down metric map surveyed by tape measure, origin under
`basler_1`, +Y along the belt and +X across it. A clean synthetic belt image at
1 mm/px (`conveyor_dashboard.png`) is used for the live display.

## 3. Detection and labelling

Real bags do not separate by colour, so the label set was built from scratch with a
promptable vision-language localiser.

- **Auto-labelling** (`training/locate_anything/`): NVIDIA **LocateAnything-3B**,
  LoRA fine-tuned on the supervisor's annotations, then used to auto-label all
  ~2,926 frames. Output cleaned with per-camera belt masks (temporal variance) and
  **tiled inference** (SAHI) to recover dense clusters. Result: `labels_final_v3`,
  7,184 boxes (~36% more than untiled). Full write-up: [`AUTOLABEL_REPORT.md`](AUTOLABEL_REPORT.md).
- **Deployed detector** (`training/yolo_polybag_v1/`): **YOLO11s** trained on
  `labels_final_v3`. Agreement with the auto-label teacher is mAP50 0.864; against
  the hand-annotated human set it reaches mean mAP50 **0.963** on the in-domain
  Basler cameras (§5). A segmentation variant (`training/yolo11seg_finetune/`) is
  available for mask overlays but has lower recall.

```bash
cd training/locate_anything && bash finetune_lora.sh && python autolabel_tiled.py
cd ../yolo_polybag_v1 && bash train.sh          # -> best.pt (provided on Drive)
```

## 4. The live digital twin (`algorithm/`)

Detection + calibration + the belt map combined into a live 2-D map over **MQTT**:
a streamer runs the detector per frame, projects each bag's belt-contact point to
world millimetres, publishes raw per-camera points (`/polybags`), fused
one-per-bag points (`/polybags_fused`), and occupancy-flux `counts`; a decoupled
browser dashboard draws all three. Full run instructions and the message schema:
[`algorithm/README.md`](algorithm/README.md).

```bash
cd algorithm
mosquitto -c mosquitto.conf
python streamer/app.py --scenario single --model detect
open http://127.0.0.1:5002/
```

**Fusion and counting.** Fusion merges detections only *across* cameras, never
within one, and keeps identity by greedy nearest-neighbour association on the belt
plane. Counting each bag by tracking it across a line is unreliable on the dense,
fast, low-frame-rate belt, so throughput is taken by **occupancy flux** (detection
occupancy in a thin band ÷ one bag's dwell time). Carrying one identity across the
blind gap in the middle of the belt is a genuine re-identification problem that was
scoped and **deliberately deferred**, not solved; the line count stands in for it.

## 5. Evaluation vs human ground truth (`benchmark_real/`)

The supervisor's hand-annotated validation recording (single moving polybags,
session of 2 June 2026, ~50 bags/camera) is the first human ground truth for the
real track. It gives the in-domain detection mAP above, and the tracking picture:
the deployed greedy tracker reaches **HOTA ~66**, decomposing into a high detection
term (DetA) and a lower association term (AssA) — detection is solved, association
is the limiter. The end-to-end occupancy-flux count recovers ~48 against a ground
truth of 50 on the direction-matching camera.

```bash
cd benchmark_real && python build_benchmark.py
```

---

## Directory layout

```
real_polybags/
├── calibration/         # metric-belt calibration web tool + committed results/
├── training/
│   ├── locate_anything/   # LocateAnything-3B LoRA fine-tune + tiled auto-label
│   ├── yolo_polybag_v1/   # deployed YOLO11s detector
│   └── yolo11seg_finetune/# segmentation variant (mask overlays)
├── benchmark_real/      # evaluation vs the hand-annotated human ground truth
├── algorithm/           # the live digital twin (streamer + dashboard + fusion + counting)
├── real_data/           # recording + sync utilities
└── annotation/          # LEGACY watershed auto-labelling (retired, kept for reference)
```

> The `annotation/` watershed scripts and the early superquadric/pseudo-label
> experiments are **superseded** by the LocateAnything pipeline and are retained
> only for reference. The large recordings, datasets, and weights are not in git;
> see [`../DATA.md`](../DATA.md).
