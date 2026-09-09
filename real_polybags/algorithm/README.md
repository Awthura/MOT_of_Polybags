# algorithm/ — camera → digital-twin live map

First step of the digital-twin track. Plays the recorded camera videos, runs
YOLO on each frame, projects every polybag's ground-contact point onto the belt
with the calibration solved in `../calibration/`, and shows the positions live
on a 2-D map of the conveyor. Transport is **MQTT** (topic `/polybags`).

**No cross-camera fusion yet** — each camera's detections are drawn on the map
as-is (own colour, own IDs). A bag seen by two cameras appears as two dots.

```
 4 synced .avi ─► streamer (Flask)
   (one session)   ├─ YOLO per frame per camera
                   ├─ foot-point → homography → world mm   (calibration/core reused)
                   ├─ per-camera nearest-neighbour track IDs
                   ├─ publish {polybags:[…]} ─► MQTT /polybags ─┐
                   └─ serve /stream/<cam> annotated MJPEG ──┐   │
                                                            ▼   ▼
        combined page:  LEFT  4 feeds  |  RIGHT  dashboard (mqtt.js → Canvas over map.png)
```

The **streamer** is the producer; the **dashboard** is a decoupled static
consumer that only needs the MQTT bus — it works with any publisher of the same
message shape, not just this video replay.

## Prerequisites

- Python deps: `pip install -r requirements.txt` (paho-mqtt was the only one not
  already present alongside the calibration tool).
- Broker: Mosquitto (`brew install mosquitto`).

## Run (three steps)

```bash
# 1. broker — two listeners: 1883 tcp (streamer) + 9001 websockets (browser)
mosquitto -c algorithm/mosquitto.conf

# 2. streamer — one scenario at a time (all use port 5002). Runs from any CWD.
ALGO=/Users/awthura/OVGU/AMS/real_polybags/algorithm
python3 $ALGO/streamer/app.py --scenario static    # timestamped 4-cam, true sync (static bags)
python3 $ALGO/streamer/app.py --scenario single    # single polybags, moving
python3 $ALGO/streamer/app.py --scenario bulk      # bulk polybags, moving

# choose the detector at launch (default: seg):
python3 $ALGO/streamer/app.py --scenario single --model detect   # yolo11s bboxes
python3 $ALGO/streamer/app.py --scenario single --model seg      # yolo11n-seg + masks

# 3. open the combined window
open http://127.0.0.1:5002/
```

The three curated sets live under `algorithm/videos/{static,single,bulk}/`.

`http://127.0.0.1:5002/` is the combined window (4 feeds + twin).
`http://127.0.0.1:5002/dashboard/` is the twin on its own.
Sanity-check the bus without a browser: `mosquitto_sub -t /polybags -v`.

## Configuration — `config.yaml`

- `scenario` / `scenarios` — three curated sets under `videos/`: `static`
  (timestamped 4-cam, TRUE sync, static bags), `single` and `bulk` (both moving,
  3 cams, normalized/approximate sync). Pick with the `scenario:` field or
  `--scenario` at launch.
- `detector.model_path` — **swap the polybag model here**; nothing else changes.
  Currently the fine-tuned `yolo11seg_finetune` (segmentation; the pipeline uses
  its boxes). Handles detect, OBB, and segment models.
- `playback.fps`, `detector.imgsz` — throughput knobs (see Notes).
- `mqtt` — broker host/ports and the `/polybags` topic.

## MQTT message (`/polybags`)

```json
{ "t": 1699999999123, "frame": 218, "session": "20260727_140907",
  "polybags": [
    {"cam":"basler_1","id":3,"x_mm":-136.5,"y_mm":-97.6,"conf":0.82,"metric":true},
    {"cam":"lucid","id":1,"x_mm":210.0,"y_mm":40.2,"conf":0.71,"metric":false}
  ] }
```

`metric:false` = a homography-only camera (lucid, rgbd_1_color): the plane
mapping is real but lens distortion is uncorrected, so accuracy falls off toward
frame edges. The dashboard draws those as dashed rings.

## How it maps a detection to the belt

1. Foot-point in image pixels — bottom-centre for an axis-aligned box, lowest
   edge midpoint for an OBB (`core/footpoint.py`). Only the belt-contact point
   projects correctly through a plane homography.
2. `core/calib.py` loads `results/<cam>.json` and applies the calibration
   tool's own `beltmap.CameraOnBelt.to_belt` — undistort-then-homography for a
   metric camera (basler_1/2), direct homography for a homography-only one.
3. World mm → map pixel on the dashboard: `origin_px + world_mm / mm_per_px`
   (from `results/_workspace/meta.json`), mirroring `worldmap.WorkspaceMap`.

Verify the transform in isolation: `python3 algorithm/smoke_test.py` — each
camera must reproduce its stored fit RMS through the raw-frame inference path.

## Notes

- The recorded AVIs report an unreliable frame count and refuse to seek; the
  streamer only reads forward and loops at end-of-file.
- Throughput: four YOLO inferences per tick. On this MacBook it runs ~3 fps at
  `imgsz 960`; lower `imgsz` or `playback.fps` to trade accuracy for smoothness.
  Publishing is decoupled from the feed rate, so a slow model just lowers the
  update rate — it never breaks the pipeline.
- Video sync is naive (same wall-clock session, read one frame per camera per
  tick). Per-camera start-offset alignment (`../real_data/utils/measure_sync.py`)
  is a later item.
- `rgbd_2_color` has no calibration result → four cameras, not five.
