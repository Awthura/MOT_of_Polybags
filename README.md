# MOT of Polybags

Multi-Object Tracking (MOT) of polybags on a conveyor belt. Developed at the **Chair of Automation / Manufacturing Systems (AMS), OVGU Magdeburg**.

The project has two parallel tracks, kept as separate subdirectories in this repo:

## [`synthetic_polybags/`](synthetic_polybags/README.md)
Physics-based DEM simulation rendered in Blender from 4 cameras, with fully automatic YOLO OBB + OBB-MOT ground truth generation, MCMOT tracking, and MOTA/MOTP/IDF1 benchmarking.

## [`real_polybags/`](real_polybags/README.md)
Auto-labelling of real conveyor footage using watershed segmentation, with iterative pseudo-label YOLO training. The old dataset was retired; a new dataset from the supervisor is being integrated — see its README for current status.

---

## Setup

```bash
pip install -r requirements.txt
```

Each subdirectory's README has track-specific structure, pipeline, dataset, and setup details.
