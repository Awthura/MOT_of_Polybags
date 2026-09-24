# polybag_v1 — training a better detector for the digital twin

Trains a YOLO11 **detect** model (single class `polybag`, axis-aligned) on the
tiled autolabels (`labels_final_v3`), to replace the preliminary
`yolo_obb_detector/best.pt` the streamer uses today. ~2,900 labelled frames
across all five cameras (basler_1/2, lucid, rgbd_1/2), so the model sees every
viewpoint.

The labels are **pseudo-labels** from LocateAnything, so this is distillation:
the goal is a fast, consistent YOLO that generalises the autolabeller's calls —
not to exceed the teacher. Expect fewer of the preliminary model's false
positives on rgbd/lucid and better recall on basler_1.

## Files
- `build_dataset.py` — assembles the YOLO layout (symlinked images + copied
  labels) with a contiguous per-camera **tail split** (last 15% → val) to avoid
  near-duplicate video frames leaking into val. Writes `dataset/data.yaml`.
- `train.sh` — the `yolo detect train` command, matching this cluster's srun +
  `~/venv` + `amp=False` conventions.

## Run on the OVGU cluster (ams / gpu-stud)

```bash
# 1. sync the repo up (labels + images travel; the built dataset/ does not —
#    it is rebuilt on the cluster so symlinks point at cluster-local paths)
rsync -av --exclude 'training/yolo_polybag_v1/dataset' \
      ~/OVGU/AMS/real_polybags/ diky85bu@<cluster>:~/MOT_of_Polybags/real_polybags/

# 2. build the dataset ON the cluster (bakes cluster-absolute paths)
cd ~/MOT_of_Polybags/real_polybags/training/yolo_polybag_v1
python3 build_dataset.py            # -> dataset/{images,labels}/{train,val} + data.yaml

# 3. train
srun -p gpu-stud --gres=gpu:1 -c 8 --mem=32G -w ant2 bash train.sh
#    result: ~/runs_polybag/polybag_v1/weights/best.pt
```

Tunables via env: `MODEL=yolo11n.pt` (faster Mac inference), `IMGSZ`, `BATCH`,
`EPOCHS`. Default is `yolo11s.pt @ 960`.

## Bring the model back into the digital twin

```bash
rsync -av diky85bu@<cluster>:~/runs_polybag/polybag_v1/weights/best.pt \
      ~/OVGU/AMS/real_polybags/training/yolo_polybag_v1/best.pt
```
Then point `algorithm/config.yaml` at it — the streamer handles both detect and
OBB models, so no code change:
```yaml
detector:
  model_path: "../training/yolo_polybag_v1/best.pt"
```

## Notes / caveats
- **Single-scale detect vs the tiled teacher.** The autolabels were made with
  tiling to catch small bags; a single-forward YOLO at 960 will be weaker on the
  smallest lucid/rgbd bags. If recall there matters, raise `IMGSZ` or add YOLO's
  built-in mosaic/scale aug (on by default).
- **Val is optimistic-safe, not distribution-clean.** The tail split reduces
  adjacent-frame leakage but train/val still share a session; treat mAP as
  relative progress, not an absolute field number.
- Re-run `build_dataset.py` whenever the autolabels are regenerated.
