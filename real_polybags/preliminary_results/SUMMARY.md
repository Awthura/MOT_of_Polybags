# Real Polybags — Preliminary Results (2026-07-22)

Status update on the new dataset (694 annotated images: 569 train / 125 val,
2 classes, from the 2026-05-28 conveyor experiments) supplied by the supervisor.

## 1. YOLO11n-OBB detector — `yolo_obb_detector/`

Trained directly on the annotated set (100 epochs, imgsz=640, batch=16),
using the same pipeline/conventions as the earlier synthetic-data model.
Training took under 5 minutes on an H100.

**Final validation metrics** (`results.csv`, epoch 100 / best.pt):
| Metric | Value |
|---|---|
| Precision | 0.874 |
| Recall | 0.957 |
| mAP50 | 0.944 |
| mAP50-95 | 0.781 |

Per-class (best.pt): `class_0` (1138 val instances) mAP50=0.907, mAP50-95=0.679;
`class_1` (28 val instances) mAP50=0.981, mAP50-95=0.883. Class semantics not
yet confirmed — see `real_polybags/README.md`.

Files: training curves (`results.png`), confusion matrix, precision/recall
curves, label distribution (`labels.jpg`), GT-vs-prediction comparisons on
3 validation batches (`val_batch*_labels.jpg` vs `val_batch*_pred.jpg`), and
the trained weights (`weights/best.pt`).

**This is the strongest result so far and the current candidate for the
production real-data detector.** It natively predicts oriented boxes (OBB),
matching the annotation format directly — no conversion needed.

## 2. LocateAnything-3B — zero-shot pilot — `locate_anything_zero_shot_pilot/`

NVIDIA's open-vocabulary grounding VLM, tested with **no fine-tuning**, prompted
for "translucent bubble-wrap polybag", on a small sample (15 annotated images +
25 images across all 5 real conveyor camera feeds).

**Result**: mean precision=0.70, mean recall=0.74 (IoU>=0.5) on the annotated
sample. Works well on well-separated bags; its main weakness is merging dense,
touching clusters of bags into a single box instead of enumerating them
individually. Correctly returns nothing on genuinely empty conveyor frames
(no false positives observed).

Files: overlay images (green=ground truth, red=prediction) for the annotated
sample, and prediction-only overlays for each of the 5 unlabelled camera feeds,
plus per-image precision/recall (`annotated_scores.json`).

Fine-tuning this model (LoRA) was attempted but is currently blocked by a
cluster environment issue (CUDA toolkit version mismatch preventing the
custom-kernel build it needs) — in progress, not abandoned.

## Next steps
- Confirm what `class_0` / `class_1` actually represent (colors? bag types?)
- Run the YOLO detector across the full ~2,927-image unlabelled set to
  produce first-pass labels
- Resolve the LocateAnything fine-tuning environment blocker, or treat YOLO
  as the primary detector going forward
