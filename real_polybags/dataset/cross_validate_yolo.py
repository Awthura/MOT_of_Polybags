"""
Adversarial cross-validation of the LocateAnything-3B autolabels against an
independently-trained YOLO11n-OBB detector (both models trained on the same
569-image annotated set, but with completely different architectures/losses/
inductive biases -- agreement between them is a much stronger signal than
either model's own confidence).

Runs entirely locally (Mac, MPS backend, no cluster needed): YOLO11n-OBB is
small and fast (~0.05s/image batched).

For each of the 2,926 unlabelled images:
  - Run YOLO OBB inference, axis-align its boxes (both classes merged, since
    LocateAnything only predicts one merged category and class_0/class_1
    semantics are unconfirmed).
  - Load the cleaned LocateAnything autolabels (labels_cleaned/).
  - Greedy IoU matching (threshold 0.5) between the two independent box sets.

A box found by BOTH models is high-confidence evidence of a real polybag.
A box found by only one model is flagged for review -- it's either a false
positive from that model, or a miss from the other; cross-referencing the
overlay tells you which.

Usage:
    python cross_validate_yolo.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

DATASET_ROOT = Path(__file__).resolve().parent
UNLABELLED_IMAGES = DATASET_ROOT / "unlabelled"
AUTOLABELS_ROOT = DATASET_ROOT / "unlabelled_autolabels"
LA_LABELS_CLEANED = AUTOLABELS_ROOT / "labels_cleaned"
YOLO_WEIGHTS = DATASET_ROOT.parent / "preliminary_results" / "yolo_obb_detector" / "best.pt"

OUT_ROOT = AUTOLABELS_ROOT / "cross_validation"
YOLO_LABELS_OUT = OUT_ROOT / "yolo_labels"
REVIEW_OVERLAYS_OUT = OUT_ROOT / "review_overlays"

IOU_THRESH = 0.5
BATCH_SIZE = 16
N_REVIEW_OVERLAYS = 60  # top-N most-disagreeing images to render for review


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_la_boxes_px(cam_name, stem, img_w, img_h):
    label_path = LA_LABELS_CLEANED / cam_name / (stem + ".txt")
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = parts
            cx, cy, w, h = float(cx) * img_w, float(cy) * img_h, float(w) * img_w, float(h) * img_h
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def match(la_boxes, yolo_boxes):
    """Greedy best-IoU matching. Returns (n_matched, unmatched_la_idx, unmatched_yolo_idx)."""
    unmatched_yolo = list(range(len(yolo_boxes)))
    unmatched_la = []
    n_matched = 0
    for i, lb in enumerate(la_boxes):
        best_iou, best_j = 0.0, -1
        for j in unmatched_yolo:
            v = iou_xyxy(lb, yolo_boxes[j])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= IOU_THRESH:
            n_matched += 1
            unmatched_yolo.remove(best_j)
        else:
            unmatched_la.append(i)
    return n_matched, unmatched_la, unmatched_yolo


def draw(img_bgr, boxes, color, label):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(img_bgr, label, (10, 25 if color == (0, 200, 0) else 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def main():
    model = YOLO(str(YOLO_WEIGHTS))
    cam_dirs = sorted(d for d in UNLABELLED_IMAGES.iterdir() if d.is_dir())

    per_image_stats = []  # dicts, for sorting to find biggest disagreements
    per_camera = {}
    yolo_boxes_cache = {}  # (cam_name, stem) -> (boxes_px, img_w, img_h)

    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        images = sorted([p for p in cam_dir.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        cam_stats = per_camera.setdefault(cam_name, {
            "n_images": 0, "la_total": 0, "yolo_total": 0, "matched_total": 0,
        })
        yolo_out_dir = YOLO_LABELS_OUT / cam_name
        yolo_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{cam_name}] running YOLO on {len(images)} images ...")
        for start in range(0, len(images), BATCH_SIZE):
            batch = images[start:start + BATCH_SIZE]
            results = model.predict([str(p) for p in batch], device="mps",
                                     verbose=False, batch=len(batch))
            for img_path, r in zip(batch, results):
                img_h, img_w = r.orig_shape
                yolo_boxes = [tuple(b.tolist()) for b in r.obb.xyxy] if r.obb is not None else []

                # write YOLO's own labels (yolo format, class merged to 0)
                lines = []
                for (x1, y1, x2, y2) in yolo_boxes:
                    cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
                    w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                (yolo_out_dir / (img_path.stem + ".txt")).write_text(
                    "\n".join(lines) + ("\n" if lines else ""))

                la_boxes = load_la_boxes_px(cam_name, img_path.stem, img_w, img_h)
                n_matched, unmatched_la, unmatched_yolo = match(la_boxes, yolo_boxes)

                cam_stats["n_images"] += 1
                cam_stats["la_total"] += len(la_boxes)
                cam_stats["yolo_total"] += len(yolo_boxes)
                cam_stats["matched_total"] += n_matched

                disagreement = len(unmatched_la) + len(unmatched_yolo)
                per_image_stats.append({
                    "cam": cam_name, "stem": img_path.stem,
                    "n_la": len(la_boxes), "n_yolo": len(yolo_boxes),
                    "n_matched": n_matched, "disagreement": disagreement,
                    "la_boxes": la_boxes, "yolo_boxes": yolo_boxes,
                })
        print(f"  done: la={cam_stats['la_total']} yolo={cam_stats['yolo_total']} "
              f"matched={cam_stats['matched_total']}")

    total_la = sum(c["la_total"] for c in per_camera.values())
    total_yolo = sum(c["yolo_total"] for c in per_camera.values())
    total_matched = sum(c["matched_total"] for c in per_camera.values())

    summary = {
        "per_camera": per_camera,
        "total_la_boxes": total_la,
        "total_yolo_boxes": total_yolo,
        "total_matched": total_matched,
        "la_agreement_rate": total_matched / total_la if total_la else 0.0,
        "yolo_agreement_rate": total_matched / total_yolo if total_yolo else 0.0,
        "iou_thresh": IOU_THRESH,
    }
    (OUT_ROOT / "cross_validation_stats.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "cross_validation_stats.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Cross-validation summary ===")
    print(f"Total LocateAnything boxes: {total_la}")
    print(f"Total YOLO boxes:           {total_yolo}")
    print(f"Mutually matched (IoU>={IOU_THRESH}): {total_matched}")
    print(f"LA-box agreement rate (matched/LA):     {summary['la_agreement_rate']:.3f}")
    print(f"YOLO-box agreement rate (matched/YOLO): {summary['yolo_agreement_rate']:.3f}")

    # Render review overlays for the most-disagreeing images
    per_image_stats.sort(key=lambda d: d["disagreement"], reverse=True)
    REVIEW_OVERLAYS_OUT.mkdir(parents=True, exist_ok=True)
    n_rendered = 0
    for d in per_image_stats[:N_REVIEW_OVERLAYS]:
        if d["disagreement"] == 0:
            break
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = UNLABELLED_IMAGES / d["cam"] / (d["stem"] + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue
        img = Image.open(img_path).convert("RGB")
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        draw(cv_img, d["la_boxes"], (0, 0, 220), "red=LocateAnything")
        draw(cv_img, d["yolo_boxes"], (0, 200, 0), "green=YOLO")
        out_dir = REVIEW_OVERLAYS_OUT / d["cam"]
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / (d["stem"] + ".jpg")), cv_img)
        n_rendered += 1

    print(f"\nTop-disagreement review overlays rendered: {n_rendered} -> {REVIEW_OVERLAYS_OUT}")
    print(f"YOLO's own labels -> {YOLO_LABELS_OUT}")
    print(f"Stats -> {OUT_ROOT / 'cross_validation_stats.json'}")


if __name__ == "__main__":
    main()
