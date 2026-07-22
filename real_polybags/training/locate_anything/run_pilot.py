"""
Zero-shot pilot for NVIDIA LocateAnything-3B on the polybag dataset.

Runs the pretrained (not fine-tuned) model against a small sample of images
from both domains:
  pilot_sample/annotated/    - clean red-backdrop bubble-wrap bags (have GT OBB labels)
  pilot_sample/unlabelled/   - real conveyor-belt footage, 5 camera folders (no GT)

For the annotated sample, GT OBB labels are converted to axis-aligned boxes
and compared to the model's predictions via IoU (best-match greedy pairing),
giving a rough precision/recall signal. For the unlabelled sample there's no
GT, so only overlays are produced for visual review.

Usage (on the cluster, inside the venv with the Eagle repo importable):
    python run_pilot.py --eagle-dir ~/Eagle --out-dir results

Output:
    results/overlays/<domain>/<name>.jpg   - boxes drawn on the image
    results/predictions.json               - raw model output per image
    results/annotated_scores.json          - per-image IoU precision/recall
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROMPT_CATEGORIES = ["translucent bubble-wrap polybag"]

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = SCRIPT_DIR / "pilot_sample"


# ── GT conversion (OBB -> axis-aligned, for IoU scoring) ─────────────────────

def load_gt_boxes(label_path: Path, img_w: int, img_h: int):
    """YOLO-OBB (class x1 y1 x2 y2 x3 y3 x4 y4, normalized) -> axis-aligned px boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        cid = int(parts[0])
        coords = list(map(float, parts[1:]))
        xs = [coords[i] * img_w for i in range(0, 8, 2)]
        ys = [coords[i + 1] * img_h for i in range(0, 8, 2)]
        boxes.append({"cls": cid, "x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)})
    return boxes


def iou(a, b):
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def score_predictions(preds, gts, thresh=0.5):
    """Greedy best-IoU matching. Returns (precision, recall, matched_ious)."""
    unmatched_gt = list(range(len(gts)))
    matched_ious = []
    tp = 0
    for p in preds:
        best_iou, best_j = 0.0, -1
        for j in unmatched_gt:
            v = iou(p, gts[j])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= thresh:
            tp += 1
            matched_ious.append(best_iou)
            unmatched_gt.remove(best_j)
    precision = tp / len(preds) if preds else 0.0
    recall = tp / len(gts) if gts else 0.0
    return precision, recall, matched_ious


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_boxes(img_bgr, boxes, color, label=None):
    for b in boxes:
        pt1 = (int(b["x1"]), int(b["y1"]))
        pt2 = (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(img_bgr, pt1, pt2, color, 2)
        if label:
            cv2.putText(img_bgr, label, (pt1[0], max(0, pt1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eagle-dir", required=True,
                    help="Path to the cloned NVlabs/Eagle repo (parent of Embodied/)")
    ap.add_argument("--model", default="nvidia/LocateAnything-3B")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.eagle_dir) / "Embodied"))
    from locateanything_worker import LocateAnythingWorker  # noqa: E402

    out_dir = Path(args.out_dir)
    (out_dir / "overlays" / "annotated").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays" / "unlabelled").mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} on {args.device} ...")
    t0 = time.time()
    worker = LocateAnythingWorker(args.model, device=args.device)
    print(f"Loaded in {time.time() - t0:.1f}s")

    predictions = {}
    annotated_scores = {}

    # ── Annotated sample (scored against GT) ──────────────────────────────
    ann_dir = SAMPLE_DIR / "annotated"
    ann_images = sorted((ann_dir / "images").glob("*.png"))
    print(f"\n[annotated] {len(ann_images)} images")
    for img_path in ann_images:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        t0 = time.time()
        result = worker.detect(img, PROMPT_CATEGORIES)
        dt = time.time() - t0
        pred_boxes = worker.parse_boxes(result["answer"], w, h)

        gt_boxes = load_gt_boxes(ann_dir / "labels" / (img_path.stem + ".txt"), w, h)
        precision, recall, ious = score_predictions(pred_boxes, gt_boxes)

        predictions[f"annotated/{img_path.name}"] = {
            "raw_answer": result["answer"], "boxes": pred_boxes, "infer_seconds": dt,
        }
        annotated_scores[img_path.name] = {
            "n_pred": len(pred_boxes), "n_gt": len(gt_boxes),
            "precision": precision, "recall": recall,
            "mean_iou_matched": float(np.mean(ious)) if ious else 0.0,
        }
        print(f"  {img_path.name}: pred={len(pred_boxes)} gt={len(gt_boxes)} "
              f"P={precision:.2f} R={recall:.2f} ({dt:.1f}s)")

        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        draw_boxes(cv_img, gt_boxes, (0, 200, 0), "gt")
        draw_boxes(cv_img, pred_boxes, (0, 0, 220), "pred")
        cv2.imwrite(str(out_dir / "overlays" / "annotated" / img_path.name), cv_img)

    # ── Unlabelled sample (visual only, no GT) ────────────────────────────
    unl_dir = SAMPLE_DIR / "unlabelled"
    for cam_dir in sorted(unl_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        cam_out = out_dir / "overlays" / "unlabelled" / cam_dir.name
        cam_out.mkdir(parents=True, exist_ok=True)
        images = sorted(cam_dir.glob("*.jpg"))
        print(f"\n[unlabelled/{cam_dir.name}] {len(images)} images")
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            t0 = time.time()
            result = worker.detect(img, PROMPT_CATEGORIES)
            dt = time.time() - t0
            pred_boxes = worker.parse_boxes(result["answer"], w, h)

            predictions[f"unlabelled/{cam_dir.name}/{img_path.name}"] = {
                "raw_answer": result["answer"], "boxes": pred_boxes, "infer_seconds": dt,
            }
            print(f"  {img_path.name}: pred={len(pred_boxes)} ({dt:.1f}s)")

            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            draw_boxes(cv_img, pred_boxes, (0, 0, 220), "pred")
            cv2.imwrite(str(cam_out / img_path.name), cv_img)

    (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))
    (out_dir / "annotated_scores.json").write_text(json.dumps(annotated_scores, indent=2))

    if annotated_scores:
        mean_p = np.mean([s["precision"] for s in annotated_scores.values()])
        mean_r = np.mean([s["recall"] for s in annotated_scores.values()])
        print(f"\n=== Annotated-sample summary: mean precision={mean_p:.2f}, "
              f"mean recall={mean_r:.2f} (IoU>=0.5) ===")
    print(f"\nDone. Overlays + JSON written to {out_dir}/")


if __name__ == "__main__":
    main()
