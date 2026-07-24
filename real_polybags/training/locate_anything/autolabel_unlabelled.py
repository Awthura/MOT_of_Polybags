"""
Batch inference of the fine-tuned LocateAnything-3B checkpoint over the full
~2,926-image unlabelled real-world dataset (5 camera folders), producing:
  - YOLO-format axis-aligned labels (single merged class, since the model
    doesn't distinguish class_0/class_1 and neither did our training data)
  - overlay images (predicted boxes drawn on the frame) for visual QA
  - a running summary JSON (counts/timing), written periodically so partial
    progress survives an interruption, plus per-camera raw-answer JSON dumps

Resumable: images whose label file already exists are skipped, so a killed/
timed-out job can just be re-launched with the same args.

Usage (on the cluster, inside the venv with the Eagle repo importable):
    python autolabel_unlabelled.py \
        --eagle-dir ~/Eagle \
        --model ~/runs_locateanything_lora/real_v3_multibox_sdpa \
        --dataset-dir ~/MOT_of_Polybags/real_polybags/dataset/unlabelled \
        --out-dir ~/MOT_of_Polybags/real_polybags/dataset/unlabelled_autolabels
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_CATEGORY = "translucent bubble-wrap polybag"


def draw_boxes(img_bgr, boxes, color=(0, 0, 220)):
    for b in boxes:
        pt1 = (int(b["x1"]), int(b["y1"]))
        pt2 = (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(img_bgr, pt1, pt2, color, 2)


def boxes_to_yolo_lines(boxes, img_w, img_h):
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eagle-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset-dir", required=True,
                     help="Path to dataset/unlabelled (parent of the 5 camera folders)")
    ap.add_argument("--out-dir", required=True,
                     help="Output root; labels/, overlays/, and summary.json go here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--category", default=DEFAULT_CATEGORY)
    ap.add_argument("--summary-every", type=int, default=100,
                     help="Write summary.json after every N images")
    args = ap.parse_args()
    prompt_categories = [args.category]

    sys.path.insert(0, str(Path(args.eagle_dir) / "Embodied"))
    from locateanything_worker import LocateAnythingWorker  # noqa: E402

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    labels_root = out_dir / "labels"
    overlays_root = out_dir / "overlays"
    labels_root.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} on {args.device} ...")
    t0 = time.time()
    worker = LocateAnythingWorker(args.model, device=args.device)
    print(f"Loaded in {time.time() - t0:.1f}s")

    cam_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    stats = {"per_camera": {}, "total_images": 0, "total_boxes": 0,
             "total_infer_seconds": 0.0, "skipped_existing": 0, "started": time.time()}

    n_done_since_summary = 0

    def write_summary():
        stats["elapsed_seconds"] = time.time() - stats["started"]
        (out_dir / "summary.json").write_text(json.dumps(stats, indent=2))

    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        images = sorted([p for p in cam_dir.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        cam_labels_dir = labels_root / cam_name
        cam_overlays_dir = overlays_root / cam_name
        cam_labels_dir.mkdir(parents=True, exist_ok=True)
        cam_overlays_dir.mkdir(parents=True, exist_ok=True)

        cam_stats = stats["per_camera"].setdefault(
            cam_name, {"n_images": 0, "n_boxes": 0, "infer_seconds": 0.0})

        print(f"\n[{cam_name}] {len(images)} images")
        for img_path in images:
            label_path = cam_labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                stats["skipped_existing"] += 1
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            t0 = time.time()
            result = worker.detect(img, prompt_categories)
            dt = time.time() - t0
            pred_boxes = worker.parse_boxes(result["answer"], w, h)

            label_path.write_text("\n".join(boxes_to_yolo_lines(pred_boxes, w, h)) + "\n"
                                   if pred_boxes else "")

            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            draw_boxes(cv_img, pred_boxes)
            cv2.imwrite(str(cam_overlays_dir / (img_path.stem + ".jpg")), cv_img)

            cam_stats["n_images"] += 1
            cam_stats["n_boxes"] += len(pred_boxes)
            cam_stats["infer_seconds"] += dt
            stats["total_images"] += 1
            stats["total_boxes"] += len(pred_boxes)
            stats["total_infer_seconds"] += dt

            print(f"  {img_path.name}: boxes={len(pred_boxes)} ({dt:.1f}s)")

            n_done_since_summary += 1
            if n_done_since_summary >= args.summary_every:
                write_summary()
                n_done_since_summary = 0

    write_summary()
    print(f"\nDone. total_images={stats['total_images']} "
          f"total_boxes={stats['total_boxes']} "
          f"skipped_existing={stats['skipped_existing']}")
    print(f"Labels -> {labels_root}")
    print(f"Overlays -> {overlays_root}")
    print(f"Summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
