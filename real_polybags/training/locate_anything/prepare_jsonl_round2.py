"""
Round-2 domain-adaptive fine-tuning data: combine the original 569-image
clean annotated set with pseudo-labels from the 2,926-image real conveyor
dataset, to adapt LocateAnything from the clean isolated-bag/red-backdrop
training domain to the cluttered real conveyor domain -- see
AUTOLABEL_REPORT.md "Next steps".

Pseudo-label selection (per unlabelled image, after belt-masking -- see
build_belt_masks.py/apply_belt_masks.py):
  - "confident positive": LocateAnything and YOLO (independently trained,
    cross-validated via cross_validate_yolo.py) mutually agree (IoU>=0.5) on
    >=1 box. Only the matched (agreed-upon) boxes are used as targets --
    unmatched boxes from either model are dropped, since we don't have a
    reliable independent signal to resolve which one is right.
  - "confident none": both models detect zero boxes after masking. Used as
    explicit negative examples.
  - "ambiguous, excluded": some boxes exist but none matched. We don't have
    ground truth to resolve these, so they're left out of training rather
    than risk injecting a wrong label (positive or negative) as a training
    target.

Both LA and YOLO's own false-positive rates on static background clutter
are camera-dependent and sometimes severe (see AUTOLABEL_REPORT.md section
5) -- this is exactly why only *mutual* agreement is trusted here, not
either model's raw output alone.

Uses the same multi-box-per-image JSONL format as jsonl_data_multibox_test/
(confirmed safe to train with `--attn_implementation sdpa`, see finetune_lora.sh).

IMPORTANT -- image resolution: the annotated set is 640x480 (4:3), the
unlabelled/pseudo set is 1280x720 (16:9). Mixing these directly in one
training run reproducibly crashed with a NaN forward pass within ~12-19
steps (2 separate attempts, including one that ruled out <box>none</box>
examples as the cause) -- root-caused to the resolution/aspect-ratio
mismatch, since real_v3 (640x480-only, no mixing) trained 858 steps with
zero crashes. Fix: letterbox-resize every pseudo image down to 640x480
(scale to fit, pad with black bars, no distortion) before use, and remap
box coordinates through the same transform. Resized images + a matching
label are written under <out-dir>/images_640x480_letterboxed/.

Usage:
    python prepare_jsonl_round2.py --out-dir jsonl_data_round2
"""

import argparse
import json
from pathlib import Path

from PIL import Image

CATEGORY = "translucent bubble-wrap polybag"
PROMPT = f"Locate all the instances that matches the following description: {CATEGORY}."

IOU_THRESH = 0.5
SRC_W, SRC_H = 1280, 720
DST_W, DST_H = 640, 480
LETTERBOX_SCALE = min(DST_W / SRC_W, DST_H / SRC_H)
LETTERBOX_NEW_W = round(SRC_W * LETTERBOX_SCALE)
LETTERBOX_NEW_H = round(SRC_H * LETTERBOX_SCALE)
LETTERBOX_PAD_X = (DST_W - LETTERBOX_NEW_W) // 2
LETTERBOX_PAD_Y = (DST_H - LETTERBOX_NEW_H) // 2


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


def load_boxes_px(path, w, h):
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
            boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes


def matched_boxes(la_boxes, yolo_boxes):
    unmatched_yolo = list(range(len(yolo_boxes)))
    matched = []
    for lb in la_boxes:
        best_iou, best_j = 0.0, -1
        for j in unmatched_yolo:
            v = iou_xyxy(lb, yolo_boxes[j])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= IOU_THRESH:
            matched.append(lb)
            unmatched_yolo.remove(best_j)
    return matched


def box_to_1000(box, w, h):
    x1, y1, x2, y2 = box
    return (round(x1 / w * 1000), round(y1 / h * 1000), round(x2 / w * 1000), round(y2 / h * 1000))


def letterbox_image(src_path: Path, dst_path: Path):
    with Image.open(src_path) as im:
        im = im.convert("RGB").resize((LETTERBOX_NEW_W, LETTERBOX_NEW_H), Image.BILINEAR)
        canvas = Image.new("RGB", (DST_W, DST_H), (0, 0, 0))
        canvas.paste(im, (LETTERBOX_PAD_X, LETTERBOX_PAD_Y))
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dst_path, quality=95)


def letterbox_box_to_1000(box_src_px):
    """Map a box in original 1280x720 pixel space through the same
    scale+pad transform as letterbox_image, then normalize to 0-1000."""
    x1, y1, x2, y2 = box_src_px
    nx1 = x1 * LETTERBOX_SCALE + LETTERBOX_PAD_X
    ny1 = y1 * LETTERBOX_SCALE + LETTERBOX_PAD_Y
    nx2 = x2 * LETTERBOX_SCALE + LETTERBOX_PAD_X
    ny2 = y2 * LETTERBOX_SCALE + LETTERBOX_PAD_Y
    return (
        round(nx1 / DST_W * 1000), round(ny1 / DST_H * 1000),
        round(nx2 / DST_W * 1000), round(ny2 / DST_H * 1000),
    )


def gpt_answer(boxes_1000):
    if not boxes_1000:
        return "<box>none</box>"
    return "".join(
        f"<ref>{CATEGORY}</ref><box><{x1}><{y1}><{x2}><{y2}></box>"
        for (x1, y1, x2, y2) in boxes_1000
    )


def build_pseudo_examples(dataset_root: Path, images_out_dir: Path, include_none: bool = True):
    la_root = dataset_root / "unlabelled_autolabels" / "labels_cleaned_masked"
    yolo_root = dataset_root / "unlabelled_autolabels" / "cross_validation" / "yolo_labels_masked"
    unlabelled_root = dataset_root / "unlabelled"
    w, h = SRC_W, SRC_H

    lines = []
    n_positive = n_none = n_excluded = 0
    n_boxes = 0
    for cam_dir in sorted(la_root.iterdir()):
        cam = cam_dir.name
        yolo_dir = yolo_root / cam
        for f in sorted(cam_dir.glob("*.txt")):
            la_boxes = load_boxes_px(f, w, h)
            yolo_boxes = load_boxes_px(yolo_dir / f.name, w, h)
            matched = matched_boxes(la_boxes, yolo_boxes)

            if matched:
                n_positive += 1
                n_boxes += len(matched)
                boxes_1000 = [letterbox_box_to_1000(b) for b in matched]
            elif not la_boxes and not yolo_boxes:
                n_none += 1
                if not include_none:
                    continue
                boxes_1000 = []
            else:
                n_excluded += 1
                continue

            img_name = f.stem + ".jpg"
            src_img = unlabelled_root / cam / img_name
            dst_img = images_out_dir / cam / img_name
            if not dst_img.exists():
                letterbox_image(src_img, dst_img)

            lines.append({
                "image": f"{cam}/{img_name}",
                "conversations": [
                    {"from": "human", "value": f"<image-1>{PROMPT}"},
                    {"from": "gpt", "value": gpt_answer(boxes_1000)},
                ],
            })

    print(f"pseudo-labelled: {n_positive} positive images ({n_boxes} boxes), "
          f"{n_none} confident-none images, {n_excluded} ambiguous (excluded)")
    print(f"letterboxed images (640x480, matching annotated set) -> {images_out_dir}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="../../dataset",
                     help="real_polybags/dataset, containing unlabelled/ and unlabelled_autolabels/")
    ap.add_argument("--out-dir", default="jsonl_data_round2")
    ap.add_argument("--annotated-recipe-entry", default="jsonl_data_multibox_test/train.jsonl",
                     help="path (relative to this script) to the existing 569-image multi-box JSONL")
    ap.add_argument("--annotated-repeat", type=float, default=3.0,
                     help="repeat_time for the clean annotated set, to keep its influence up "
                          "against the larger noisier pseudo-labelled set")
    ap.add_argument("--cluster-root", default="/home/diky85bu/MOT_of_Polybags/real_polybags",
                     help="real_polybags root as it appears on the training cluster")
    ap.add_argument("--no-none-examples", action="store_true",
                     help="exclude confident-none (empty-box) pseudo examples -- real_v3's working "
                          "training data was all-positive, so this isolates whether a from-scratch "
                          "NaN crash is related to the newly-introduced <box>none</box> examples")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir = out_dir / "images_640x480_letterboxed"

    pseudo_lines = build_pseudo_examples(dataset_root, images_out_dir, include_none=not args.no_none_examples)
    pseudo_path = out_dir / "pseudo_unlabelled.jsonl"
    with open(pseudo_path, "w") as fh:
        for line in pseudo_lines:
            fh.write(json.dumps(line) + "\n")
    print(f"pseudo_unlabelled: {len(pseudo_lines)} images -> {pseudo_path}")

    cluster_root = args.cluster_root.rstrip("/")
    annotated_jsonl_path = Path(args.annotated_recipe_entry)
    n_annotated = sum(1 for _ in open(annotated_jsonl_path))

    recipe = {
        "real_polybags_annotated": {
            "annotation": f"{cluster_root}/training/locate_anything/{args.annotated_recipe_entry}",
            "root": f"{cluster_root}/dataset/annotated/train_v11_obb_final",
            "repeat_time": args.annotated_repeat,
        },
        "real_polybags_pseudo_unlabelled": {
            "annotation": f"{cluster_root}/training/locate_anything/{out_dir.name}/pseudo_unlabelled.jsonl",
            "root": f"{cluster_root}/training/locate_anything/{out_dir.name}/images_640x480_letterboxed",
            "repeat_time": 1.0,
        },
    }
    recipe_path = out_dir / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2))

    n_effective = n_annotated * args.annotated_repeat + len(pseudo_lines)
    print(f"recipe -> {recipe_path}")
    print(f"annotated: {n_annotated} images x repeat {args.annotated_repeat} = "
          f"{n_annotated * args.annotated_repeat:.0f} effective")
    print(f"pseudo: {len(pseudo_lines)} images x repeat 1.0")
    print(f"total effective images/epoch: {n_effective:.0f}")


if __name__ == "__main__":
    main()
