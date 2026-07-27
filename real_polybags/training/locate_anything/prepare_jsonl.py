"""
Convert the annotated real-polybag dataset (train_v11_obb_final /
val_v11_obb_final, YOLO-OBB format) into LocateAnything's ShareGPT-style
JSONL training format.

OBB (8 normalized coords) -> axis-aligned box -> normalized int [0, 1000],
matching LocateAnything's <box><x1><y1><x2><y2></box> convention. Both class
0 and class 1 are merged into a single category (matches the zero-shot pilot
prompt) since we don't yet know what the two raw class ids represent.

ONE TRAINING EXAMPLE PER BOX, not per image: NVlabs' vendored training code
(locany_finetune_magi_stream.py's MTP/stream-packing path) has a real bug
that corrupts its attention-range computation whenever a single training
example has more than one supervised detection ("block") in its GPT answer —
confirmed via a deterministic "illegal memory access" CUDA crash, root-caused
to _find_sample_x0_len_packed (mask_sdpa_utils.py) only handling the first
of multiple position_id resets, plus a stride mismatch in
convert_mtp_mask_to_magi_plan (mask_magi_utils.py). Emitting each box as its
own (image, single-box-answer) example sidesteps this entirely — every
example has exactly one detection, so the multi-block condition never
occurs. This does expand the dataset substantially (~8000 examples from 569
images) — see finetune_lora.sh's EPOCHS default, tuned down accordingly.

Usage:
    python prepare_jsonl.py --dataset-dir ../../dataset/annotated --out-dir jsonl_data

Output:
    jsonl_data/train.jsonl
    jsonl_data/val.jsonl
    jsonl_data/recipe.json
"""

import argparse
import json
from pathlib import Path

from PIL import Image

CATEGORY = "translucent bubble-wrap polybag"
PROMPT = f"Locate all the instances that matches the following description: {CATEGORY}."


def obb_to_box_1000(parts, img_w, img_h):
    coords = list(map(float, parts[1:]))
    xs = [coords[i] * img_w for i in range(0, 8, 2)]
    ys = [coords[i + 1] * img_h for i in range(0, 8, 2)]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return (
        round(x1 / img_w * 1000),
        round(y1 / img_h * 1000),
        round(x2 / img_w * 1000),
        round(y2 / img_h * 1000),
    )


def convert_split(images_dir: Path, labels_dir: Path):
    lines = []
    n_boxes = 0
    n_images_no_boxes = 0
    for img_path in sorted(images_dir.glob("*.png")):
        label_path = labels_dir / (img_path.stem + ".txt")
        with Image.open(img_path) as im:
            w, h = im.size

        boxes = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                boxes.append(obb_to_box_1000(parts, w, h))

        if not boxes:
            # One example, "none" answer — no per-box splitting possible.
            n_images_no_boxes += 1
            lines.append({
                "image": f"images/{img_path.name}",
                "conversations": [
                    {"from": "human", "value": f"<image-1>{PROMPT}"},
                    {"from": "gpt", "value": "<box>none</box>"},
                ],
            })
            continue

        # One training example PER BOX — see module docstring for why.
        for (x1, y1, x2, y2) in boxes:
            n_boxes += 1
            gpt_answer = f"<ref>{CATEGORY}</ref><box><{x1}><{y1}><{x2}><{y2}></box>"
            lines.append({
                "image": f"images/{img_path.name}",
                "conversations": [
                    {"from": "human", "value": f"<image-1>{PROMPT}"},
                    {"from": "gpt", "value": gpt_answer},
                ],
            })
    if n_images_no_boxes:
        print(f"  ({n_images_no_boxes} image(s) had no boxes at all — kept as single 'none' examples)")
    return lines, n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True,
                     help="Path to dataset/annotated (parent of train_v11_obb_final/val_v11_obb_final)")
    ap.add_argument("--out-dir", default="jsonl_data")
    ap.add_argument("--cluster-root",
                     default="/home/diky85bu/MOT_of_Polybags/real_polybags",
                     help="real_polybags root as it appears on the training cluster, "
                          "used to write recipe.json with absolute cluster paths "
                          "(annotation/root must resolve there, not on this Mac)")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": dataset_dir / "train_v11_obb_final",
        "val": dataset_dir / "val_v11_obb_final",
    }
    recipe = {}
    for split_name, split_dir in splits.items():
        lines, n_boxes = convert_split(split_dir / "images", split_dir / "labels")
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        print(f"{split_name}: {len(lines)} images, {n_boxes} boxes -> {out_path}")

        if split_name == "train":
            cluster_root = args.cluster_root.rstrip("/")
            recipe["real_polybags_train"] = {
                "annotation": f"{cluster_root}/training/locate_anything/jsonl_data/train.jsonl",
                "root": f"{cluster_root}/dataset/annotated/{split_dir.name}",
                "repeat_time": 1.0,
            }

    recipe_path = out_dir / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2))
    print(f"recipe -> {recipe_path}")


if __name__ == "__main__":
    main()
