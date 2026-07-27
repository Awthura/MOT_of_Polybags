"""
Standalone test-only generator for the ORIGINAL multi-box-per-image JSONL
format (all boxes for an image concatenated in one GPT response) — used to
verify the mask_magi_utils.py patch actually fixes the crash it was written
for, since prepare_jsonl.py itself now emits one-box-per-example instead.
Not part of the normal pipeline; delete once the patch is validated.
"""

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
        round(x1 / img_w * 1000), round(y1 / img_h * 1000),
        round(x2 / img_w * 1000), round(y2 / img_h * 1000),
    )


def main():
    dataset_dir = Path("../../dataset/annotated/train_v11_obb_final")
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    out_dir = Path("jsonl_data_multibox_test")
    out_dir.mkdir(exist_ok=True)

    lines = []
    for img_path in sorted(images_dir.glob("*.png")):
        label_path = labels_dir / (img_path.stem + ".txt")
        with Image.open(img_path) as im:
            w, h = im.size
        boxes_str = ""
        n = 0
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                x1, y1, x2, y2 = obb_to_box_1000(parts, w, h)
                boxes_str += f"<ref>{CATEGORY}</ref><box><{x1}><{y1}><{x2}><{y2}></box>"
                n += 1
        if n < 2:
            continue  # only want multi-box samples for this crash test
        lines.append({
            "image": f"images/{img_path.name}",
            "conversations": [
                {"from": "human", "value": f"<image-1>{PROMPT}"},
                {"from": "gpt", "value": boxes_str},
            ],
        })

    out_path = out_dir / "train.jsonl"
    with open(out_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    print(f"{len(lines)} multi-box images -> {out_path}")

    recipe = {
        "real_polybags_train_multibox_test": {
            "annotation": "/home/diky85bu/MOT_of_Polybags/real_polybags/training/locate_anything/jsonl_data_multibox_test/train.jsonl",
            "root": "/home/diky85bu/MOT_of_Polybags/real_polybags/dataset/annotated/train_v11_obb_final",
            "repeat_time": 1.0,
        }
    }
    (out_dir / "recipe.json").write_text(json.dumps(recipe, indent=2))
    print(f"recipe -> {out_dir / 'recipe.json'}")


if __name__ == "__main__":
    main()
