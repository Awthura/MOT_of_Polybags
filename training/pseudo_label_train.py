"""
Iterative pseudo-labelling with YOLO11-OBB

Round 0: train on 86 hand-annotated images only
Round N: train on GT + pseudo-labels from previous round's predictions
         High-confidence predictions replace auto-labels; low-confidence
         frames keep their colour-based auto-label.

Directory layout created here:
  pseudo_label/
  ├── round_0/
  │   ├── images/train  ← 86 GT images (symlinks)
  │   ├── images/val    ← held-out 20% of GT
  │   ├── labels/train
  │   ├── labels/val
  │   └── data.yaml
  ├── round_1/          ← built after round_0 training
  │   ├── images/{train,val}
  │   ├── labels/{train,val}
  │   └── data.yaml
  └── ...

Usage
-----
  # 1. Run round 0 training (edit EPOCHS / IMGSZ as needed):
  python pseudo_label_train.py --setup 0
  yolo obb train data=pseudo_label/round_0/data.yaml model=yolo11n-obb.pt \
       epochs=100 imgsz=1920 batch=4 name=round_0 project=pseudo_label/runs

  # 2. Generate round 1 dataset from round 0 weights:
  python pseudo_label_train.py --setup 1 \
      --weights pseudo_label/runs/round_0/weights/best.pt \
      --conf 0.35

  # 3. Train round 1, then repeat with --setup 2, etc.
"""

import argparse
import shutil
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
parser_pre = argparse.ArgumentParser(add_help=False)
parser_pre.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1])
pre_args, _ = parser_pre.parse_known_args()

BASE       = pre_args.base
DATASET    = BASE / "full_dataset"
LABELS_ALL = DATASET / "labels_all"   # one .txt per image (manual+auto)
PL_ROOT    = BASE / "pseudo_label"

CLASSES = ["pink_polybag", "blue_polybag", "yellow_polybag",
           "grey_polybag",  "green_polybag", "red_polybag"]

# GT images = those that have a manual label
gt_stems   = {p.stem for p in (DATASET / "labels_manual").glob("*.txt")}
all_images = sorted((DATASET / "images").glob("*.png"))
gt_images  = [p for p in all_images if p.stem in gt_stems]
unl_images = [p for p in all_images if p.stem not in gt_stems]

VAL_RATIO   = 0.15   # fraction of GT held out for validation every round


def make_yaml(data_dir: Path) -> Path:
    yaml = data_dir / "data.yaml"
    yaml.write_text(
        f"path: {data_dir}\n"
        f"train: images/train\n"
        f"val:   images/val\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    return yaml


def symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def setup_round_0():
    """Round 0: only the 86 GT images."""
    rdir = PL_ROOT / "round_0"
    print(f"\n[Round 0] Setting up {rdir} …")

    random.seed(42)
    shuffled = list(gt_images)
    random.shuffle(shuffled)
    n_val   = max(1, int(len(shuffled) * VAL_RATIO))
    val_set = set(p.stem for p in shuffled[:n_val])

    counts = {"train": 0, "val": 0}
    for img in gt_images:
        split = "val" if img.stem in val_set else "train"
        symlink(img,                                  rdir / "images" / split / img.name)
        symlink(LABELS_ALL / (img.stem + ".txt"),     rdir / "labels" / split / (img.stem + ".txt"))
        counts[split] += 1

    make_yaml(rdir)
    print(f"  train: {counts['train']}  val: {counts['val']}")
    print(f"  data.yaml → {rdir/'data.yaml'}")
    print(f"\nNext step — train:")
    print(f"  yolo obb train data={rdir/'data.yaml'} model=yolo11n-obb.pt \\")
    print(f"       epochs=100 imgsz=1920 batch=4 name=round_0 project={PL_ROOT/'runs'}")


def setup_round_n(round_n: int, weights: Path, conf_thresh: float):
    """
    Round N: run inference on unlabelled images with round_(N-1) weights.
    High-confidence predictions (≥ conf_thresh) replace auto-labels.
    Low-confidence frames keep their auto-label.
    GT frames always use GT labels.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed — run:  pip install ultralytics")
        return

    print(f"\n[Round {round_n}] Generating pseudo-labels from {weights} …")
    model   = YOLO(str(weights))
    rdir    = PL_ROOT / f"round_{round_n}"

    # ── Run inference on all unlabelled images ────────────────────────────────
    pseudo_dir = PL_ROOT / f"pseudo_round_{round_n}"
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    replaced = kept_auto = 0
    for img_path in unl_images:
        auto_lbl = LABELS_ALL / (img_path.stem + ".txt")
        out_lbl  = pseudo_dir  / (img_path.stem + ".txt")

        results = model.predict(str(img_path), conf=0.01, verbose=False)
        r       = results[0]

        if r.obb is None or len(r.obb) == 0:
            # No predictions → keep auto-label
            shutil.copy2(auto_lbl, out_lbl)
            kept_auto += 1
            continue

        # Filter by confidence
        boxes = r.obb.xywhr.cpu().numpy()   # (N,5) cx cy w h angle
        confs = r.obb.conf.cpu().numpy()
        clss  = r.obb.cls.cpu().numpy().astype(int)
        xyxyxyxy = r.obb.xyxyxyxy.cpu().numpy()  # (N,4,2) normalised corners

        ih, iw = r.orig_shape
        high_conf = confs >= conf_thresh

        if high_conf.sum() == 0:
            shutil.copy2(auto_lbl, out_lbl)
            kept_auto += 1
            continue

        lines = []
        for i in np.where(high_conf)[0]:
            pts = xyxyxyxy[i]                  # (4,2) in pixel coords
            yolo = " ".join(f"{x/iw:.6f} {y/ih:.6f}" for x, y in pts)
            lines.append(f"{clss[i]} {yolo}")

        out_lbl.write_text("\n".join(lines))
        replaced += 1

    print(f"  Pseudo-labels: {replaced} from model  |  {kept_auto} kept auto-label")

    # ── Build round_n dataset: GT + pseudo ───────────────────────────────────
    import numpy as np
    random.seed(round_n)
    shuffled = list(gt_images); random.shuffle(shuffled)
    n_val   = max(1, int(len(shuffled) * VAL_RATIO))
    val_set = set(p.stem for p in shuffled[:n_val])

    counts = {"train": 0, "val": 0}

    for img in gt_images:
        split = "val" if img.stem in val_set else "train"
        symlink(img,                              rdir / "images" / split / img.name)
        symlink(LABELS_ALL / (img.stem + ".txt"), rdir / "labels" / split / (img.stem + ".txt"))
        counts[split] += 1

    for img in unl_images:
        symlink(img,                               rdir / "images" / "train" / img.name)
        symlink(pseudo_dir / (img.stem + ".txt"),  rdir / "labels" / "train" / (img.stem + ".txt"))
        counts["train"] += 1

    make_yaml(rdir)
    print(f"  train: {counts['train']}  val: {counts['val']}")
    print(f"\nNext step — train:")
    print(f"  yolo obb train data={rdir/'data.yaml'} model={weights} \\")
    print(f"       epochs=50 imgsz=1920 batch=4 name=round_{round_n} project={PL_ROOT/'runs'}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--setup",   type=int, required=True,
                        help="Round number to set up (0 = GT only, 1+ = pseudo)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Path to .pt weights from previous round (required for round ≥ 1)")
    parser.add_argument("--conf",    type=float, default=0.35,
                        help="Confidence threshold for keeping model prediction (default 0.35)")
    args = parser.parse_args()

    PL_ROOT.mkdir(exist_ok=True)

    if args.setup == 0:
        setup_round_0()
    else:
        if not args.weights or not args.weights.exists():
            print(f"--weights required for round {args.setup}")
        else:
            setup_round_n(args.setup, args.weights, args.conf)
