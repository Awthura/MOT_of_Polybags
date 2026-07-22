#!/bin/bash
# Collect all overlay PNGs into a single flat review folder.
# Files are named {subdir}_{stem}.png to avoid collisions.

SRC="/Users/awthura/OVGU/AMS/real_polybags/real_data_labels/overlays"
DST="/Users/awthura/OVGU/AMS/real_polybags/real_data_labels/review"

mkdir -p "$DST"

count=0
for f in "$SRC"/*/rgb_frame_*.png; do
    subdir=$(basename "$(dirname "$f")")
    stem=$(basename "$f" .png)
    ln -sf "$f" "$DST/${subdir}_${stem}.png"
    ((count++))
done

echo "Linked $count overlay images → $DST"
