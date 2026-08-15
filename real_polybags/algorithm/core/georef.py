"""
Map georeference — OVGU AMS algorithm phase.

The dashboard draws polybags on the top-down conveyor map
(`calibration/results/_workspace/map.png`). To place a world-mm point on that
image it needs the same two numbers the calibration tool stored when it
georeferenced the map: `mm_per_px` and `origin_px`. This reads them out of
`meta.json` (no image decode, no cv2 — just the numbers), so the streamer can
hand them to the browser via /api/config.

Convention mirrors `worldmap.WorkspaceMap.mm_to_px`:
    map_px = origin_px + world_mm / mm_per_px
with world +X = map-image right, +Y = map-image down — the same handedness the
belt frame uses, so no sign flips.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_georef(meta_path: Path) -> dict:
    """Return {mm_per_px, origin_px:[x,y], image_size:[w,h]} from meta.json."""
    meta = json.loads(Path(meta_path).read_text())
    mm_per_px = meta.get("mm_per_px")
    origin_px = meta.get("origin_px")
    image_size = meta.get("image_size")
    if mm_per_px is None or origin_px is None:
        raise ValueError(
            f"{meta_path} is not georeferenced (missing mm_per_px / origin_px)")
    return {
        "mm_per_px": float(mm_per_px),
        "origin_px": [float(origin_px[0]), float(origin_px[1])],
        "image_size": [int(image_size[0]), int(image_size[1])] if image_size else None,
        "notes": meta.get("notes", ""),
    }
