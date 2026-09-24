"""
Detections and their ground-contact point — OVGU AMS algorithm phase.

A detection is a box in image pixels; to place a polybag on the belt we need the
one pixel where it *touches* the belt plane, because that is the only point the
image->belt homography maps correctly (the homography is a plane-to-plane map,
so anything above the plane — the top of a bag — projects to the wrong spot).

The ground-contact point is taken as the **bottom of the box in the image**:

- axis-aligned box  -> bottom-centre, ``(cx, cy + h/2)``
- oriented box(OBB) -> midpoint of the box's two lowest (largest-y) vertices

Both live here so the smoke test and the streamer share one definition, and so
this module stays free of the heavy ultralytics import (``detector.py`` builds
these ``Box`` objects from YOLO output and depends on this, not the reverse).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Box:
    """One detection, either axis-aligned (``kind='aabb'``) or oriented (``'obb'``).

    - ``aabb``: ``xywh`` is (cx, cy, w, h) in image pixels.
    - ``obb`` : ``corners`` is a (4, 2) array of image-pixel vertices.
    """
    kind: str                       # 'aabb' | 'obb'
    conf: float
    cls: int = 0
    xywh: tuple[float, float, float, float] | None = None
    corners: np.ndarray | None = None
    mask: np.ndarray | None = None  # (K,2) segmentation polygon in image px, if any

    def foot_px(self) -> tuple[float, float]:
        """The (x, y) image pixel where this box meets the belt plane."""
        if self.kind == "aabb":
            if self.xywh is None:
                raise ValueError("aabb Box has no xywh")
            cx, cy, w, h = self.xywh
            return (float(cx), float(cy + h / 2.0))
        if self.kind == "obb":
            if self.corners is None:
                raise ValueError("obb Box has no corners")
            c = np.asarray(self.corners, float).reshape(-1, 2)
            # Two vertices with the largest image y are the lowest edge; their
            # midpoint is the bag's contact line with the belt.
            lowest = c[np.argsort(c[:, 1])[-2:]]
            return (float(lowest[:, 0].mean()), float(lowest[:, 1].mean()))
        raise ValueError(f"unknown box kind {self.kind!r}")

    def center_px(self) -> tuple[float, float]:
        """Box centre in image pixels — for drawing the label, not for mapping."""
        if self.kind == "aabb":
            cx, cy, _, _ = self.xywh
            return (float(cx), float(cy))
        c = np.asarray(self.corners, float).reshape(-1, 2)
        return (float(c[:, 0].mean()), float(c[:, 1].mean()))
