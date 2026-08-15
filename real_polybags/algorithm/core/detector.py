"""
YOLO detector wrapper — OVGU AMS algorithm phase.

Wraps ultralytics YOLO and normalizes its two output shapes into one `Box`
list, so the rest of the pipeline never branches on model type:

- an **OBB** model (e.g. yolo11n-obb) yields oriented boxes -> `Box(kind='obb')`
- a **detect** model yields axis-aligned boxes -> `Box(kind='aabb')`

The model path is config-driven: it runs today against existing weights and
swaps to the final trained polybag model by editing config.yaml alone.
"""

from __future__ import annotations

import cv2
import numpy as np

from footpoint import Box


class Detector:
    def __init__(self, model_path: str, imgsz: int = 960, conf: float = 0.25,
                 iou: float = 0.5, device: str = ""):
        from ultralytics import YOLO   # heavy import, deferred to construction
        self.model = YOLO(str(model_path))
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device or None
        self.task = getattr(self.model, "task", "detect")

    def detect(self, frame_bgr: np.ndarray) -> list[Box]:
        res = self.model.predict(
            frame_bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False)[0]
        boxes: list[Box] = []

        obb = getattr(res, "obb", None)
        if obb is not None and obb.xyxyxyxy is not None and len(obb) > 0:
            corners = obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
            confs = obb.conf.cpu().numpy()
            clss = obb.cls.cpu().numpy()
            for c, cf, cl in zip(corners, confs, clss):
                boxes.append(Box(kind="obb", conf=float(cf), cls=int(cl),
                                 corners=c.astype(float)))
            return boxes

        b = getattr(res, "boxes", None)
        if b is not None and len(b) > 0:
            xywh = b.xywh.cpu().numpy()
            confs = b.conf.cpu().numpy()
            clss = b.cls.cpu().numpy()
            for (cx, cy, w, h), cf, cl in zip(xywh, confs, clss):
                boxes.append(Box(kind="aabb", conf=float(cf), cls=int(cl),
                                 xywh=(float(cx), float(cy), float(w), float(h))))
        return boxes


# ── Overlay drawing for the MJPEG preview ────────────────────────────────────

def draw_detections(frame_bgr: np.ndarray, boxes: list[Box],
                    color=(60, 220, 60), ref: str = "center") -> np.ndarray:
    """Draw boxes + the reference point on a copy of the frame for the feed.

    `ref` selects which pixel is marked (and projected downstream): "center"
    (bbox centre) or "foot" (bbox bottom-centre).
    """
    out = frame_bgr
    for b in boxes:
        if b.kind == "obb" and b.corners is not None:
            pts = np.asarray(b.corners, np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, color, 2)
        elif b.kind == "aabb" and b.xywh is not None:
            cx, cy, w, h = b.xywh
            p1 = (int(cx - w / 2), int(cy - h / 2))
            p2 = (int(cx + w / 2), int(cy + h / 2))
            cv2.rectangle(out, p1, p2, color, 2)
        rx, ry = b.center_px() if ref == "center" else b.foot_px()
        cv2.circle(out, (int(rx), int(ry)), 5, (0, 200, 255), -1)  # reference = orange
        cv2.circle(out, (int(rx), int(ry)), 5, (0, 0, 0), 1)
    return out
