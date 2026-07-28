"""
Belt map — OVGU AMS calibration tool.

Turns per-camera calibrations into one **top-down metric map of the conveyor**:
a single canvas in belt millimetres that every camera can be rectified into and
every detection projected onto.

This is what the calibration is *for*. Once each camera has `K`, `D`, `R`, `t`
against the belt plane, a bag seen at pixel (u, v) in one camera and pixel
(u', v') in another resolve to the same (X, Y) in millimetres — so cross-camera
association stops being a heuristic about class colour and left-to-right order
and becomes a distance comparison.

What this module produces:

- **`BeltFrame`** — the map's extent and resolution: which millimetres are on
  the canvas, and how many millimetres a map pixel covers.
- **`rectify`** — one camera's view warped to top-down, as that camera would
  see the belt from directly above.
- **`footprint`** — the belt region a camera actually covers, as a polygon in
  millimetres. This is what answers the open question of whether the cameras
  overlap: it is measured, not assumed.
- **`mosaic`** — all cameras composited into one top-down image.
- **`coverage_report`** — per-camera area, pairwise overlap, and belt regions no
  camera sees.

A caution the API deliberately makes visible: rectification is only meaningful
**on the belt plane**. A bag has height, so its top surface sits above Z = 0 and
will be projected slightly outward from the camera — the further from that
camera's nadir, the larger the error. For association on a flat conveyor this is
usually acceptable, but it is a real bias, not noise, and `parallax_error_mm`
quantifies it rather than leaving it to be discovered later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

import extrinsics as extr


@dataclass
class BeltFrame:
    """Extent and resolution of the top-down map, in belt millimetres."""
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    mm_per_px: float = 2.0

    @property
    def width_mm(self) -> float: return self.x_max_mm - self.x_min_mm
    @property
    def height_mm(self) -> float: return self.y_max_mm - self.y_min_mm
    @property
    def width_px(self) -> int: return max(1, int(round(self.width_mm / self.mm_per_px)))
    @property
    def height_px(self) -> int: return max(1, int(round(self.height_mm / self.mm_per_px)))

    def mm_to_px(self, pts_mm: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_mm, np.float64).reshape(-1, 2)
        return np.column_stack([(p[:, 0] - self.x_min_mm) / self.mm_per_px,
                                (p[:, 1] - self.y_min_mm) / self.mm_per_px])

    def px_to_mm(self, pts_px: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_px, np.float64).reshape(-1, 2)
        return np.column_stack([p[:, 0] * self.mm_per_px + self.x_min_mm,
                                p[:, 1] * self.mm_per_px + self.y_min_mm])

    def to_dict(self) -> dict:
        return {"x_min_mm": self.x_min_mm, "x_max_mm": self.x_max_mm,
                "y_min_mm": self.y_min_mm, "y_max_mm": self.y_max_mm,
                "mm_per_px": self.mm_per_px,
                "size_px": [self.width_px, self.height_px]}

    @classmethod
    def from_belt(cls, width_mm: float, length_mm: float,
                  mm_per_px: float = 2.0, margin_mm: float = 0.0,
                  x0_mm: float = 0.0, y0_mm: float = 0.0) -> "BeltFrame":
        """Preferred constructor: the map is the BELT, at known dimensions.

        `auto_frame` sizes the canvas to everything the cameras happen to see —
        which on a tilted view includes floor, framing and machinery, and can be
        several times the belt area. That wastes most of the map and makes the
        region of interest small and hard to read. Since the belt's dimensions
        are a measurable, known quantity, state them.
        """
        return cls(x0_mm - margin_mm, x0_mm + width_mm + margin_mm,
                   y0_mm - margin_mm, y0_mm + length_mm + margin_mm, mm_per_px)


@dataclass
class CameraOnBelt:
    """One calibrated camera, ready to be placed on the map."""
    name: str
    K: np.ndarray
    D: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    image_size: tuple[int, int]        # (w, h)
    H_image_to_belt: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.H_image_to_belt is None:
            H = extr.homography_from_pose(self.K, self.rvec, self.tvec)
            self.H_image_to_belt = np.linalg.inv(H)

    @property
    def position_mm(self) -> np.ndarray:
        R, _ = cv2.Rodrigues(self.rvec)
        return (-R.T @ self.tvec).ravel()


def footprint(cam: CameraOnBelt, samples_per_edge: int = 24) -> np.ndarray:
    """The belt region this camera sees, as a polygon in millimetres.

    Samples along the image border rather than using only the four corners:
    lens distortion bends straight image edges into curves on the belt plane, so
    a four-point quad would misstate the covered area — outward on some edges,
    inward on others.
    """
    w, h = cam.image_size
    t = np.linspace(0, 1, samples_per_edge, endpoint=False)
    top = np.column_stack([t * w, np.zeros_like(t)])
    right = np.column_stack([np.full_like(t, w - 1), t * h])
    bottom = np.column_stack([(1 - t) * w, np.full_like(t, h - 1)])
    left = np.column_stack([np.zeros_like(t), (1 - t) * h])
    border = np.vstack([top, right, bottom, left]).astype(np.float32)
    return extr.image_to_belt(border, cam.K, cam.D, cam.H_image_to_belt)


def auto_frame(cams: list[CameraOnBelt], margin_mm: float = 50.0,
               mm_per_px: float = 2.0, max_px: int = 4000) -> BeltFrame:
    """A map extent that contains every camera's footprint.

    Guards against a degenerate camera: one solved with its optical axis nearly
    parallel to the belt projects to enormous or infinite coordinates, which
    would blow the canvas up to gigabytes. Such footprints are clipped to a
    robust percentile rather than allowed to define the map.
    """
    pts = np.vstack([footprint(c) for c in cams])
    finite = pts[np.isfinite(pts).all(axis=1)]
    if len(finite) == 0:
        raise ValueError("no finite footprint points — check the calibrations")

    lo = np.percentile(finite, 1, axis=0)
    hi = np.percentile(finite, 99, axis=0)
    span = np.maximum(hi - lo, 1.0)
    # Keep a generous band around the robust core, but not the wild tail.
    x_min, y_min = lo - 0.25 * span - margin_mm
    x_max, y_max = hi + 0.25 * span + margin_mm

    frame = BeltFrame(float(x_min), float(x_max), float(y_min), float(y_max), mm_per_px)
    biggest = max(frame.width_px, frame.height_px)
    if biggest > max_px:
        frame.mm_per_px = mm_per_px * biggest / max_px
    return frame


def rectify(image: np.ndarray, cam: CameraOnBelt, frame: BeltFrame,
            border_value=0) -> tuple[np.ndarray, np.ndarray]:
    """Warp one camera view to top-down. Returns (rectified BGR, valid mask).

    Maps backwards from the destination — every map pixel asks which source
    pixel it came from — so the output has no holes. `projectPoints` is used
    rather than the homography alone so lens distortion is included; the
    homography describes only the ideal pinhole part.
    """
    W, H = frame.width_px, frame.height_px
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    map_px = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    mm = frame.px_to_mm(map_px)

    obj = np.column_stack([mm, np.zeros(len(mm))]).astype(np.float32).reshape(-1, 1, 3)
    img_pts, _ = cv2.projectPoints(obj, cam.rvec, cam.tvec, cam.K, cam.D)
    img_pts = img_pts.reshape(-1, 2)

    mx = img_pts[:, 0].reshape(H, W).astype(np.float32)
    my = img_pts[:, 1].reshape(H, W).astype(np.float32)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    out = cv2.remap(image, mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(border_value,) * 3)

    w, h = cam.image_size
    valid = ((mx >= 0) & (mx < w) & (my >= 0) & (my < h)
             & np.isfinite(mx) & np.isfinite(my)).astype(np.uint8) * 255
    return out, valid


def mosaic(images: dict[str, np.ndarray], cams: list[CameraOnBelt],
           frame: BeltFrame) -> tuple[np.ndarray, dict]:
    """Composite every camera's rectified view into one top-down image.

    Where views overlap the mean is taken. A feathered blend would look nicer,
    but a hard mean makes misalignment *visible* as ghosting — which is exactly
    what you want to see while checking a calibration, rather than hidden.
    """
    W, H = frame.width_px, frame.height_px
    acc = np.zeros((H, W, 3), np.float32)
    cnt = np.zeros((H, W), np.float32)
    per_cam = {}

    for cam in cams:
        img = images.get(cam.name)
        if img is None:
            continue
        rect, valid = rectify(img, cam, frame)
        m = valid > 0
        acc[m] += rect[m].astype(np.float32)
        cnt[m] += 1
        per_cam[cam.name] = int(m.sum())

    out = np.zeros((H, W, 3), np.uint8)
    nz = cnt > 0
    out[nz] = (acc[nz] / cnt[nz, None]).astype(np.uint8)
    return out, {"pixels_per_camera": per_cam,
                 "overlap_px": int((cnt > 1).sum()),
                 "covered_px": int(nz.sum()),
                 "map_px": int(W * H)}


def _poly_px(cam: CameraOnBelt, frame: BeltFrame) -> np.ndarray:
    fp = footprint(cam)
    fp = fp[np.isfinite(fp).all(axis=1)]
    return frame.mm_to_px(fp).astype(np.int32)


def coverage_report(cams: list[CameraOnBelt], frame: BeltFrame) -> dict:
    """Per-camera area, pairwise overlap, and unseen belt — all in mm^2.

    This is the measurement that settles whether the cameras share a view.
    Nothing in the design assumes they do; this simply reports what is true.
    """
    W, H = frame.width_px, frame.height_px
    px_area_mm2 = frame.mm_per_px ** 2
    masks = {}
    for cam in cams:
        m = np.zeros((H, W), np.uint8)
        poly = _poly_px(cam, frame)
        if len(poly) >= 3:
            cv2.fillPoly(m, [poly], 1)
        masks[cam.name] = m

    per_cam = {n: float(m.sum() * px_area_mm2) for n, m in masks.items()}
    names = list(masks)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = float((masks[a] & masks[b]).sum() * px_area_mm2)
            smaller = min(per_cam[a], per_cam[b]) or 1.0
            pairs[f"{a}|{b}"] = {"overlap_mm2": inter,
                                 "fraction_of_smaller": inter / smaller}

    union = np.zeros((H, W), np.uint8)
    for m in masks.values():
        union |= m
    return {
        "per_camera_mm2": per_cam,
        "pairwise": pairs,
        "covered_mm2": float(union.sum() * px_area_mm2),
        "map_area_mm2": float(W * H * px_area_mm2),
        "any_overlap": any(v["overlap_mm2"] > 0 for v in pairs.values()),
    }


def draw_overlay(base: np.ndarray, cams: list[CameraOnBelt], frame: BeltFrame,
                 grid_mm: float = 100.0) -> np.ndarray:
    """Annotate the map: metric grid, camera footprints, camera positions."""
    out = base.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # Metric grid — makes the map readable as millimetres, not just pixels.
    g = (80, 80, 80)
    x = np.ceil(frame.x_min_mm / grid_mm) * grid_mm
    while x <= frame.x_max_mm:
        px = int((x - frame.x_min_mm) / frame.mm_per_px)
        cv2.line(out, (px, 0), (px, out.shape[0]), g, 1)
        cv2.putText(out, f"{x:.0f}", (px + 3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, g, 1)
        x += grid_mm
    y = np.ceil(frame.y_min_mm / grid_mm) * grid_mm
    while y <= frame.y_max_mm:
        py = int((y - frame.y_min_mm) / frame.mm_per_px)
        cv2.line(out, (0, py), (out.shape[1], py), g, 1)
        cv2.putText(out, f"{y:.0f}", (3, py - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, g, 1)
        y += grid_mm

    colours = [(90, 220, 110), (235, 160, 60), (90, 140, 245),
               (220, 210, 70), (200, 110, 220)]
    for i, cam in enumerate(cams):
        c = colours[i % len(colours)]
        poly = _poly_px(cam, frame)
        if len(poly) >= 3:
            cv2.polylines(out, [poly], True, c, 2)
            cv2.putText(out, cam.name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, c, 2)
        # The camera's own position projected straight down onto the belt.
        p = frame.mm_to_px(cam.position_mm[:2].reshape(1, 2))[0]
        if np.isfinite(p).all():
            cv2.drawMarker(out, (int(p[0]), int(p[1])), c, cv2.MARKER_TRIANGLE_UP, 14, 2)
    return out


def parallax_error_mm(cam: CameraOnBelt, point_mm: np.ndarray,
                      object_height_mm: float) -> float:
    """How far an object of a given height is displaced on the map.

    The map assumes everything lies on Z = 0. A bag's visible top surface does
    not, so it projects outward from the camera's nadir by roughly
    `height * horizontal_distance / camera_height`. This is a systematic bias,
    not noise: it does not average away, and it grows with distance from the
    camera. Worth knowing before treating map positions as exact.
    """
    p = np.asarray(point_mm, float).ravel()[:2]
    cam_xy = cam.position_mm[:2]
    cam_z = abs(cam.position_mm[2])
    if cam_z <= 1e-6:
        return float("inf")
    horiz = float(np.linalg.norm(p - cam_xy))
    return float(object_height_mm * horiz / cam_z)
