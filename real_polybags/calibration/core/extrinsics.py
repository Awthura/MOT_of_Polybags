"""
Extrinsic calibration and belt-plane mapping — OVGU AMS calibration tool.

Once a camera's intrinsics are known, one view of the board **lying flat on the
belt** fixes that camera's pose relative to the belt. The belt is treated as the
world Z = 0 plane, so every camera ends up expressed in one shared metric frame
and cross-camera association becomes a nearest-neighbour question in
millimetres.

Two things follow from the same solve:

- `R`, `t` — the camera's true pose relative to the belt frame (the extrinsics).
- `H` — the homography mapping belt coordinates to image pixels, and `H⁻¹` for
  the reverse. Because the world plane is Z = 0, `H = K · [r1 r2 t]`: the third
  rotation column drops out, since it only ever multiplies Z.

**This does not require the cameras to see each other or share a view.** Each is
solved independently against the same belt frame. Overlap is measured as a
by-product (which cameras saw the board simultaneously), not assumed.

Ordering matters: **undistort first, then apply the homography.** A homography
is a projective map between planes and cannot absorb radial distortion; fitting
one to distorted points bakes the lens error into the plane mapping, where it
shows up as position error that grows toward the frame edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ExtrinsicResult:
    camera: str
    rvec: np.ndarray                    # (3,1) Rodrigues, board/belt -> camera
    tvec: np.ndarray                    # (3,1) mm
    H_belt_to_image: np.ndarray         # (3,3)
    H_image_to_belt: np.ndarray         # (3,3)
    reproj_error_px: float
    n_points: int
    board_origin_offset_mm: tuple[float, float] = (0.0, 0.0)
    warnings: list[str] = field(default_factory=list)

    @property
    def R(self) -> np.ndarray:
        R, _ = cv2.Rodrigues(self.rvec)
        return R

    @property
    def camera_position_mm(self) -> np.ndarray:
        """Camera centre in belt coordinates: C = -R^T * t."""
        return (-self.R.T @ self.tvec).ravel()

    @property
    def height_above_belt_mm(self) -> float:
        return float(self.camera_position_mm[2])


def solve_pose(object_points: np.ndarray, image_points: np.ndarray,
               K: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Board pose from a single view. Returns (rvec, tvec) in board units (mm)."""
    obj = np.asarray(object_points, np.float32).reshape(-1, 1, 3)
    img = np.asarray(image_points, np.float32).reshape(-1, 1, 2)
    if len(obj) < 4:
        raise ValueError(f"need >=4 point correspondences, got {len(obj)}")

    # IPPE is designed for planar targets and is both faster and better
    # conditioned than the iterative default when all points share a plane.
    planar = np.allclose(obj[:, 0, 2], obj[0, 0, 2])
    flags = cv2.SOLVEPNP_IPPE if planar and len(obj) >= 4 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=flags)
    if not ok:
        raise RuntimeError("solvePnP failed to find a pose")
    # Refine against all points; IPPE gives a good starting estimate.
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
    return rvec, tvec


def homography_from_pose(K: np.ndarray, rvec: np.ndarray,
                         tvec: np.ndarray) -> np.ndarray:
    """H mapping world plane Z=0 -> image pixels.

    For a point (X, Y, 0): x ~ K [r1 r2 r3 | t] (X,Y,0,1)^T = K [r1 r2 t] (X,Y,1)^T
    so the third rotation column is simply dropped.
    """
    R, _ = cv2.Rodrigues(rvec)
    H = K @ np.column_stack([R[:, 0], R[:, 1], tvec.ravel()])
    return H / H[2, 2]


def calibrate_extrinsics(object_points: np.ndarray, image_points: np.ndarray,
                         K: np.ndarray, D: np.ndarray, camera: str = "camera",
                         origin_offset_mm: tuple[float, float] = (0.0, 0.0)
                         ) -> ExtrinsicResult:
    """Full extrinsic solve for one camera from a board lying on the belt.

    `origin_offset_mm` shifts the board's own origin to the shared belt origin,
    so cameras that saw the board at different measured positions along the belt
    still land in one common frame.
    """
    obj = np.asarray(object_points, np.float32).reshape(-1, 1, 3).copy()
    obj[:, 0, 0] += origin_offset_mm[0]
    obj[:, 0, 1] += origin_offset_mm[1]
    img = np.asarray(image_points, np.float32).reshape(-1, 1, 2)

    rvec, tvec = solve_pose(obj, img, K, D)

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err = float(cv2.norm(img, proj, cv2.NORM_L2) / np.sqrt(len(proj)))

    H = homography_from_pose(K, rvec, tvec)
    res = ExtrinsicResult(
        camera=camera, rvec=rvec, tvec=tvec,
        H_belt_to_image=H, H_image_to_belt=np.linalg.inv(H),
        reproj_error_px=err, n_points=len(obj),
        board_origin_offset_mm=tuple(origin_offset_mm),
    )

    if err > 2.0:
        res.warnings.append(
            f"extrinsic reprojection error {err:.2f} px — board may not be flat "
            f"on the belt, or the intrinsics are wrong")
    if res.height_above_belt_mm <= 0:
        res.warnings.append(
            f"camera solved to {res.height_above_belt_mm:.0f} mm, i.e. below the "
            f"belt plane — the board's coordinate frame is probably flipped")
    return res


# ── Plane transforms ─────────────────────────────────────────────────────────

def image_to_belt(points_px: np.ndarray, K: np.ndarray, D: np.ndarray,
                  H_image_to_belt: np.ndarray) -> np.ndarray:
    """Pixel coordinates -> belt coordinates (mm).

    Undistorts first: a homography cannot represent radial distortion, so
    skipping this leaves lens error in the result, worst at the frame edges.
    """
    pts = np.asarray(points_px, np.float32).reshape(-1, 1, 2)
    # P=K maps the undistorted normalised points back to pixels, which is the
    # space the homography was fitted in.
    undist = cv2.undistortPoints(pts, K, D, P=K)
    belt = cv2.perspectiveTransform(undist, H_image_to_belt)
    return belt.reshape(-1, 2)


def belt_to_image(points_mm: np.ndarray, K: np.ndarray, D: np.ndarray,
                  rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Belt coordinates (mm) -> pixel coordinates, distortion included.

    Uses projectPoints rather than the homography so the result lands in the
    real (distorted) image, which is where detections actually live.
    """
    pts = np.asarray(points_mm, np.float32).reshape(-1, 2)
    obj = np.column_stack([pts, np.zeros(len(pts), np.float32)]).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    return proj.reshape(-1, 2)


def mm_per_pixel(H_image_to_belt: np.ndarray, K: np.ndarray, D: np.ndarray,
                 at_px: tuple[float, float], delta: float = 1.0) -> float:
    """Local scale in mm per pixel at a given image location.

    Not a constant: an oblique view compresses distant parts of the belt, so a
    single global figure would be wrong almost everywhere. Reported per location.
    """
    x, y = at_px
    pts = np.array([[x, y], [x + delta, y], [x, y + delta]], np.float32)
    belt = image_to_belt(pts, K, D, H_image_to_belt)
    dx = np.linalg.norm(belt[1] - belt[0]) / delta
    dy = np.linalg.norm(belt[2] - belt[0]) / delta
    return float((dx + dy) / 2)
