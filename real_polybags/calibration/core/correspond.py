"""
Point-correspondence extrinsics — OVGU AMS calibration tool.

The board-free route to extrinsics, and the one that generalizes to any rig:
take **one frame from each camera** and **one map of the world plane**, click
a handful of points visible in both, and solve the image->world mapping
directly. No ChArUco board on the floor, no second trip to the rig, and it
works retroactively on footage already recorded.

This mirrors the workflow of the CalibrationHub tool, with two deliberate
differences that address the failure its own manual describes:

> "The algorithm that calculates the homography matrix wants to 'satisfy'
>  every mapping. That's why you should rather remove incorrect mappings than
>  leaving the incorrect ones in the json array and adding more correct ones."

That is a real property of a plain least-squares `findHomography`: one
mis-clicked point quietly bends the whole fit, and the operator is left to
guess which one. So:

1. **RANSAC by default.** Outliers are *rejected* rather than averaged in, so
   a single bad click stops being fatal.
2. **Per-point residuals are reported, in millimetres.** The operator is told
   *which* correspondence disagrees and by how much, instead of being advised
   to delete points until it looks better. Quality over quantity is good
   advice; measuring quality is better than eyeballing it.

Two levels of result, exactly as elsewhere in this tool:

- **With intrinsics** — image points are undistorted first, and the resulting
  homography is decomposed into a real `R`, `t`. Lens distortion is corrected,
  the camera's 3-D position is recovered, and the result is indistinguishable
  downstream from a board solve.
- **Without intrinsics** — a raw pixel->world homography. Perfectly usable for
  association on the plane, but distortion is uncorrected (a homography cannot
  represent it), so accuracy degrades toward the frame edges and there is no
  camera position. Recorded as such, never silently promoted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MIN_CORRESPONDENCES = 4          # a plane homography has 8 DOF -> 4 point pairs


@dataclass
class CorrespondenceResult:
    camera: str
    H_image_to_world: np.ndarray          # (3,3) undistorted px -> world mm
    image_points: np.ndarray              # (N,2) as clicked, distorted px
    world_points: np.ndarray              # (N,2) mm
    residuals_mm: np.ndarray              # (N,) per-point error after the fit
    inliers: np.ndarray                   # (N,) bool
    used_intrinsics: bool
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def H_world_to_image(self) -> np.ndarray:
        return np.linalg.inv(self.H_image_to_world)

    @property
    def rms_mm(self) -> float:
        r = self.residuals_mm[self.inliers]
        return float(np.sqrt((r ** 2).mean())) if len(r) else float("nan")

    @property
    def max_mm(self) -> float:
        r = self.residuals_mm[self.inliers]
        return float(r.max()) if len(r) else float("nan")

    @property
    def camera_position_mm(self) -> np.ndarray:
        """Camera centre in world mm — only recoverable with intrinsics.

        A homography fixes where the plane goes, not where the camera is:
        infinitely many poses produce the same plane mapping. NaN rather than
        a fabricated number, matching `beltmap.CameraOnBelt.position_mm`.
        """
        if self.rvec is None:
            return np.array([np.nan, np.nan, np.nan])
        R, _ = cv2.Rodrigues(self.rvec)
        return (-R.T @ self.tvec).ravel()


def _decompose_to_pose(H_world_to_image: np.ndarray, K: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Recover (rvec, tvec) from a world-plane->image homography.

    For a plane at Z = 0, `H ~ K [r1 r2 t]`. Inverting K gives those three
    columns up to one common scale, which is fixed by requiring the rotation
    columns to be unit length. `r3 = r1 x r2` completes the basis, and an SVD
    projects the result onto the nearest true rotation — necessary because
    clicked points are noisy and the raw columns come out slightly
    non-orthonormal.
    """
    B = np.linalg.inv(K) @ H_world_to_image
    # Scale: the two rotation columns must be unit length. Use their mean so
    # noise in either one does not dominate.
    lam = 2.0 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    B = B * lam
    r1, r2, t = B[:, 0], B[:, 1], B[:, 2]
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:                 # keep it a rotation, not a flip
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
        t = -t
    # A camera must be in front of the plane it is looking at; the overall
    # sign of H is arbitrary, so flip if the solve came out behind.
    if t[2] < 0:
        R, t = -R, -t
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t.reshape(3, 1)


def solve(image_points, world_points_mm, camera: str = "camera",
          K: np.ndarray | None = None, D: np.ndarray | None = None,
          ransac_reproj_mm: float | None = None) -> CorrespondenceResult:
    """Solve image -> world-plane from clicked point pairs.

    `ransac_reproj_mm` is the inlier threshold in **millimetres** (world
    units), not pixels — the operator knows how accurately they can identify
    a point on the floor, and does not know what that is in pixels. `None`
    picks a threshold from the spread of the points themselves. With exactly
    4 pairs RANSAC has nothing to vote with, so a plain solve is used and
    that is said out loud.
    """
    img = np.asarray(image_points, np.float64).reshape(-1, 2)
    wld = np.asarray(world_points_mm, np.float64).reshape(-1, 2)
    if len(img) != len(wld):
        raise ValueError(f"{len(img)} image points but {len(wld)} world points")
    if len(img) < MIN_CORRESPONDENCES:
        raise ValueError(
            f"need at least {MIN_CORRESPONDENCES} point pairs to fit a plane "
            f"homography (8 degrees of freedom), got {len(img)}")

    warnings: list[str] = []
    used_intrinsics = K is not None
    if used_intrinsics:
        D = np.zeros(5) if D is None else np.asarray(D, np.float64).ravel()
        # Undistort into pixel space (P=K) so the homography is fitted in the
        # same coordinates the rest of the tool undistorts into.
        src = cv2.undistortPoints(img.reshape(-1, 1, 2).astype(np.float32),
                                  K, D, P=K).reshape(-1, 2).astype(np.float64)
    else:
        src = img
        warnings.append(
            "solved without intrinsics — lens distortion is not corrected, so "
            "accuracy degrades toward the frame edges, and the camera's "
            "position cannot be recovered")

    if ransac_reproj_mm is None:
        # ~1.5% of the spread of the clicked world points: tight enough to
        # catch a genuine mis-click, loose enough not to reject honest
        # click precision.
        spread = float(np.linalg.norm(wld.max(axis=0) - wld.min(axis=0)))
        ransac_reproj_mm = max(5.0, 0.015 * spread)

    if len(img) == MIN_CORRESPONDENCES:
        H, mask = cv2.findHomography(src, wld, 0)
        warnings.append(
            "exactly 4 point pairs — the fit passes through all of them "
            "exactly, so the residuals below are 0 by construction and prove "
            "nothing. Add a 5th point to get a real error estimate")
        inliers = np.ones(len(img), bool)
    else:
        H, mask = cv2.findHomography(src, wld, cv2.RANSAC,
                                     ransacReprojThreshold=ransac_reproj_mm)
        inliers = (mask.ravel() > 0) if mask is not None else np.ones(len(img), bool)

    if H is None:
        raise ValueError(
            "could not fit a homography to these points — they may be "
            "collinear, or several may be duplicates. Spread them out across "
            "the region you care about")

    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    residuals = np.linalg.norm(proj - wld, axis=1)

    n_out = int((~inliers).sum())
    if n_out:
        worst = int(np.argmax(np.where(inliers, -np.inf, residuals)))
        warnings.append(
            f"{n_out} of {len(img)} correspondences rejected as outliers "
            f"(worst: point {worst + 1}, off by {residuals[worst]:.0f} mm) — "
            f"they were excluded from the fit, but check them: a mis-click on "
            f"either the image or the map is the usual cause")

    rvec = tvec = None
    if used_intrinsics:
        try:
            rvec, tvec = _decompose_to_pose(np.linalg.inv(H), K)
        except np.linalg.LinAlgError:
            warnings.append("homography could not be decomposed into a camera "
                            "pose; the plane mapping is still usable")

    res = CorrespondenceResult(
        camera=camera, H_image_to_world=H, image_points=img, world_points=wld,
        residuals_mm=residuals, inliers=inliers, used_intrinsics=used_intrinsics,
        rvec=rvec, tvec=tvec, warnings=warnings)

    if len(img) > MIN_CORRESPONDENCES and res.rms_mm > ransac_reproj_mm:
        res.warnings.append(
            f"RMS {res.rms_mm:.0f} mm across the inliers is high relative to "
            f"the {ransac_reproj_mm:.0f} mm tolerance — the points may be "
            f"clustered in one part of the frame, or the map scale may be off")
    if used_intrinsics and rvec is not None:
        h = abs(float(res.camera_position_mm[2]))
        if h < 50:
            res.warnings.append(
                f"camera solved to only {h:.0f} mm above the plane — "
                f"implausible; check the map's scale (mm per pixel)")
    return res


def spread_score(world_points_mm, footprint_mm=None) -> dict:
    """How well the clicked points span the area, not just how many there are.

    Four points clustered in one corner fit a homography that is excellent
    there and arbitrarily wrong elsewhere — the failure CalibrationHub's
    manual gestures at with "points outside of your ROI won't be accurate".
    Reports the covered fraction so it can be said before the solve rather
    than discovered during validation.
    """
    p = np.asarray(world_points_mm, np.float64).reshape(-1, 2)
    if len(p) < 3:
        return {"hull_mm2": 0.0, "fraction_of_roi": 0.0, "ok": False}
    hull = cv2.convexHull(p.astype(np.float32))
    area = float(cv2.contourArea(hull))
    out = {"hull_mm2": area, "fraction_of_roi": None, "ok": area > 0}
    if footprint_mm is not None and len(footprint_mm) >= 3:
        roi = cv2.contourArea(np.asarray(footprint_mm, np.float32))
        if roi > 0:
            out["fraction_of_roi"] = area / roi
            out["ok"] = area / roi > 0.25
    return out
