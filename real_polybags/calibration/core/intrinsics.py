"""
Intrinsic calibration — OVGU AMS calibration tool.

Recovers the camera matrix `K` and distortion coefficients `D` from multiple
views of a ChArUco board (or a plain checkerboard).

All lengths are in **millimetres**, matching the printed board spec, so nothing
downstream has to guess at a unit conversion.

## Why this reports pose coverage, not just an error number

The usual way intrinsic calibration fails is silent. Take twenty shots of the
board held flat-on at the same distance and OpenCV returns a low reprojection
error and a confidently wrong `K` — focal length and distance are ambiguous
unless the board is *tilted*, and distortion coefficients are unconstrained
unless the board reaches the frame edges where distortion actually lives.

A low RMS therefore does not mean a good calibration; it can simply mean the
poses were too similar to disagree with each other. So this module reports:

- **frame coverage** — how much of the image the board has actually visited,
- **tilt spread** — the range of board orientations seen,
- **per-view error** — so a single bad shot is identifiable rather than averaged
  into a healthy-looking mean,

and turns those into an explicit verdict rather than leaving it to judgement.

## OpenCV version note

`cv2.aruco.calibrateCameraCharuco` was removed in OpenCV 4.7+ (absent in 4.13,
which this repo uses). The current path is:

    CharucoDetector.detectBoard() -> board.matchImagePoints() -> cv2.calibrateCamera()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

# A view contributing fewer than this many points is more likely to hurt than
# help — few points means a poorly-constrained pose, which drags the shared
# intrinsics with it.
MIN_POINTS_PER_VIEW = 8
MIN_VIEWS = 5


@dataclass
class Detection:
    """One usable view of the board."""
    source: str
    image_size: tuple[int, int]        # (w, h)
    object_points: np.ndarray          # (N,1,3) float32, mm, board frame
    image_points: np.ndarray           # (N,1,2) float32, px
    n_points: int


@dataclass
class IntrinsicResult:
    camera: str
    image_size: tuple[int, int]
    K: np.ndarray
    D: np.ndarray
    rms: float                          # overall RMS reprojection error, px
    per_view_error: list[float]         # px, same order as views
    view_sources: list[str]
    rvecs: list[np.ndarray] = field(default_factory=list)
    tvecs: list[np.ndarray] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def fx(self) -> float: return float(self.K[0, 0])
    @property
    def fy(self) -> float: return float(self.K[1, 1])
    @property
    def cx(self) -> float: return float(self.K[0, 2])
    @property
    def cy(self) -> float: return float(self.K[1, 2])


# ── Detection ────────────────────────────────────────────────────────────────

def detect_charuco(image: np.ndarray, spec, source: str = "") -> Detection | None:
    """Detect a ChArUco board in one image. Returns None if unusable."""
    board = spec.board()
    detector = aruco.CharucoDetector(board)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or len(charuco_corners) < MIN_POINTS_PER_VIEW:
        return None

    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_pts is None or len(obj_pts) < MIN_POINTS_PER_VIEW:
        return None

    h, w = gray.shape[:2]
    return Detection(source=source, image_size=(w, h),
                     object_points=np.asarray(obj_pts, np.float32),
                     image_points=np.asarray(img_pts, np.float32),
                     n_points=len(obj_pts))


def detect_checkerboard(image: np.ndarray, inner_corners: tuple[int, int],
                        square_mm: float, source: str = "") -> Detection | None:
    """Detect a plain checkerboard. `inner_corners` = (cols, rows) of INTERIOR
    corners — a 7x8-square board has 6x7 interior corners."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # findChessboardCornersSB is markedly more robust than the legacy detector
    # and needs no separate cornerSubPix refinement.
    ok, corners = cv2.findChessboardCornersSB(
        gray, inner_corners, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok or corners is None:
        return None

    cols, rows = inner_corners
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    h, w = gray.shape[:2]
    return Detection(source=source, image_size=(w, h),
                     object_points=obj.reshape(-1, 1, 3),
                     image_points=np.asarray(corners, np.float32).reshape(-1, 1, 2),
                     n_points=len(obj))


# ── Coverage analysis ────────────────────────────────────────────────────────

def _tilt_deg(rvec: np.ndarray) -> float:
    """Angle between the board normal and the camera's optical axis, degrees.

    0 means the board is exactly face-on to the camera — the degenerate case
    that makes focal length and distance indistinguishable.
    """
    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2]                       # board +Z in camera coordinates
    cos = abs(float(normal[2])) / (np.linalg.norm(normal) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def analyse_coverage(dets: list[Detection], image_size: tuple[int, int],
                     rvecs: list[np.ndarray], grid: int = 4) -> dict:
    """How well the views span the frame and the space of orientations."""
    w, h = image_size
    occupied = np.zeros((grid, grid), bool)
    for d in dets:
        for p in d.image_points.reshape(-1, 2):
            gx = min(grid - 1, max(0, int(p[0] / w * grid)))
            gy = min(grid - 1, max(0, int(p[1] / h * grid)))
            occupied[gy, gx] = True

    tilts = [_tilt_deg(r) for r in rvecs] if rvecs else []
    return {
        "frame_cells_seen": int(occupied.sum()),
        "frame_cells_total": grid * grid,
        "frame_fraction": float(occupied.sum()) / (grid * grid),
        "grid": occupied.astype(int).tolist(),
        "tilt_deg_min": float(min(tilts)) if tilts else None,
        "tilt_deg_max": float(max(tilts)) if tilts else None,
        "tilt_deg_mean": float(np.mean(tilts)) if tilts else None,
        "n_views_tilted_over_15deg": int(sum(t > 15 for t in tilts)) if tilts else 0,
    }


# ── Calibration ──────────────────────────────────────────────────────────────

def calibrate(dets: list[Detection], camera: str = "camera",
              fix_aspect: bool = False) -> IntrinsicResult:
    """Solve for K and D from a set of detections."""
    if len(dets) < MIN_VIEWS:
        raise ValueError(f"need at least {MIN_VIEWS} usable views, got {len(dets)}")
    sizes = {d.image_size for d in dets}
    if len(sizes) != 1:
        raise ValueError(f"views have differing image sizes: {sizes}")
    image_size = dets[0].image_size

    obj = [d.object_points for d in dets]
    img = [d.image_points for d in dets]

    flags = cv2.CALIB_FIX_ASPECT_RATIO if fix_aspect else 0
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, image_size, None, None, flags=flags)

    # Per-view error, so a single bad shot is visible instead of averaged away.
    per_view = []
    for i, d in enumerate(dets):
        proj, _ = cv2.projectPoints(d.object_points, rvecs[i], tvecs[i], K, D)
        err = cv2.norm(d.image_points, proj, cv2.NORM_L2) / np.sqrt(len(proj))
        per_view.append(float(err))

    coverage = analyse_coverage(dets, image_size, rvecs)
    result = IntrinsicResult(
        camera=camera, image_size=image_size, K=K, D=D, rms=float(rms),
        per_view_error=per_view, view_sources=[d.source for d in dets],
        rvecs=[np.asarray(r) for r in rvecs], tvecs=[np.asarray(t) for t in tvecs],
        coverage=coverage,
    )
    result.warnings = assess(result)
    return result


def assess(res: IntrinsicResult) -> list[str]:
    """Problems worth surfacing. A low RMS alone does not mean a good result."""
    w = []
    if res.rms > 1.0:
        w.append(f"high reprojection error ({res.rms:.2f} px) — check for blurred "
                 f"or mis-detected views")
    if res.coverage.get("frame_fraction", 1.0) < 0.6:
        seen = res.coverage.get("frame_cells_seen")
        tot = res.coverage.get("frame_cells_total")
        w.append(f"board only reached {seen}/{tot} of the frame — distortion "
                 f"coefficients are poorly constrained near the edges")
    if (res.coverage.get("tilt_deg_max") or 0) < 15:
        w.append("no strongly tilted views — focal length and distance are nearly "
                 "degenerate, so fx/fy may be confidently wrong despite a low RMS")
    if res.coverage.get("n_views_tilted_over_15deg", 0) < 3:
        w.append("fewer than 3 tilted views — add more off-axis poses")
    if len(res.per_view_error) < 10:
        w.append(f"only {len(res.per_view_error)} views — 15-25 is a healthier set")
    worst = max(res.per_view_error) if res.per_view_error else 0
    if worst > 3 * max(res.rms, 1e-6):
        i = int(np.argmax(res.per_view_error))
        w.append(f"view {i} ({res.view_sources[i]}) has {worst:.2f} px error, far "
                 f"above the {res.rms:.2f} px mean — consider removing it")
    return w


def format_report(res: IntrinsicResult, expected_fx: float | None = None) -> str:
    """Human-readable summary, in the style of measure_sync.py."""
    L = []
    L.append("=" * 78)
    L.append(f"INTRINSICS — {res.camera}   {res.image_size[0]}x{res.image_size[1]}")
    L.append("=" * 78)
    L.append(f"  fx = {res.fx:9.2f}    fy = {res.fy:9.2f}")
    L.append(f"  cx = {res.cx:9.2f}    cy = {res.cy:9.2f}"
             f"   (image centre {res.image_size[0]/2:.1f}, {res.image_size[1]/2:.1f})")
    L.append(f"  D  = {np.asarray(res.D).ravel().round(5).tolist()}")
    L.append("")
    L.append(f"  views {len(res.per_view_error)}   RMS {res.rms:.3f} px   "
             f"worst view {max(res.per_view_error):.3f} px")
    cov = res.coverage
    L.append(f"  frame coverage {cov.get('frame_cells_seen')}/{cov.get('frame_cells_total')} cells"
             f"   tilt {cov.get('tilt_deg_min', 0):.0f}-{cov.get('tilt_deg_max', 0):.0f} deg")

    if expected_fx:
        dev = abs(res.fx - expected_fx) / expected_fx * 100
        verdict = "OK" if dev < 10 else "OUT OF BAND"
        L.append(f"  expected fx ~{expected_fx:.0f} (lens/sensor)  ->  "
                 f"{dev:.1f}% deviation  [{verdict}]")

    if res.warnings:
        L.append("")
        L.append("  WARNINGS:")
        for msg in res.warnings:
            L.append(f"    - {msg}")
    else:
        L.append("")
        L.append("  No issues detected — coverage and tilt look adequate.")
    L.append("=" * 78)
    return "\n".join(L)


# Distortion models this tool's undistortion pipeline can represent. Both
# `extrinsics.image_to_belt` and `belt_to_image` go through cv2.undistortPoints
# / cv2.projectPoints, which assume the Brown-Conrady radial-tangential model —
# a fisheye camera reporting Kannala-Brandt or F-Theta coefficients through
# that path would produce a plausible-looking but wrong undistortion, with
# nothing downstream able to tell the difference from a correct one.
COMPATIBLE_FACTORY_MODELS = ("brown conrady", "none")


def from_factory_intrinsics(camera: str, factory: dict) -> IntrinsicResult:
    """Adopt a camera's own factory calibration instead of fitting one.

    The RealSense D435 reports `fx, fy, ppx, ppy` and a distortion model
    directly from the sensor — the only camera on this rig with independent
    ground truth. This skips the board-capture flow entirely rather than
    merely comparing against it (see `intrinsics.assess` / the "vs known
    reference" panel for the comparison path, which is still worth doing once
    to validate the tool itself).

    There is nothing to report an RMS or frame coverage for — this was never
    fitted here — so those fields are left empty rather than filled with a
    zero that would read as a suspiciously perfect result.
    """
    model = str(factory.get("model", "")).strip()
    if not any(m in model.lower() for m in COMPATIBLE_FACTORY_MODELS):
        raise ValueError(
            f"distortion model '{model}' is not Brown-Conrady compatible — "
            f"this tool's undistortion (cv2.undistortPoints / projectPoints) "
            f"assumes that model, and applying it to a different one would "
            f"silently produce a wrong result rather than an obvious failure")

    w, h = int(factory["width"]), int(factory["height"])
    K = np.array([[factory["fx"], 0.0, factory["cx"]],
                  [0.0, factory["fy"], factory["cy"]],
                  [0.0, 0.0, 1.0]])
    # RealSense's Brown-Conrady coeffs are ordered (k1, k2, p1, p2, k3),
    # matching OpenCV's convention directly — no reordering needed.
    coeffs = factory.get("coeffs") or [0.0] * 5
    D = np.array(coeffs, dtype=float)

    return IntrinsicResult(
        camera=camera, image_size=(w, h), K=K, D=D, rms=float("nan"),
        per_view_error=[], view_sources=[], coverage={},
        warnings=[f"factory calibration ({model or 'unknown model'}) — not "
                  f"fitted by this tool, so there is no reprojection error or "
                  f"frame coverage to report"])


def format_factory_report(res: IntrinsicResult, model: str) -> str:
    """Human-readable summary for an adopted factory calibration.

    Deliberately separate from `format_report`: that function assumes a
    non-empty `per_view_error` (it calls `max()` on it) and reports coverage
    and per-view statistics that simply do not exist for a number the sensor
    reported rather than one this tool fitted.
    """
    D = np.asarray(res.D).ravel().round(5).tolist()
    L = ["=" * 78,
         f"FACTORY INTRINSICS — {res.camera}   "
         f"{res.image_size[0]}x{res.image_size[1]}   ({model})",
         "=" * 78,
         f"  fx = {res.fx:9.2f}    fy = {res.fy:9.2f}",
         f"  cx = {res.cx:9.2f}    cy = {res.cy:9.2f}",
         f"  D  = {D}",
         "",
         "  Reported by the sensor's own factory calibration, not fitted from",
         "  board captures here — there is no reprojection error or frame",
         "  coverage to show."]
    return "\n".join(L)


def expected_fx(focal_mm: float, pixel_size_um: float) -> float:
    """fx in pixels predicted from lens focal length and sensor pixel pitch.

    An independent sanity check for cameras with no factory intrinsics (the
    Basler and Lucid units). It catches a degenerate pose set, which can yield a
    low reprojection error alongside a badly wrong K.
    """
    return focal_mm / (pixel_size_um / 1000.0)
