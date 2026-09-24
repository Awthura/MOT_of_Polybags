"""
Calibration bridge — OVGU AMS algorithm phase.

Loads the solved per-camera calibration produced by the calibration tool and
exposes the one operation this phase needs: **image pixel -> belt millimetre**.

Nothing here re-implements the transform. The calibration package already has a
correct, reviewed implementation (`beltmap.CameraOnBelt.to_belt`, which
undistorts with the board intrinsics before applying the homography for a metric
camera, and applies the raw-pixel homography directly for a homography-only
one). We import it as the single source of truth rather than copy the maths.

World frame (from the calibration tool, confirmed consistent with the map):
origin = centre of the 8x11 board under basler_1; +X along the belt, +Y across;
millimetres; Z = 0 on the belt plane.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# algorithm/core/calib.py -> algorithm/ -> real_polybags/
ALGO_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = ALGO_DIR.parent
CALIB_DIR = REPO_DIR / "calibration"
CALIB_CORE = CALIB_DIR / "core"

# The calibration modules import each other bare (e.g. beltmap does
# `import extrinsics`), exactly as calibration/app.py sets up, so its core/ must
# be on sys.path before importing them.
if str(CALIB_CORE) not in sys.path:
    sys.path.insert(0, str(CALIB_CORE))

import beltmap  # noqa: E402  (calibration/core)
import store    # noqa: E402  (calibration/core)


class Camera:
    """A calibrated camera ready to map detections onto the belt.

    Thin wrapper over `beltmap.CameraOnBelt` that also remembers where it came
    from and whether the result is metric (full intrinsics + pose) or a
    homography-only salvage — the streamer publishes that flag so the dashboard
    can mark edge-accuracy caveats.
    """

    def __init__(self, name: str, on_belt: "beltmap.CameraOnBelt", record: dict):
        self.name = name
        self.on_belt = on_belt
        self.record = record

    @property
    def metric(self) -> bool:
        return bool(self.on_belt.has_intrinsics)

    @property
    def image_size(self) -> tuple[int, int]:
        return tuple(self.on_belt.image_size)

    def foot_to_world(self, foot_px) -> tuple[float, float]:
        """One (x, y) image pixel -> (x_mm, y_mm) belt millimetres."""
        w = self.on_belt.to_belt(np.asarray(foot_px, float).reshape(1, 2))
        return (float(w[0, 0]), float(w[0, 1]))

    def feet_to_world(self, feet_px) -> np.ndarray:
        """(N, 2) image pixels -> (N, 2) belt millimetres, in one call."""
        pts = np.asarray(feet_px, float).reshape(-1, 2)
        if len(pts) == 0:
            return np.empty((0, 2))
        return self.on_belt.to_belt(pts)


def results_path(name: str, results_dir: Path) -> Path:
    return Path(results_dir) / f"{name}.json"


def load_camera(name: str, results_dir: Path) -> Camera:
    """Build a Camera from results/<name>.json.

    A homography-only record has no `rvec`, so `CameraOnBelt.has_intrinsics` is
    False and `to_belt` takes the direct-homography path — the intrinsics half,
    if present, is not (mis)used for undistortion. That matches how the
    homography was fitted (on raw pixels), so we simply hand the loaded arrays
    to CameraOnBelt and let it choose the route.
    """
    path = results_path(name, results_dir)
    if not path.exists():
        raise FileNotFoundError(f"no calibration for {name!r}: {path}")
    record = store.load(path)
    arr = store.load_arrays(record)
    if "H_image_to_belt" not in arr:
        raise ValueError(f"{name}: results file has no extrinsics.H_image_to_belt")
    image_size = arr.get("image_size")
    if image_size is None:
        raise ValueError(f"{name}: results file has no image_size")

    on_belt = beltmap.CameraOnBelt(
        name=name,
        image_size=tuple(image_size),
        K=arr.get("K"),
        D=arr.get("D"),
        rvec=arr.get("rvec"),
        tvec=arr.get("tvec"),
        H_image_to_belt=arr["H_image_to_belt"],
    )
    return Camera(name, on_belt, record)


def load_cameras(names, results_dir: Path) -> dict[str, Camera]:
    """Load several cameras; skips (with a printed note) any that fail to load."""
    out: dict[str, Camera] = {}
    for n in names:
        try:
            out[n] = load_camera(n, results_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"[calib] skipping {n}: {e}")
    return out
