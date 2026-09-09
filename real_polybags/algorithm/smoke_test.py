#!/usr/bin/env python3
"""
Smoke test for the calibration bridge and foot-point logic.

Runs without a model, a broker, or video — it checks the two pieces the whole
pipeline rests on:

1. `Camera.feet_to_world` reproduces the stored `world_points_mm` when fed the
   stored `image_points_px` from each camera's own results file. This is the
   round-trip the calibration solve was fitted to, so agreement to ~RMS-mm
   proves the bridge applies the homography exactly as the tool intended.
2. `Box.foot_px` returns bottom-centre for an axis-aligned box and the lowest
   edge midpoint for an oriented one.

Usage:  python3 algorithm/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))

import appconfig          # noqa: E402
import calib              # noqa: E402
from footpoint import Box  # noqa: E402


def _distort_to_raw(pts_rect, K, D) -> np.ndarray:
    """Rectified (undistorted) pixels -> raw distorted pixels.

    The metric cameras were calibrated on *rectified* frames, so their stored
    `image_points_px` are already undistorted and `H_image_to_belt` expects
    rectified input. Inference, however, runs on *raw* frames and relies on
    `to_belt` to undistort first. To validate that real path we push the stored
    rectified points back through the lens model (identity K projection with
    distortion) to recover the raw pixels a detector would actually produce.
    """
    pts = np.asarray(pts_rect, float).reshape(-1, 2)
    norm = (np.linalg.inv(K) @ np.c_[pts, np.ones(len(pts))].T).T[:, :2]
    pts3d = np.c_[norm, np.ones(len(norm))].astype(np.float64)
    raw, _ = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), K, D)
    return raw.reshape(-1, 2)


def test_transform(cfg) -> bool:
    """Confirm to_belt reproduces the stored fit through the raw-frame path.

    For a metric camera the stored points are rectified, so we synthesize the
    raw pixels a detector would see and feed those through `to_belt` (which
    undistorts). For a homography-only camera the stored points are already raw
    and `to_belt` applies the homography directly. In both cases the reproduced
    RMS must match the fit's own stored `rms_mm`.
    """
    results_dir = cfg.path("results_dir")
    ok = True
    print("== transform round-trip via the raw-frame inference path ==")
    # Every solved camera (a results/<cam>.json), independent of scenario.
    names = sorted(p.stem for p in results_dir.glob("*.json")
                   if not p.stem.startswith("_"))
    for name in names:
        try:
            cam = calib.load_camera(name, results_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:14s} SKIP  ({e})")
            continue
        corr = cam.record.get("correspondences", {})
        img = corr.get("image_points_px")
        wld = corr.get("world_points_mm")
        if not img or not wld:
            print(f"  {name:14s} no stored correspondences to check "
                  f"(metric={cam.metric})")
            continue
        want = np.asarray(wld, float).reshape(-1, 2)
        if cam.metric:
            K = cam.on_belt.K
            D = cam.on_belt.D
            raw_px = _distort_to_raw(img, K, D)     # rectified -> raw
        else:
            raw_px = np.asarray(img, float)         # already raw
        got = cam.feet_to_world(raw_px)
        resid = np.linalg.norm(got - want, axis=1)
        rms = float(np.sqrt((resid ** 2).mean()))
        stored = float(cam.record["extrinsics"].get("rms_mm", float("nan")))
        # We should reproduce the fit's own residual, whatever it is.
        good = (not np.isnan(stored)) and abs(rms - stored) < 0.5
        ok &= good
        print(f"  {name:14s} {'OK ' if good else '!! '} metric={str(cam.metric):5s} "
              f"n={len(want):2d}  reproduced RMS={rms:6.3f} mm  "
              f"(stored {stored:6.3f} mm)")
    return ok


def test_footpoint() -> bool:
    print("== foot-point extraction ==")
    ok = True

    aabb = Box(kind="aabb", conf=0.9, xywh=(100.0, 200.0, 40.0, 60.0))
    fx, fy = aabb.foot_px()
    exp = (100.0, 230.0)  # cx, cy + h/2
    good = np.allclose([fx, fy], exp)
    ok &= good
    print(f"  aabb foot = ({fx:.1f}, {fy:.1f})  expect {exp}  "
          f"{'OK' if good else '!!'}")

    # A square rotated 45°, centre (100,100), so vertices at N/E/S/W; the two
    # lowest (largest y) are S and its neighbours -> midpoint near the bottom.
    corners = np.array([[100, 60], [140, 100], [100, 140], [60, 100]], float)
    obb = Box(kind="obb", conf=0.8, corners=corners)
    fx, fy = obb.foot_px()
    # lowest two vertices are (100,140) and one of the y=100 pair -> mean y=120
    good = abs(fy - 120.0) < 1e-6
    ok &= good
    print(f"  obb  foot = ({fx:.1f}, {fy:.1f})  (lowest-edge midpoint)  "
          f"{'OK' if good else '!!'}")
    return ok


def main() -> int:
    cfg = appconfig.load()
    a = test_transform(cfg)
    b = test_footpoint()
    print()
    if a and b:
        print("ALL OK")
        return 0
    print("FAILURES above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
