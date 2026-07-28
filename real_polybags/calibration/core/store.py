"""
Calibration result storage — OVGU AMS calibration tool.

Named `store`, not `io`: a module called `io.py` in a directory that is on
`sys.path` shadows Python's standard-library `io`, and the import silently
resolves to whichever wins. That produced an `AttributeError` deep inside a
request handler rather than at import time, which is exactly the kind of
failure that is expensive to trace.

A calibration is only useful later if it is unambiguous later. The schema
therefore records not just the numbers but everything needed to know whether
they still apply:

- **`image_size`** — intrinsics are resolution-specific. The Basler a2A1920 has
  a 1920x1200 sensor but this rig records 1280x720, so the camera is cropping or
  scaling; either way `K` measured at one resolution does not transfer to the
  other. A stored `K` without the resolution it was measured at is a trap.
- **the board used** — square size and dictionary. A calibration derived from a
  differently-sized reprint is silently wrong in scale.
- **quality figures and warnings** — so a poor calibration cannot be quietly
  reused as though it were good.
- **schema version and timestamp** — so old files remain readable and it is
  clear which is current.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def _clean(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def build_record(camera: str, intr_result=None, extr_result=None,
                 board_spec=None, source: str = "", notes: str = "",
                 reference: dict | None = None) -> dict:
    rec = {
        "schema_version": SCHEMA_VERSION,
        "camera": camera,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "notes": notes,
    }
    if board_spec is not None:
        rec["board"] = _clean(asdict(board_spec))

    if intr_result is not None:
        rec["intrinsics"] = _clean({
            "image_size": list(intr_result.image_size),
            "K": intr_result.K,
            "D": np.asarray(intr_result.D).ravel(),
            "fx": intr_result.fx, "fy": intr_result.fy,
            "cx": intr_result.cx, "cy": intr_result.cy,
            "rms_px": intr_result.rms,
            "n_views": len(intr_result.per_view_error),
            "per_view_error_px": intr_result.per_view_error,
            "coverage": intr_result.coverage,
            "warnings": intr_result.warnings,
        })

    if extr_result is not None:
        rec["extrinsics"] = _clean({
            "rvec": np.asarray(extr_result.rvec).ravel(),
            "tvec_mm": np.asarray(extr_result.tvec).ravel(),
            "R": extr_result.R,
            "camera_position_mm": extr_result.camera_position_mm,
            "height_above_belt_mm": extr_result.height_above_belt_mm,
            "H_belt_to_image": extr_result.H_belt_to_image,
            "H_image_to_belt": extr_result.H_image_to_belt,
            "reproj_error_px": extr_result.reproj_error_px,
            "n_points": extr_result.n_points,
            "board_origin_offset_mm": list(extr_result.board_origin_offset_mm),
            "warnings": extr_result.warnings,
        })

    # An independent reference (e.g. RealSense factory intrinsics, or an
    # expected fx from lens and sensor spec) is stored alongside so the
    # comparison stays visible rather than being made once and forgotten.
    if reference:
        rec["reference"] = _clean(reference)
    return rec


def save(record: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{record['camera']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def load(path: Path) -> dict:
    rec = json.loads(Path(path).read_text())
    v = rec.get("schema_version")
    if v != SCHEMA_VERSION:
        rec.setdefault("warnings", []).append(
            f"schema version {v} differs from current {SCHEMA_VERSION}")
    return rec


def load_arrays(record: dict) -> dict:
    """Pull the matrices back out as numpy, ready to use."""
    out = {}
    if "intrinsics" in record:
        out["K"] = np.array(record["intrinsics"]["K"], float)
        out["D"] = np.array(record["intrinsics"]["D"], float)
        out["image_size"] = tuple(record["intrinsics"]["image_size"])
    if "extrinsics" in record:
        e = record["extrinsics"]
        out["rvec"] = np.array(e["rvec"], float).reshape(3, 1)
        out["tvec"] = np.array(e["tvec_mm"], float).reshape(3, 1)
        out["H_belt_to_image"] = np.array(e["H_belt_to_image"], float)
        out["H_image_to_belt"] = np.array(e["H_image_to_belt"], float)
    return out


def summarise(results_dir: Path) -> str:
    """One-line-per-camera overview of everything calibrated so far."""
    files = sorted(Path(results_dir).glob("*.json"))
    if not files:
        return "No calibrations saved yet."
    L = [f"{'camera':<16}{'res':>11}{'fx':>9}{'fy':>9}{'rms':>7}{'ext err':>9}  status"]
    for f in files:
        r = load(f)
        i = r.get("intrinsics", {})
        e = r.get("extrinsics", {})
        size = i.get("image_size")
        warns = len(i.get("warnings", [])) + len(e.get("warnings", []))
        L.append(
            f"{r['camera']:<16}"
            f"{(f'{size[0]}x{size[1]}' if size else '-'):>11}"
            f"{i.get('fx', float('nan')):>9.1f}"
            f"{i.get('fy', float('nan')):>9.1f}"
            f"{i.get('rms_px', float('nan')):>7.2f}"
            f"{e.get('reproj_error_px', float('nan')):>9.2f}"
            f"  {'OK' if warns == 0 else f'{warns} warning(s)'}")
    return "\n".join(L)
