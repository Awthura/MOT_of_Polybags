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


def build_plane_record(camera: str, H_image_to_belt, image_size,
                       width_mm: float, length_mm: float,
                       image_points, session: str = "", notes: str = "",
                       propagated_from: str | None = None) -> dict:
    """A calibration obtained from four clicked points and a tape measure.

    Written into the SAME schema as a board calibration so everything
    downstream reads one format — but with `method` recorded, because the two
    are not equivalent and a consumer must be able to tell them apart:
    this route has no intrinsics, so lens distortion is uncorrected and
    accuracy degrades toward the frame edges.

    `reference_points` are kept deliberately. When intrinsics are measured
    later, the same clicks can be undistorted and re-solved into a better
    homography without going back to the belt with a tape measure.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "camera": camera,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "homography_tape",
        "source": session,
        "notes": notes,
        "extrinsics": _clean({
            "H_image_to_belt": H_image_to_belt,
            "H_belt_to_image": np.linalg.inv(np.asarray(H_image_to_belt, float)),
            "image_size": list(image_size),
        }),
        "reference_points": _clean({
            "image_points": image_points,
            "belt_points_mm": [[0, 0], [width_mm, 0],
                               [width_mm, length_mm], [0, length_mm]],
            "width_mm": width_mm, "length_mm": length_mm,
            "propagated_from": propagated_from,
        }),
        "limitations": [
            "no intrinsics: lens distortion is not corrected, so error grows "
            "toward the frame edges",
            "no camera pose: a homography fixes the plane mapping, not where "
            "the camera is",
        ],
    }


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


def carry_forward_extrinsics(record: dict, results_dir: Path) -> dict:
    """Keep an already-saved pose when the record being written has none.

    `results/<camera>.json` holds one camera's whole calibration, and a save
    rewrites the file. Intrinsics and extrinsics are now measured in separate
    sittings on separate pages, so the ordinary case — solve the pose at the
    rig, then later re-measure the lens at a desk — would otherwise delete the
    pose as a side effect of saving the lens. Nothing would report that; the
    camera would simply drop off the belt map.

    If the intrinsics changed, the carried pose is kept but flagged: it was
    solved against a different `K`, so it is stale rather than merely old. It is
    not silently discarded either — deciding which is right belongs to the
    operator, who knows whether the camera has been touched.
    """
    if "extrinsics" in record:
        return record
    path = Path(results_dir) / f"{record['camera']}.json"
    if not path.exists():
        return record
    try:
        old = load(path)
    except Exception:
        return record
    if "extrinsics" not in old:
        return record

    carried = dict(old["extrinsics"])
    old_K, new_K = (old.get("intrinsics") or {}).get("K"), \
                   (record.get("intrinsics") or {}).get("K")
    if old_K is not None and new_K is not None and \
            not np.allclose(np.array(old_K, float), np.array(new_K, float)):
        carried.setdefault("warnings", [])
        carried["warnings"] = list(carried["warnings"]) + [
            "pose carried over from a previous save and solved against "
            "different intrinsics — re-solve the belt plane, or discard it"]
        carried["stale"] = True
    record["extrinsics"] = carried
    record.setdefault("notes", "")
    return record


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


def method_of(record: dict) -> str:
    """How this calibration was obtained: 'board' or 'homography_tape'."""
    return record.get("method", "board" if "intrinsics" in record else "unknown")


def is_synthetic(record: dict) -> bool:
    """Was this calibration measured from the synthetic camera?

    Worth its own check because the failure it prevents is invisible. The
    synthetic source renders at 1280x720 — exactly what this rig records at — so
    a synthetic `basler_1.json` passes the resolution guard, reloads silently
    into a real session, and a real camera's pose is then solved against a
    lens model that belongs to no lens. Every millimetre downstream is wrong,
    and nothing about the result looks unusual.
    """
    return record.get("source") == "synthetic" or "synthetic_truth" in (
        record.get("reference") or {})


def load_arrays(record: dict) -> dict:
    """Pull the matrices back out as numpy, ready to use."""
    out = {}
    if "intrinsics" in record:
        out["K"] = np.array(record["intrinsics"]["K"], float)
        out["D"] = np.array(record["intrinsics"]["D"], float)
        out["image_size"] = tuple(record["intrinsics"]["image_size"])
    if "extrinsics" in record:
        e = record["extrinsics"]
        # A tape-measured record has the homographies but no pose, so these are
        # fetched conditionally rather than assumed present.
        if "rvec" in e:
            out["rvec"] = np.array(e["rvec"], float).reshape(3, 1)
        if "tvec_mm" in e:
            out["tvec"] = np.array(e["tvec_mm"], float).reshape(3, 1)
        if "H_belt_to_image" in e:
            out["H_belt_to_image"] = np.array(e["H_belt_to_image"], float)
        if "H_image_to_belt" in e:
            out["H_image_to_belt"] = np.array(e["H_image_to_belt"], float)
        if "image_size" in e and "image_size" not in out:
            out["image_size"] = tuple(e["image_size"])
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
