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
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

SCHEMA_VERSION = 1


def _clean(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        v = o.item()
        return None if isinstance(v, float) and math.isnan(v) else v
    # A factory-adopted calibration has no fitted RMS, recorded as NaN rather
    # than a zero that would read as a suspiciously perfect fit. NaN is not
    # valid JSON (json.dumps emits a bare `NaN` token that Python accepts but
    # JavaScript's JSON.parse rejects outright), so it is written as null —
    # the same "not applicable" this schema already uses elsewhere.
    if isinstance(o, float) and math.isnan(o):
        return None
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


def build_correspondence_record(camera: str, res, image_size,
                                intr_record: dict | None = None,
                                source: str = "", notes: str = "") -> dict:
    """A calibration solved from clicked image<->map point pairs.

    Written into the SAME `extrinsics` block as a board solve, so everything
    downstream — the belt map, the status board, `load_arrays` — consumes it
    without knowing which route produced it. What differs is recorded, not
    inferred:

    - `method` distinguishes it from a board solve for anyone who needs to.
    - The correspondences themselves are kept. They are the entire input, they
      cost nothing to store, and keeping them means a fit can be re-run,
      audited, or improved by adding a point later without re-clicking the
      ones already done.
    - With intrinsics the decomposed `rvec`/`tvec` are stored too, so the
      camera gets a real position and the map can draw it. Without them,
      those keys are simply absent rather than filled with placeholders.
    """
    ext = {
        "H_image_to_belt": res.H_image_to_world,
        "H_belt_to_image": res.H_world_to_image,
        "image_size": list(image_size),
        "rms_mm": res.rms_mm,
        "max_error_mm": res.max_mm,
        "n_points": int(len(res.image_points)),
        "n_inliers": int(res.inliers.sum()),
        "used_intrinsics": bool(res.used_intrinsics),
        "warnings": list(res.warnings),
    }
    if res.rvec is not None:
        R, _ = cv2.Rodrigues(np.asarray(res.rvec, float))
        ext.update({
            "rvec": np.asarray(res.rvec).ravel(),
            "tvec_mm": np.asarray(res.tvec).ravel(),
            "R": R,
            "camera_position_mm": res.camera_position_mm,
            "height_above_belt_mm": abs(float(res.camera_position_mm[2])),
        })
    # NB: no `reproj_error_px` — this route's error is in millimetres on the
    # world plane (`rms_mm`), not pixels in the image. Writing it under the
    # board route's key would put two different quantities in one column.

    rec = {
        "schema_version": SCHEMA_VERSION,
        "camera": camera,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "homography_points",
        "source": source,
        "notes": notes,
        "extrinsics": _clean(ext),
        "correspondences": _clean({
            "image_points_px": res.image_points,
            "world_points_mm": res.world_points,
            "residuals_mm": res.residuals_mm,
            "inliers": res.inliers.astype(int),
        }),
        "limitations": ([] if res.used_intrinsics else [
            "no intrinsics: lens distortion is not corrected, so error grows "
            "toward the frame edges",
            "no camera pose: a homography fixes the plane mapping, not where "
            "the camera is",
        ]),
    }
    # Carry the camera's existing intrinsics through untouched — this route
    # solves extrinsics only, and a save must not drop the other half.
    if intr_record:
        rec["intrinsics"] = intr_record
    return rec


def build_record(camera: str, intr_result=None, extr_result=None,
                 board_spec=None, source: str = "", notes: str = "",
                 reference: dict | None = None,
                 intrinsics_method: str = "board",
                 device: dict | None = None) -> dict:
    rec = {
        "schema_version": SCHEMA_VERSION,
        "camera": camera,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "notes": notes,
    }
    # A board fit still needs the board spec on record even when the ChArUco
    # capture flow was skipped; harmless when the board simply wasn't used.
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
            # "board" (fitted from ChArUco captures here) or "factory" (adopted
            # from the camera's own calibration, e.g. RealSense). Distinguishes
            # a fitted RMS of "not applicable" from one that is genuinely zero.
            "method": intrinsics_method,
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
    # Which physical camera this was actually measured from — see
    # `sources.FrameSource.device_info`. Distinct from `camera`, which is only
    # ever a label the operator typed in.
    if device:
        rec["device"] = _clean(device)
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


def carry_forward_device(record: dict, results_dir: Path) -> dict:
    """Keep an already-known device serial when this save doesn't supply one.

    Unlike a pose, device identity has no staleness concept — a camera's
    serial does not change when its lens is re-measured, so this carries it
    forward unconditionally rather than flagging it. Matters most for Route B
    (recorded video -> extracted frames -> Folder source): a folder session
    has no live hardware to ask, so without this, re-calibrating a camera's
    lens from an old recording would quietly erase whatever serial an earlier
    live session — or an operator manually asserting it from the recorder's
    console output — had recorded.
    """
    if record.get("device"):
        return record
    path = Path(results_dir) / f"{record['camera']}.json"
    if not path.exists():
        return record
    try:
        old = load(path)
    except Exception:
        return record
    if old.get("device"):
        record["device"] = old["device"]
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


def intrinsics_method_of(record: dict) -> str | None:
    """How the intrinsics half was obtained: 'board' or 'factory'.

    None means this record has no intrinsics at all (e.g. a tape-measure
    `homography_tape` record). 'board' is the default for records saved before
    this field existed — every intrinsics fit was a board capture until the
    factory-adoption path was added, so that default is not a guess.
    """
    intr = record.get("intrinsics")
    return intr.get("method", "board") if intr else None


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


def _num(v, default=float("nan")) -> float:
    """A stored number, or NaN — `None` is a legitimate value in this schema
    (a factory calibration has no fitted RMS), and formatting it would raise."""
    return default if v is None else v


def summarise(results_dir: Path) -> str:
    """One-line-per-camera overview of everything calibrated so far.

    The extrinsic-error column carries different units per route — pixels for
    a board solve, millimetres on the world plane for a point-correspondence
    solve — so it is labelled per row rather than pretending one number means
    the same thing everywhere.
    """
    files = sorted(Path(results_dir).glob("*.json"))
    if not files:
        return "No calibrations saved yet."
    L = [f"{'camera':<16}{'res':>11}{'fx':>9}{'fy':>9}{'rms':>7}"
         f"{'ext err':>12}  status"]
    for f in files:
        r = load(f)
        i = r.get("intrinsics", {})
        e = r.get("extrinsics", {})
        size = i.get("image_size")
        warns = len(i.get("warnings", [])) + len(e.get("warnings", []))
        if e.get("reproj_error_px") is not None:
            ext_err = f"{e['reproj_error_px']:.2f}px"
        elif e.get("rms_mm") is not None:
            ext_err = f"{e['rms_mm']:.1f}mm"
        else:
            ext_err = "-"
        L.append(
            f"{r['camera']:<16}"
            f"{(f'{size[0]}x{size[1]}' if size else '-'):>11}"
            f"{_num(i.get('fx')):>9.1f}"
            f"{_num(i.get('fy')):>9.1f}"
            f"{_num(i.get('rms_px')):>7.2f}"
            f"{ext_err:>12}"
            f"  {'OK' if warns == 0 else f'{warns} warning(s)'}")
    return "\n".join(L)
