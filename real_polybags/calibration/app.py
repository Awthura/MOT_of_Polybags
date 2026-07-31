"""
OVGU AMS calibration tool — local web app.

Serves a single page and a small JSON API over the calibration core. No build
step, no Node, no Docker: `python3 app.py` and open the printed URL.

    python3 app.py
    python3 app.py --port 5001 --results-dir results

Design notes:

- **State lives in one in-process session.** This is a single-operator tool run
  next to the rig, not a service. A database would add operational weight for
  no benefit; results are written as JSON files that are readable without this
  app running.
- **The live preview is MJPEG** (`multipart/x-mixed-replace`), which every
  browser handles natively with no client-side video library.
- **The synthetic source works with no hardware**, so the whole workflow can be
  exercised — and, because its true `K` is known, actually verified — before
  the lab session.
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "core"))

import board as board_mod            # noqa: E402
import intrinsics as intr            # noqa: E402
import extrinsics as extr            # noqa: E402
import sources as src_mod            # noqa: E402
import beltmap as bmap               # noqa: E402
import store as calstore             # noqa: E402  (NOT 'io' — that
                                     #   shadows the stdlib module on sys.path)
import plan as rigplan               # noqa: E402
import worldmap as wmap              # noqa: E402
import correspond as corr            # noqa: E402

app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="")


class Session:
    """Everything the operator is currently working on, for one camera."""

    def __init__(self):
        self.lock = threading.Lock()
        self.source = None
        self.source_id = None
        self.camera = "camera"
        self.spec = board_mod.PRESETS["small"]
        self.shots: list[intr.Detection] = []
        self.thumbs: list[str] = []
        self.intr_result = None
        # Whether the intrinsics currently in hand came from the synthetic
        # camera. Tracked because reusing them on real hardware is undetectable
        # by any other means — see the guard in /api/extrinsics.
        self.intr_synthetic = False
        # How the intrinsics currently in hand were obtained: "board" (fitted
        # from ChArUco captures here) or "factory" (adopted from the camera's
        # own calibration). Carried into the saved record so a reload can tell
        # the two apart later.
        self.intr_method = "board"
        self.extr_result = None
        # Which physical camera this session actually opened — serial and
        # model, from the source itself, not from what the operator typed as
        # the camera name. None where the source has no such identity
        # (synthetic, folder).
        self.device_info = None
        # The point-correspondence solve in progress (map <-> one frame), kept
        # in the session so validate/save operate on exactly what was solved.
        self.corr_result = None
        self.corr_image_size = None
        self.corr_intr_record = None
        self.last_frame = None
        self.last_detection = None
        self.results_dir = HERE / "results"

    def close(self):
        if self.source is not None:
            try:
                self.source.close()
            except Exception:
                pass
        self.source = None
        self.source_id = None


S = Session()


# ── Frame handling ───────────────────────────────────────────────────────────

def annotate(frame, spec):
    """Draw the current detection and a coverage hint over a preview frame."""
    out = frame.copy()
    det = intr.detect_charuco(frame, spec, source="live")
    h, w = out.shape[:2]

    if det is not None:
        pts = det.image_points.reshape(-1, 2)
        for p in pts:
            cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 220, 90), -1)
        x0, y0 = pts.min(axis=0).astype(int)
        x1, y1 = pts.max(axis=0).astype(int)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 90), 2)
        label = f"{det.n_points}/{spec.n_corners} corners"
        colour = (0, 220, 90)
    else:
        label = "no board detected"
        colour = (60, 60, 235)

    cv2.rectangle(out, (0, 0), (w, 34), (28, 28, 32), -1)
    cv2.putText(out, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
    cv2.putText(out, f"shots: {len(S.shots)}", (w - 130, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 205), 2)
    return out, det


def mjpeg():
    while True:
        with S.lock:
            source = S.source
            spec = S.spec
        if source is None:
            time.sleep(0.2)
            continue
        try:
            frame = source.read()
        except Exception as e:
            frame = None
            print(f"[source] read failed: {e}")
        if frame is None:
            time.sleep(0.05)
            continue

        annotated, det = annotate(frame, spec)
        with S.lock:
            S.last_frame = frame
            S.last_detection = det

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(1 / 15)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/extrinsics")
def extrinsics_page():
    """The rig-phase page, deliberately separate from the intrinsics one.

    Intrinsics are bench work and extrinsics are rig work; the manual already
    treats them as two sittings, and putting them on one page invites the wrong
    order. This page is usable on its own: it reloads saved intrinsics, shows
    what is still outstanding across every camera, and checks each solve against
    the measurement that was anticipated for it.
    """
    return send_from_directory(app.static_folder, "extrinsics.html")


@app.route("/api/config")
def api_config():
    return jsonify({
        "sources": src_mod.probe(),
        "boards": [{"id": k, "name": s.name, "squares": [s.squares_x, s.squares_y],
                    "square_mm": s.square_mm, "dictionary": s.dictionary,
                    "corners": s.n_corners, "paper": s.paper}
                   for k, s in board_mod.PRESETS.items()],
        "results_dir": str(S.results_dir),
    })


@app.route("/api/session", methods=["POST"])
def api_session():
    data = request.get_json(force=True)
    src_id = data.get("source", "synthetic")
    preset = data.get("board", "small")
    camera = (data.get("camera") or "camera").strip() or "camera"
    # Which physical unit to open, for sources where more than one can be on
    # the rig at once (two Baslers; potentially two RealSense). Left blank
    # when there is only one device — BaslerSource / RealSenseSource accept
    # that and pick the one device themselves, but refuse to guess between two.
    device_serial = (data.get("device_serial") or "").strip() or None

    with S.lock:
        S.close()
        S.spec = board_mod.PRESETS[preset]
        S.camera = camera
        S.shots, S.thumbs = [], []
        S.intr_result = S.extr_result = None
        S.intr_synthetic = False
        S.intr_method = "board"
        S.device_info = None
        try:
            if src_id == "synthetic":
                S.source = src_mod.SyntheticSource(S.spec)
            elif src_id == "folder":
                S.source = src_mod.FolderSource(data.get("folder", ""))
            elif src_id in ("basler", "realsense"):
                S.source = src_mod.AVAILABLE[src_id](serial=device_serial)
            else:
                S.source = src_mod.AVAILABLE[src_id]()
            S.source_id = src_id
            info = S.source.device_info()
            if info is None and device_serial:
                # No live device to ask (folder / synthetic), but the operator
                # is asserting which physical camera these frames came from —
                # typically read off the recorder's console output at record
                # time, which already prints the serial for exactly this
                # purpose. Recorded as a claim, not a hardware fact: routes
                # that DO ask hardware never set "asserted".
                info = {"serial": device_serial, "model": None, "asserted": True}
            S.device_info = info
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 400

    ref = None
    if hasattr(S.source, "factory_intrinsics"):
        ref = S.source.factory_intrinsics()

    # Reload this camera's saved intrinsics if it has been calibrated before.
    # Without this, extrinsics could only ever follow intrinsics inside one
    # sitting - which forbids the natural order of work: calibrate all four
    # cameras' lenses first (that needs no rig access at all), then mount them,
    # place the board on the belt once, and solve each camera's pose against
    # that single placement without disturbing it.
    loaded = None
    path = S.results_dir / f"{camera}.json"
    if path.exists():
        try:
            rec = calstore.load(path)
            arr = calstore.load_arrays(rec)
            if "K" in arr:
                iv = rec["intrinsics"]
                stored_size = tuple(iv["image_size"])
                r = intr.IntrinsicResult(
                    camera=camera, image_size=stored_size,
                    K=arr["K"], D=arr["D"], rms=iv.get("rms_px", float("nan")),
                    per_view_error=iv.get("per_view_error_px", []),
                    view_sources=[], coverage=iv.get("coverage", {}),
                    warnings=list(iv.get("warnings", [])))
                with S.lock:
                    S.intr_result = r
                    S.intr_synthetic = calstore.is_synthetic(rec)
                    S.intr_method = calstore.intrinsics_method_of(rec) or "board"
                loaded = {"fx": r.fx, "fy": r.fy, "cx": r.cx, "cy": r.cy,
                          "rms": r.rms, "image_size": list(stored_size),
                          "created": rec.get("created_utc"),
                          "saved_source": rec.get("source"),
                          "method": calstore.intrinsics_method_of(rec) or "board",
                          # The synthetic camera renders at 1280x720, the same
                          # size this rig records at, so the resolution guard
                          # above cannot catch a rehearsal result being reused
                          # on real hardware. This can.
                          "synthetic": calstore.is_synthetic(rec),
                          "warnings": r.warnings}
        except Exception as e:
            loaded = {"error": f"could not reload {path.name}: {e}"}

    return jsonify({"ok": True, "source": src_id, "camera": camera,
                    "board": preset, "truth": getattr(S.source, "truth", None),
                    "factory_intrinsics": ref, "loaded_intrinsics": loaded,
                    "device": S.device_info})


@app.route("/api/stream")
def api_stream():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/capture", methods=["POST"])
def api_capture():
    with S.lock:
        frame, det = S.last_frame, S.last_detection
        if frame is None:
            return jsonify({"ok": False, "error": "no frame yet"}), 400
        if det is None:
            return jsonify({"ok": False, "error": "no board detected in this frame"}), 400
        det.source = f"shot_{len(S.shots):02d}"
        S.shots.append(det)
        thumb = cv2.resize(frame, (160, 90))
        ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        import base64
        S.thumbs.append(base64.b64encode(buf).decode() if ok else "")
        n = len(S.shots)
    return jsonify({"ok": True, "n_shots": n, "coverage": coverage_now()})


@app.route("/api/capture_all", methods=["POST"])
def api_capture_all():
    """Ingest every image in a folder source exactly once.

    Clicking *Capture* cannot do this job. The preview reads the folder at
    15 fps, so a 20-image set cycles in 1.3 seconds: hand-clicking samples it
    effectively at random and takes some views twice. Duplicates are not
    harmless — repeating a view weights it in the fit, which is the same
    degeneracy as capturing twenty frontal shots, and it arrives disguised as a
    healthy sample size.

    Each file is read from disk directly rather than through `source.read()`, so
    this neither races the preview thread nor depends on where its cursor
    happens to be.
    """
    import base64
    with S.lock:
        source, spec = S.source, S.spec
    if not isinstance(source, src_mod.FolderSource):
        return jsonify({"ok": False,
                        "error": "capture-all applies to the folder source only"}), 400

    added, rejected = 0, []
    for path in source.files:
        img = cv2.imread(path)
        name = Path(path).name
        if img is None:
            rejected.append({"file": name, "why": "unreadable"})
            continue
        det = intr.detect_charuco(img, spec, source=name)
        if det is None:
            rejected.append({"file": name, "why": "no board detected"})
            continue
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (160, 90)),
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        with S.lock:
            S.shots.append(det)
            S.thumbs.append(base64.b64encode(buf).decode() if ok else "")
        added += 1

    return jsonify({"ok": True, "added": added, "n_files": len(source.files),
                    "rejected": rejected, "coverage": coverage_now()})


@app.route("/api/shots")
def api_shots():
    with S.lock:
        shots = [{"i": i, "source": d.source, "points": d.n_points,
                  "thumb": S.thumbs[i] if i < len(S.thumbs) else ""}
                 for i, d in enumerate(S.shots)]
    return jsonify({"shots": shots, "coverage": coverage_now()})


@app.route("/api/shots/<int:i>", methods=["DELETE"])
def api_delete_shot(i):
    with S.lock:
        if 0 <= i < len(S.shots):
            S.shots.pop(i)
            if i < len(S.thumbs):
                S.thumbs.pop(i)
    return jsonify({"ok": True, "n_shots": len(S.shots), "coverage": coverage_now()})


def coverage_now(grid: int = 4) -> dict:
    """Live coverage feedback — the thing that actually determines quality.

    Reported before calibration, not after, so the operator can fix a thin pose
    set while still holding the board rather than discovering it afterwards.
    """
    if not S.shots:
        return {"cells": [[0] * grid for _ in range(grid)], "seen": 0,
                "total": grid * grid, "n_shots": 0, "ready": False,
                "advice": "Capture shots with the board tilted and moved around the frame."}
    w, h = S.shots[0].image_size
    occ = np.zeros((grid, grid), bool)
    for d in S.shots:
        for p in d.image_points.reshape(-1, 2):
            gx = min(grid - 1, max(0, int(p[0] / w * grid)))
            gy = min(grid - 1, max(0, int(p[1] / h * grid)))
            occ[gy, gx] = True
    seen = int(occ.sum())
    n = len(S.shots)
    ready = n >= 10 and seen >= grid * grid * 0.75

    if n < 10:
        advice = f"{n} shots — aim for at least 10-15."
    elif seen < grid * grid * 0.75:
        advice = (f"Only {seen}/{grid*grid} regions covered. Move the board into "
                  f"the frame edges and corners, where distortion is strongest.")
    else:
        advice = "Coverage looks good. Tilt variety is checked after calibrating."
    return {"cells": occ.astype(int).tolist(), "seen": seen, "total": grid * grid,
            "n_shots": n, "ready": ready, "advice": advice}


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    with S.lock:
        shots = list(S.shots)
        camera, spec = S.camera, S.spec
        truth = getattr(S.source, "truth", None) if S.source else None
    if len(shots) < intr.MIN_VIEWS:
        return jsonify({"ok": False,
                        "error": f"need at least {intr.MIN_VIEWS} shots, have {len(shots)}"}), 400
    try:
        res = intr.calibrate(shots, camera=camera)
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 400

    with S.lock:
        S.intr_result = res
        S.intr_synthetic = (S.source_id == "synthetic")
        S.intr_method = "board"

    payload = {
        "ok": True,
        "fx": res.fx, "fy": res.fy, "cx": res.cx, "cy": res.cy,
        "K": res.K.tolist(), "D": np.asarray(res.D).ravel().tolist(),
        "rms": res.rms, "per_view_error": res.per_view_error,
        "image_size": list(res.image_size),
        "coverage": res.coverage, "warnings": res.warnings,
        "report": intr.format_report(res),
    }
    # When the source knows its own truth (synthetic) or carries factory values
    # (RealSense), show the comparison immediately — that is the difference
    # between a plausible number and a verified one.
    if truth:
        payload["truth"] = truth
        payload["truth_delta"] = {
            "fx_pct": abs(res.fx - truth["fx"]) / truth["fx"] * 100,
            "fy_pct": abs(res.fy - truth["fy"]) / truth["fy"] * 100,
            "cx_px": abs(res.cx - truth["cx"]),
            "cy_px": abs(res.cy - truth["cy"]),
        }
    return jsonify(payload)


@app.route("/api/factory_intrinsics", methods=["POST"])
def api_factory_intrinsics():
    """Adopt the camera's own factory calibration instead of fitting one.

    The RealSense D435 is factory-calibrated — the only camera on this rig
    with independent ground truth — so its own numbers can be used directly
    rather than always re-fitting from a board capture. Doing the board fit
    once and comparing (see the "vs known reference" panel after Calibrate) is
    still the acceptance test for the method itself; this is the fast path for
    every session after that has been established once.
    """
    with S.lock:
        source, camera = S.source, S.camera
    if source is None or not hasattr(source, "factory_intrinsics"):
        return jsonify({"ok": False, "error":
                        "this source has no factory intrinsics — only the "
                        "RealSense reports its own calibration"}), 400
    factory = source.factory_intrinsics()
    if not factory:
        return jsonify({"ok": False, "error":
                        "the camera did not report factory intrinsics"}), 400
    try:
        res = intr.from_factory_intrinsics(camera, factory)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    with S.lock:
        S.intr_result = res
        S.intr_synthetic = False
        S.intr_method = "factory"

    return jsonify({
        "ok": True, "method": "factory", "model": factory.get("model"),
        "fx": res.fx, "fy": res.fy, "cx": res.cx, "cy": res.cy,
        "K": res.K.tolist(), "D": np.asarray(res.D).ravel().tolist(),
        "image_size": list(res.image_size), "warnings": res.warnings,
        "report": intr.format_factory_report(res, factory.get("model", "")),
    })


@app.route("/api/extrinsics", methods=["POST"])
def api_extrinsics():
    """Solve this camera's pose from the board lying flat on the belt."""
    data = request.get_json(force=True) or {}
    off = (float(data.get("origin_x_mm", 0.0)), float(data.get("origin_y_mm", 0.0)))
    with S.lock:
        res_i = S.intr_result
        det = S.last_detection
        camera = S.camera
        synth_intr = S.intr_synthetic
        src_id = S.source_id
    if res_i is None:
        return jsonify({"ok": False, "error": "calibrate intrinsics first"}), 400
    # Synthetic intrinsics on a real camera. The synthetic source renders at
    # 1280x720 — what this rig records at — so the resolution check cannot see
    # this, and the resulting pose looks entirely normal while being wrong by
    # whatever the real lens differs from the virtual one. Refused rather than
    # warned: there is no case where it is the intended thing to do.
    if synth_intr and src_id != "synthetic":
        return jsonify({"ok": False, "error":
                        f"the saved intrinsics for {camera} were measured from "
                        f"the SYNTHETIC camera, but this session is running on "
                        f"'{src_id}'. Those numbers describe a virtual lens. "
                        f"Delete results/{camera}.json and calibrate this "
                        f"camera's lens for real before solving its pose."}), 400
    if det is None:
        return jsonify({"ok": False, "error": "no board visible — place it flat on the belt"}), 400
    # Intrinsics are resolution-specific. Reloaded ones may have been measured
    # at a different capture resolution, and silently mixing them would produce
    # a plausible-looking pose that is wrong by the scale ratio.
    if tuple(det.image_size) != tuple(res_i.image_size):
        return jsonify({"ok": False, "error":
                        f"intrinsics were measured at {res_i.image_size[0]}x"
                        f"{res_i.image_size[1]} but this camera is delivering "
                        f"{det.image_size[0]}x{det.image_size[1]} — recalibrate "
                        f"intrinsics at the resolution you record at"}), 400
    try:
        ext = extr.calibrate_extrinsics(det.object_points, det.image_points,
                                        res_i.K, res_i.D, camera=camera,
                                        origin_offset_mm=off)
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 400
    with S.lock:
        S.extr_result = ext

    # Check the solve against what was anticipated for this camera. A pose is
    # easy to look at and hard to judge on its own — 298 mm above the belt reads
    # as a fine number until it is set beside the 1400 mm the camera is actually
    # mounted at.
    p = rigplan.load_plan(S.results_dir)
    cplan = rigplan.camera_plan(p, camera) or rigplan.blank_camera(camera)
    checks = rigplan.verify(cplan, ext.height_above_belt_mm,
                            ext.reproj_error_px, off)

    return jsonify({
        "ok": True,
        "camera": camera,
        "camera_position_mm": ext.camera_position_mm.tolist(),
        "height_above_belt_mm": ext.height_above_belt_mm,
        "reproj_error_px": ext.reproj_error_px,
        "n_points": ext.n_points,
        "H_image_to_belt": ext.H_image_to_belt.tolist(),
        "warnings": ext.warnings,
        "checks": checks,
        "verdict": rigplan.worst_status(checks),
    })


@app.route("/api/to_belt", methods=["POST"])
def api_to_belt():
    """Click-to-validate: a pixel in, belt millimetres out."""
    data = request.get_json(force=True)
    with S.lock:
        res_i, ext = S.intr_result, S.extr_result
    if res_i is None or ext is None:
        return jsonify({"ok": False, "error": "need both intrinsics and extrinsics"}), 400
    pt = np.array([[float(data["x"]), float(data["y"])]], np.float32)
    belt = extr.image_to_belt(pt, res_i.K, res_i.D, ext.H_image_to_belt)[0]
    scale = extr.mm_per_pixel(ext.H_image_to_belt, res_i.K, res_i.D,
                              (float(data["x"]), float(data["y"])))
    return jsonify({"ok": True, "x_mm": float(belt[0]), "y_mm": float(belt[1]),
                    "mm_per_px": scale})


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(force=True) or {}
    with S.lock:
        if S.intr_result is None:
            return jsonify({"ok": False, "error": "nothing to save"}), 400
        ref = None
        if S.source is not None:
            if hasattr(S.source, "factory_intrinsics"):
                ref = {"factory_intrinsics": S.source.factory_intrinsics()}
            elif getattr(S.source, "truth", None):
                ref = {"synthetic_truth": S.source.truth}
        rec = calstore.build_record(
            camera=S.camera, intr_result=S.intr_result, extr_result=S.extr_result,
            board_spec=S.spec, source=S.source_id or "", reference=ref,
            notes=data.get("notes", ""), intrinsics_method=S.intr_method,
            device=S.device_info)
        # Saving rewrites the whole per-camera file. Intrinsics and extrinsics
        # are measured in separate sittings, so a bench save must not delete a
        # pose solved earlier at the rig; nor should it erase a device serial
        # this session's source had no way to ask about (e.g. a folder replay).
        rec = calstore.carry_forward_extrinsics(rec, S.results_dir)
        rec = calstore.carry_forward_device(rec, S.results_dir)
        path = calstore.save(rec, S.results_dir)
    return jsonify({"ok": True, "path": str(path)})


@app.route("/api/results")
def api_results():
    return jsonify({"summary": calstore.summarise(S.results_dir)})


# Camera names become filenames (results/<camera>.json) and URL path segments
# for the delete route below; restricting the charset rules out path traversal
# via a name like "../../etc" before it ever reaches the filesystem.
_CAMERA_NAME_RE = re.compile(r"^[\w.\-]+$")


@app.route("/api/results/<camera>", methods=["DELETE"])
def api_delete_results(camera):
    """Remove a camera's saved calibration entirely — intrinsics and any pose.

    A result file holds both halves together, so there is no smaller unit to
    delete without inventing a partial schema for it. This is the in-app form
    of the `rm results/<camera>.json` step the procedure already documents for
    clearing a rehearsal (synthetic) result before real work — most useful for
    exactly that, and for discarding a bad calibration to force a redo.
    """
    if not _CAMERA_NAME_RE.match(camera):
        return jsonify({"ok": False, "error": "invalid camera name"}), 400
    path = S.results_dir / f"{camera}.json"
    existed = path.exists()
    if existed:
        path.unlink()
    with S.lock:
        if S.camera == camera:
            S.intr_result = None
            S.extr_result = None
            S.intr_synthetic = False
            S.intr_method = "board"
            # S.device_info is deliberately left alone: it describes which
            # physical camera THIS SESSION is currently talking to, which the
            # deleted file has no bearing on.
    payload = _plan_payload(rigplan.load_plan(S.results_dir))
    payload.update(deleted=existed, camera=camera)
    return jsonify(payload)


# ── Rig plan and status board ────────────────────────────────────────────────

def _plan_payload(p: dict) -> dict:
    rows = rigplan.roster(S.results_dir, p)
    return {"ok": True, "plan": p, "roster": rows,
            "summary": rigplan.summarise_roster(rows),
            "plan_path": str(rigplan.plan_path(S.results_dir))}


@app.route("/api/plan", methods=["GET"])
def api_plan_get():
    """The session plan plus the live status of every camera.

    The roster is rebuilt from `results/*.json` on every request rather than
    cached: the point of this page is to answer "what is still outstanding",
    and a stale answer to that question is worse than no answer.
    """
    return jsonify(_plan_payload(rigplan.load_plan(S.results_dir)))


@app.route("/api/plan", methods=["POST"])
def api_plan_post():
    data = request.get_json(force=True) or {}
    try:
        rigplan.save_plan(data, S.results_dir)
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 400
    return jsonify(_plan_payload(rigplan.load_plan(S.results_dir)))


# ── Belt map ─────────────────────────────────────────────────────────────────

@app.route("/api/beltmap", methods=["POST"])
def api_beltmap():
    """Build the top-down conveyor map from every saved calibration.

    Reads results/*.json rather than only the live session, because the map is
    inherently multi-camera: it is the artefact that shows what the separately
    calibrated cameras look like *together*, including whether they overlap at
    all — which this rig has never established.
    """
    import base64
    data = request.get_json(force=True) or {}
    width = float(data.get("belt_width_mm", 700))
    length = float(data.get("belt_length_mm", 1400))
    mmpp = float(data.get("mm_per_px", 2.0))
    margin = float(data.get("margin_mm", 50))

    cams, skipped = [], []
    for f in sorted(S.results_dir.glob("*.json")):
        rec = calstore.load(f)
        a = calstore.load_arrays(rec)
        if "K" not in a or "rvec" not in a:
            skipped.append({"camera": rec.get("camera", f.stem),
                            "why": "no extrinsics — solve the belt plane for it"})
            continue
        cams.append(bmap.CameraOnBelt(
            name=rec["camera"], K=a["K"], D=a["D"],
            rvec=a["rvec"], tvec=a["tvec"], image_size=a["image_size"]))

    if not cams:
        return jsonify({"ok": False,
                        "error": "no calibrations with extrinsics found — "
                                 "calibrate a camera and solve its belt plane first",
                        "skipped": skipped}), 400

    # A georeferenced workspace map, when one is uploaded, defines the frame
    # AND supplies the backdrop — footprints render on the actual floor plan
    # rather than a blank rectangle. This is what generalizes the tool past
    # "a belt of width x length": any rig with a mapped plane works the same
    # way. The typed belt dimensions remain the fallback, and `use_workspace:
    # false` forces them even when a map exists.
    workspace = None
    ws = wmap.load(S.results_dir)
    if ws is not None and ws[0].is_georeferenced and data.get("use_workspace", True):
        wsmap, wsimg = ws
        frame = bmap.BeltFrame(**wsmap.frame_dict())
        if mmpp and mmpp != frame.mm_per_px:
            frame.mm_per_px = mmpp
        # Cap the canvas: a building-scale map at 2 mm/px would be enormous.
        biggest = max(frame.width_px, frame.height_px)
        if biggest > 4000:
            frame.mm_per_px *= biggest / 4000
        background = wmap.background_for_frame(wsimg, wsmap, frame)
        workspace = {"source": wsmap.source, "mm_per_px": wsmap.mm_per_px,
                     "origin_px": list(wsmap.origin_px),
                     "image_size": list(wsmap.image_size)}
    else:
        frame = bmap.BeltFrame.from_belt(width, length, mm_per_px=mmpp,
                                         margin_mm=margin)
        background = np.full((frame.height_px, frame.width_px, 3), 26, np.uint8)

    # Rectify the live camera into the map if it happens to be one of these.
    images = {}
    with S.lock:
        if S.last_frame is not None and S.camera in [c.name for c in cams]:
            images[S.camera] = S.last_frame
    if images:
        mos, stats = bmap.mosaic(images, cams, frame)
        covered = np.any(mos > 0, axis=2)
        # Blend the rectified view over the backdrop rather than replacing it,
        # so the floor plan stays legible underneath the camera imagery.
        base = background.copy()
        base[covered] = (0.65 * mos[covered] + 0.35 * background[covered]
                         ).astype(np.uint8)
    else:
        base = background
    # grid_mm=None -> spacing chosen from the map extent, so a 20 m floor plan
    # does not come back as solid hatching.
    canvas = bmap.draw_overlay(base, cams, frame, grid_mm=None)

    # Mark the world origin whenever it is on the canvas — with an uploaded
    # map this is the point everything was georeferenced against, and seeing
    # it drift from where it should sit is the fastest visual sanity check.
    o = frame.mm_to_px(np.array([[0.0, 0.0]]))[0]
    if 0 <= o[0] < canvas.shape[1] and 0 <= o[1] < canvas.shape[0]:
        cv2.drawMarker(canvas, (int(o[0]), int(o[1])), (60, 60, 230),
                       cv2.MARKER_CROSS, 18, 2)

    rep = bmap.coverage_report(cams, frame)
    ok, buf = cv2.imencode(".png", canvas)
    return jsonify({
        "ok": True,
        "png": base64.b64encode(buf).decode() if ok else "",
        "frame": frame.to_dict(),
        "cameras": [c.name for c in cams],
        "skipped": skipped,
        "coverage": rep,
        "live_overlaid": list(images),
        "workspace": workspace,
    })


# ── Workspace map (generalized world plane) ──────────────────────────────────

def _workspace_payload(include_png: bool = True) -> dict:
    import base64
    ws = wmap.load(S.results_dir)
    if ws is None:
        return {"ok": True, "map": None}
    m, img = ws
    out = {"ok": True, "map": m.to_dict()}
    out["map"]["is_georeferenced"] = m.is_georeferenced
    if include_png:
        okc, buf = cv2.imencode(".png", img)
        out["png"] = base64.b64encode(buf).decode() if okc else ""
    return out


@app.route("/api/workspace", methods=["GET"])
def api_workspace_get():
    return jsonify(_workspace_payload())


@app.route("/api/workspace/upload", methods=["POST"])
def api_workspace_upload():
    """Upload the world-plane map: a top-down PNG/JPEG, or a GLB model.

    A PNG arrives unitless and must be georeferenced afterwards (origin click
    + scale). A GLB georeferences itself: glTF fixes the units at metres and
    the world origin is the model's own origin, so the render comes back
    ready to use.
    """
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"ok": False, "error": "no file in the upload"}), 400
    suffix = Path(f.filename).suffix.lower()

    if suffix == ".glb":
        tmp = wmap.workspace_dir(S.results_dir)
        tmp.mkdir(parents=True, exist_ok=True)
        raw = tmp / "original.glb"
        f.save(str(raw))
        try:
            img, m = wmap.render_glb_topdown(raw)
        except ValueError as e:
            raw.unlink(missing_ok=True)
            return jsonify({"ok": False, "error": f"GLB: {e}"}), 400
    elif suffix in (".png", ".jpg", ".jpeg"):
        buf = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"ok": False, "error": "could not decode the image"}), 400
        m = wmap.WorkspaceMap(image_size=(img.shape[1], img.shape[0]),
                              source="png",
                              notes=f"uploaded from {f.filename}")
    else:
        return jsonify({"ok": False, "error":
                        f"unsupported file type '{suffix}' — upload a "
                        f"top-down .png/.jpg, or a .glb model"}), 400

    wmap.save(m, img, S.results_dir)
    return jsonify(_workspace_payload())


@app.route("/api/workspace/georef", methods=["POST"])
def api_workspace_georef():
    """Set origin / scale / +Y axis on the uploaded map, in any order.

    - {"origin_px": [x, y]}                    click the world origin
    - {"mm_per_px": v}                         type the scale directly
    - {"scale_points": [[x,y],[x,y]],
       "distance_mm": d}                       or measure it: two clicks + tape
    - {"y_point_px": [x, y]}                   a point along world +Y from the
                                               origin; the rotation is baked
                                               into the stored image
    """
    data = request.get_json(force=True) or {}
    ws = wmap.load(S.results_dir)
    if ws is None:
        return jsonify({"ok": False, "error": "upload a map first"}), 400
    m, img = ws

    try:
        if "scale_points" in data:
            p1, p2 = data["scale_points"]
            m.mm_per_px = wmap.scale_from_points(p1, p2,
                                                 float(data["distance_mm"]))
        elif "mm_per_px" in data:
            v = float(data["mm_per_px"])
            if v <= 0:
                raise ValueError("mm_per_px must be positive")
            m.mm_per_px = v

        if "origin_px" in data:
            ox, oy = (float(v) for v in data["origin_px"])
            w, h = m.image_size
            if not (0 <= ox < w and 0 <= oy < h):
                raise ValueError("origin must lie inside the map image")
            m.origin_px = (ox, oy)

        if "y_point_px" in data:
            if m.origin_px is None:
                raise ValueError("set the origin before the +Y direction — "
                                 "the axis is defined relative to it")
            img, m.origin_px = wmap.rotate_to_axis(img, m.origin_px,
                                                   data["y_point_px"])
            m.image_size = (img.shape[1], img.shape[0])
            wmap.save(m, img, S.results_dir)
            return jsonify(_workspace_payload())
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    wmap.save_meta(m, S.results_dir)
    return jsonify(_workspace_payload())


@app.route("/api/workspace", methods=["DELETE"])
def api_workspace_delete():
    existed = wmap.delete(S.results_dir)
    return jsonify({"ok": True, "deleted": existed})


# ── Point-correspondence extrinsics (one map + one frame per camera) ─────────

@app.route("/api/correspondence/frame", methods=["POST"])
def api_correspondence_frame():
    """Grab the current live frame to click correspondences on.

    Frozen deliberately: the operator needs a still image to click accurately,
    and the MJPEG preview is moving. Stored per camera under the workspace
    directory so a half-finished job survives a page reload.
    """
    import base64
    with S.lock:
        frame, camera = S.last_frame, S.camera
    if frame is None:
        return jsonify({"ok": False, "error": "no frame yet — start a session "
                                              "first"}), 400
    d = wmap.workspace_dir(S.results_dir) / "frames"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{camera}.png"
    cv2.imwrite(str(path), frame)
    ok, buf = cv2.imencode(".png", frame)
    return jsonify({"ok": True, "camera": camera,
                    "image_size": [frame.shape[1], frame.shape[0]],
                    "png": base64.b64encode(buf).decode() if ok else ""})


@app.route("/api/correspondence/frame/<camera>", methods=["GET"])
def api_correspondence_frame_get(camera):
    import base64
    if not _CAMERA_NAME_RE.match(camera):
        return jsonify({"ok": False, "error": "invalid camera name"}), 400
    path = wmap.workspace_dir(S.results_dir) / "frames" / f"{camera}.png"
    if not path.exists():
        return jsonify({"ok": True, "png": None})
    img = cv2.imread(str(path))
    ok, buf = cv2.imencode(".png", img)
    return jsonify({"ok": True, "camera": camera,
                    "image_size": [img.shape[1], img.shape[0]],
                    "png": base64.b64encode(buf).decode() if ok else ""})


@app.route("/api/correspondence/solve", methods=["POST"])
def api_correspondence_solve():
    """Solve one camera's plane mapping from clicked image<->map point pairs.

    This is the board-free route: one frame per camera and one georeferenced
    map, no ChArUco on the floor and no second rig visit. World points arrive
    already converted to millimetres by the client, which holds the map's
    georeference.

    Intrinsics are used when the camera has them — they undistort the clicked
    points and let the homography be decomposed into a real camera pose — and
    their absence is recorded rather than worked around.
    """
    data = request.get_json(force=True) or {}
    camera = (data.get("camera") or "").strip()
    if not _CAMERA_NAME_RE.match(camera or ""):
        return jsonify({"ok": False, "error": "invalid or missing camera name"}), 400
    pairs = data.get("pairs") or []
    try:
        img_pts = [p["image"] for p in pairs]
        wld_pts = [p["world"] for p in pairs]
    except (KeyError, TypeError):
        return jsonify({"ok": False, "error":
                        "each pair needs an 'image' and a 'world' point"}), 400

    # Reuse the camera's saved intrinsics when it has them. Read from disk
    # rather than the live session so this works for any camera, not only the
    # one currently streaming.
    K = D = None
    intr_record = None
    image_size = data.get("image_size")
    path = S.results_dir / f"{camera}.json"
    if path.exists():
        try:
            rec = calstore.load(path)
            if calstore.is_synthetic(rec) and S.source_id not in (None, "synthetic"):
                return jsonify({"ok": False, "error":
                                f"the saved intrinsics for {camera} came from "
                                f"the SYNTHETIC camera — clear them before "
                                f"solving against real imagery"}), 400
            arr = calstore.load_arrays(rec)
            if "K" in arr:
                K, D = arr["K"], arr["D"]
                intr_record = rec.get("intrinsics")
                if image_size is None:
                    image_size = list(arr.get("image_size", (0, 0)))
        except Exception as e:
            return jsonify({"ok": False,
                            "error": f"could not read {path.name}: {e}"}), 400
    if image_size is None:
        return jsonify({"ok": False, "error": "image_size is required when the "
                                              "camera has no saved intrinsics"}), 400

    # Intrinsics are resolution-specific; clicking on a frame of a different
    # size than they were measured at would undistort against the wrong model.
    if K is not None and intr_record:
        stored = tuple(intr_record.get("image_size", ()))
        if stored and tuple(image_size) != stored:
            return jsonify({"ok": False, "error":
                            f"intrinsics were measured at {stored[0]}x{stored[1]} "
                            f"but this frame is {image_size[0]}x{image_size[1]} — "
                            f"grab the frame at the calibrated resolution"}), 400

    try:
        res = corr.solve(img_pts, wld_pts, camera=camera, K=K, D=D,
                         ransac_reproj_mm=data.get("tolerance_mm"))
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    with S.lock:
        S.corr_result = res
        S.corr_image_size = tuple(image_size)
        S.corr_intr_record = intr_record

    spread = corr.spread_score(wld_pts)
    return jsonify({
        "ok": True, "camera": camera,
        "rms_mm": res.rms_mm, "max_mm": res.max_mm,
        "n_points": len(res.image_points),
        "n_inliers": int(res.inliers.sum()),
        "residuals_mm": res.residuals_mm.tolist(),
        "inliers": res.inliers.astype(int).tolist(),
        "used_intrinsics": res.used_intrinsics,
        "camera_position_mm": res.camera_position_mm.tolist(),
        "warnings": res.warnings,
        "spread": spread,
    })


@app.route("/api/correspondence/validate", methods=["POST"])
def api_correspondence_validate():
    """Click a pixel on the frozen frame, get world millimetres back.

    The check CalibrationHub's manual asks for, against the solve currently
    in the session — click somewhere you can identify on the map and see
    whether the answer lands there.
    """
    data = request.get_json(force=True) or {}
    with S.lock:
        res = S.corr_result
    if res is None:
        return jsonify({"ok": False, "error": "solve the correspondences first"}), 400
    pt = np.array([[float(data["x"]), float(data["y"])]], np.float32)
    if res.used_intrinsics:
        path = S.results_dir / f"{res.camera}.json"
        arr = calstore.load_arrays(calstore.load(path))
        pt = cv2.undistortPoints(pt.reshape(-1, 1, 2), arr["K"], arr["D"],
                                 P=arr["K"]).reshape(-1, 2)
    w = cv2.perspectiveTransform(pt.reshape(-1, 1, 2).astype(np.float64),
                                 res.H_image_to_world).reshape(-1, 2)[0]
    return jsonify({"ok": True, "x_mm": float(w[0]), "y_mm": float(w[1])})


@app.route("/api/correspondence/save", methods=["POST"])
def api_correspondence_save():
    """Persist the current correspondence solve as this camera's extrinsics."""
    data = request.get_json(force=True) or {}
    with S.lock:
        res = S.corr_result
        image_size = S.corr_image_size
        intr_record = S.corr_intr_record
    if res is None:
        return jsonify({"ok": False, "error": "nothing solved to save"}), 400

    rec = calstore.build_correspondence_record(
        camera=res.camera, res=res, image_size=image_size,
        intr_record=intr_record, source=S.source_id or "",
        notes=data.get("notes", ""))
    rec = calstore.carry_forward_device(rec, S.results_dir)
    path = calstore.save(rec, S.results_dir)
    return jsonify({"ok": True, "path": str(path),
                    **_plan_payload(rigplan.load_plan(S.results_dir))})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    args = ap.parse_args()
    S.results_dir = Path(args.results_dir)

    print("=" * 70)
    print("  OVGU AMS — camera calibration")
    print("=" * 70)
    print(f"  open   http://{args.host}:{args.port}")
    print(f"  results -> {S.results_dir}")
    print("  no hardware? choose the synthetic source to exercise the whole flow")
    print("=" * 70)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
