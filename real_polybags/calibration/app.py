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
        self.extr_result = None
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

    with S.lock:
        S.close()
        S.spec = board_mod.PRESETS[preset]
        S.camera = camera
        S.shots, S.thumbs = [], []
        S.intr_result = S.extr_result = None
        S.intr_synthetic = False
        try:
            if src_id == "synthetic":
                S.source = src_mod.SyntheticSource(S.spec)
            elif src_id == "folder":
                S.source = src_mod.FolderSource(data.get("folder", ""))
            else:
                S.source = src_mod.AVAILABLE[src_id]()
            S.source_id = src_id
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
                loaded = {"fx": r.fx, "fy": r.fy, "cx": r.cx, "cy": r.cy,
                          "rms": r.rms, "image_size": list(stored_size),
                          "created": rec.get("created_utc"),
                          "saved_source": rec.get("source"),
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
                    "factory_intrinsics": ref, "loaded_intrinsics": loaded})


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
            notes=data.get("notes", ""))
        # Saving rewrites the whole per-camera file. Intrinsics and extrinsics
        # are measured in separate sittings, so a bench save must not delete a
        # pose solved earlier at the rig.
        rec = calstore.carry_forward_extrinsics(rec, S.results_dir)
        path = calstore.save(rec, S.results_dir)
    return jsonify({"ok": True, "path": str(path)})


@app.route("/api/results")
def api_results():
    return jsonify({"summary": calstore.summarise(S.results_dir)})


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

    frame = bmap.BeltFrame.from_belt(width, length, mm_per_px=mmpp, margin_mm=margin)

    # Rectify the live camera into the map if it happens to be one of these.
    images = {}
    with S.lock:
        if S.last_frame is not None and S.camera in [c.name for c in cams]:
            images[S.camera] = S.last_frame
    base = (bmap.mosaic(images, cams, frame)[0] if images
            else np.full((frame.height_px, frame.width_px, 3), 26, np.uint8))
    canvas = bmap.draw_overlay(base, cams, frame, grid_mm=100.0)

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
    })


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
