"""
Rig-session plan and extrinsics status board — OVGU AMS calibration tool.

Intrinsics are bench work; extrinsics are rig work, and the rig is available for
a small fraction of the time a lens is. What makes a lab session succeed is
knowing *before* walking up to the conveyor which cameras still need a pose,
which numbers have to be taken with a tape measure while standing there, and
what the answers ought to look like.

That last part is the point of this module. A solved pose is easy to look at and
hard to judge: 298 mm above the belt is a perfectly plausible-looking number
until you remember the camera is mounted 1.4 m up. Recording the *anticipated*
measurement up front turns the solve into a comparison — measured against
expected — which is the difference between inspecting a result and checking it.

The plan is deliberately NOT stored as `results/<something>.json`. Both the belt
map and the summary table enumerate calibrations with
`results_dir.glob("*.json")`, so a planning file dropped beside them would be
picked up and parsed as a camera record. It lives one directory down, in
`results/_plan/`, where that glob cannot reach it.

**Camera names are load-bearing.** A calibration is joined to detections by name,
so `results/<name>.json` must use the same `<name>` the recorder writes its video
under — `basler_1`, `rgbd_2_color`, and so on. A calibration filed under a name
nothing else uses is invisible to everything downstream, and nothing will report
that: it simply never matches.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import store

SCHEMA_VERSION = 1

# Named to match what `record_all_5_cameras_macos.py` writes its streams as, so
# a saved calibration lands under a name the rest of the pipeline already uses.
# Two RealSense units are recorded, hence two rgbd entries; the roster is
# editable because a rig is not a constant.
DEFAULT_RIG_CAMERAS = ["basler_1", "basler_2", "lucid",
                       "rgbd_1_color", "rgbd_2_color"]

DEFAULT_BELT = {"width_mm": 700.0, "length_mm": 1400.0}

# Tolerances for checking a solve against its anticipated value. Height is a
# tape measurement against a mount, so a few percent is expected and only a
# gross disagreement is meaningful — the failure this catches (wrong board
# preset, or a print scaled by "fit to page") is off by tens of percent, not by
# a few millimetres.
HEIGHT_OK_FRAC = 0.05
HEIGHT_WARN_FRAC = 0.15

# Extrinsic reprojection thresholds, matching the guidance in the manual
# ("under about 1 px is good").
REPROJ_OK_PX = 1.0
REPROJ_WARN_PX = 2.0

# An origin offset is typed in, not measured by the solver, so this only checks
# that what was entered is what was planned. A millimetre of slack absorbs
# float formatting, nothing more.
OFFSET_TOL_MM = 1.0

PLAN_DIRNAME = "_plan"
PLAN_FILENAME = "rig_plan.json"


# ── Storage ──────────────────────────────────────────────────────────────────

def plan_path(results_dir: Path) -> Path:
    return Path(results_dir) / PLAN_DIRNAME / PLAN_FILENAME


def blank_camera(name: str) -> dict:
    return {
        "name": name,
        "expected_height_mm": None,
        "origin_offset_mm": [0.0, 0.0],
        "offset_measured": False,
        "note": "",
    }


def default_plan() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_utc": None,
        "method": "A",
        "belt": dict(DEFAULT_BELT),
        "cameras": [blank_camera(n) for n in DEFAULT_RIG_CAMERAS],
    }


def load_plan(results_dir: Path) -> dict:
    """The saved plan, or a fresh default one. Never raises on a missing file.

    A malformed plan falls back to the default rather than taking the page down
    with it: the plan is an aid, and losing it must not block a rig session that
    is already underway.
    """
    path = plan_path(results_dir)
    if not path.exists():
        return default_plan()
    try:
        raw = json.loads(path.read_text())
    except Exception as e:
        p = default_plan()
        p["load_error"] = f"could not read {path.name}: {e}"
        return p
    return normalise_plan(raw)


def normalise_plan(raw: dict) -> dict:
    """Coerce whatever came from disk or the browser into the full shape.

    Every field is defaulted rather than required, so a plan written by an older
    version — or a partial one posted by the page — stays usable.
    """
    base = default_plan()
    out = {
        "schema_version": SCHEMA_VERSION,
        "updated_utc": raw.get("updated_utc"),
        "method": "B" if str(raw.get("method", "A")).upper() == "B" else "A",
        "belt": {
            "width_mm": _num(raw.get("belt", {}).get("width_mm"),
                             DEFAULT_BELT["width_mm"]),
            "length_mm": _num(raw.get("belt", {}).get("length_mm"),
                              DEFAULT_BELT["length_mm"]),
        },
        "cameras": [],
    }
    seen = set()
    for c in raw.get("cameras") or []:
        name = str(c.get("name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        off = c.get("origin_offset_mm") or [0.0, 0.0]
        out["cameras"].append({
            "name": name,
            "expected_height_mm": _num_or_none(c.get("expected_height_mm")),
            "origin_offset_mm": [_num(off[0] if len(off) > 0 else 0, 0.0),
                                 _num(off[1] if len(off) > 1 else 0, 0.0)],
            "offset_measured": bool(c.get("offset_measured", False)),
            "note": str(c.get("note", "")),
        })
    if not out["cameras"]:
        out["cameras"] = base["cameras"]
    return out


def save_plan(plan: dict, results_dir: Path) -> Path:
    plan = normalise_plan(plan)
    plan["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = plan_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path


def camera_plan(plan: dict, name: str) -> dict | None:
    for c in plan.get("cameras", []):
        if c["name"] == name:
            return c
    return None


def _num(v, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f else default          # reject NaN


def _num_or_none(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


# ── Status board ─────────────────────────────────────────────────────────────

def _read_records(results_dir: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(Path(results_dir).glob("*.json")):
        try:
            rec = store.load(f)
        except Exception as e:
            out[f.stem] = {"camera": f.stem, "_error": str(e)}
            continue
        out[rec.get("camera", f.stem)] = rec
    return out


def _state_of(rec: dict | None) -> str:
    """Where a camera stands, in one word.

    - `blocked`     nothing saved, or intrinsics missing: cannot solve a pose
    - `ready`       intrinsics in hand, no pose yet — this is the rig work
    - `provisional` a tape-measured homography: usable, but no lens correction
    - `solved`      full board extrinsics
    """
    if rec is None or "_error" in rec:
        return "blocked"
    ext = rec.get("extrinsics") or {}
    has_intr = "K" in (rec.get("intrinsics") or {})
    if "rvec" in ext:
        return "solved"
    if ext:
        return "provisional"
    return "ready" if has_intr else "blocked"


def _pending_for(state: str, rec: dict | None, cplan: dict, method: str) -> list[str]:
    """What still has to happen for this camera, in the order it happens."""
    todo = []
    ext = (rec or {}).get("extrinsics") or {}
    if rec and "_error" not in rec and store.is_synthetic(rec):
        todo.append("measured from the SYNTHETIC camera, not this one — a "
                    "rehearsal result. Delete it and recalibrate before the "
                    "rig session, or the real pose is solved against a lens "
                    "model that belongs to no lens")
    if state == "solved" and ext.get("stale"):
        todo.append("pose was carried over from an earlier save and solved "
                    "against different intrinsics — re-solve it")
    if state == "blocked":
        todo.append("intrinsics not measured — bench work, no rig access needed")
    if state in ("blocked", "ready", "provisional"):
        if state == "provisional":
            todo.append("only a tape homography — re-solve against the board "
                        "for a distortion-corrected pose")
        else:
            todo.append("belt pose not solved")
        if method == "B" and not cplan.get("offset_measured"):
            todo.append("origin offset not measured — tape from the origin "
                        "placement (X across the belt, Y along travel)")
        if cplan.get("expected_height_mm") is None:
            todo.append("expected height not recorded — without it the solved "
                        "height cannot be checked")
    return todo


def roster(results_dir: Path, plan: dict) -> list[dict]:
    """One row per camera: what is saved, what is planned, what is outstanding.

    Cameras come from the plan, plus any camera that has a saved calibration but
    is not in the plan — so a result can never be invisible here just because
    somebody forgot to list it.
    """
    records = _read_records(results_dir)
    method = plan.get("method", "A")
    names = [c["name"] for c in plan.get("cameras", [])]
    names += [n for n in sorted(records) if n not in names]

    rows = []
    for name in names:
        rec = records.get(name)
        cplan = camera_plan(plan, name) or blank_camera(name)
        state = _state_of(rec)
        intr = (rec or {}).get("intrinsics") or {}
        ext = (rec or {}).get("extrinsics") or {}
        rows.append({
            "name": name,
            "in_plan": camera_plan(plan, name) is not None,
            "state": state,
            "error": (rec or {}).get("_error"),
            "method": store.method_of(rec) if rec and "_error" not in rec else None,
            "synthetic": bool(rec and "_error" not in rec and store.is_synthetic(rec)),
            "saved_utc": (rec or {}).get("created_utc"),
            "has_intrinsics": "K" in intr,
            "image_size": intr.get("image_size"),
            "intrinsics_rms_px": intr.get("rms_px"),
            "height_above_belt_mm": ext.get("height_above_belt_mm"),
            "extr_error_px": ext.get("reproj_error_px"),
            "solved_offset_mm": ext.get("board_origin_offset_mm"),
            "plan": cplan,
            "pending": _pending_for(state, rec, cplan, method),
        })
    return rows


def summarise_roster(rows: list[dict]) -> dict:
    counts = {"total": len(rows), "solved": 0, "ready": 0,
              "provisional": 0, "blocked": 0}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    counts["headline"] = (
        f"{counts['solved']} of {counts['total']} cameras solved · "
        f"{counts['ready']} ready to solve now")
    return counts


# ── Checking a solve against what was anticipated ────────────────────────────

def _fmt(v, unit: str, digits: int = 0) -> str:
    return "—" if v is None else f"{v:.{digits}f} {unit}"


def verify(cplan: dict, height_mm: float, reproj_px: float,
           offset_used_mm: tuple[float, float]) -> list[dict]:
    """Compare a fresh solve against the plan. Returns one row per check.

    A check whose expectation was never recorded comes back as `unset` rather
    than passing silently — an unmade comparison should not look like a
    successful one.
    """
    checks = []

    exp_h = cplan.get("expected_height_mm")
    if exp_h is None:
        checks.append({
            "key": "height", "label": "height above belt",
            "measured": _fmt(height_mm, "mm"), "expected": "not recorded",
            "delta": "", "status": "unset",
            "hint": "Record the expected height in the plan and re-solve to "
                    "have this checked.",
        })
    else:
        rel = (height_mm - exp_h) / exp_h if exp_h else 0.0
        if abs(rel) <= HEIGHT_OK_FRAC:
            status, hint = "ok", ""
        elif abs(rel) <= HEIGHT_WARN_FRAC:
            status, hint = "warn", ("Off by more than a tape measure explains — "
                                    "check the expected value before the solve.")
        else:
            status, hint = "bad", (
                "Far from the expected height. The usual cause is a board "
                "mismatch: the wrong preset selected, or a print scaled by "
                "'fit to page'. Every distance is then wrong by that factor "
                "and nothing downstream can detect it.")
        checks.append({
            "key": "height", "label": "height above belt",
            "measured": _fmt(height_mm, "mm"), "expected": _fmt(exp_h, "mm"),
            "delta": f"{rel * 100:+.1f}%", "status": status, "hint": hint,
        })

    if reproj_px <= REPROJ_OK_PX:
        status, hint = "ok", ""
    elif reproj_px <= REPROJ_WARN_PX:
        status, hint = "warn", "Acceptable, but the board may not be lying flat."
    else:
        status, hint = "bad", ("The board is probably not flat on the belt, or "
                               "the intrinsics being used are wrong.")
    checks.append({
        "key": "reproj", "label": "reprojection error",
        "measured": f"{reproj_px:.3f} px",
        "expected": f"< {REPROJ_OK_PX:.0f} px",
        "delta": "", "status": status, "hint": hint,
    })

    planned = cplan.get("origin_offset_mm") or [0.0, 0.0]
    used = list(offset_used_mm)
    drift = max(abs(used[0] - planned[0]), abs(used[1] - planned[1]))
    if drift <= OFFSET_TOL_MM:
        status, hint = "ok", ""
    else:
        status, hint = "warn", (
            "The offset solved with is not the one in the plan. Whichever is "
            "right, they disagree — and this is what puts the camera in the "
            "shared belt frame.")
    checks.append({
        "key": "offset", "label": "origin offset",
        "measured": f"{used[0]:.0f}, {used[1]:.0f} mm",
        "expected": f"{planned[0]:.0f}, {planned[1]:.0f} mm",
        "delta": "", "status": status, "hint": hint,
    })

    if not cplan.get("offset_measured") and any(abs(v) > OFFSET_TOL_MM for v in used):
        checks.append({
            "key": "offset_source", "label": "offset provenance",
            "measured": "not marked as measured", "expected": "tape measured",
            "delta": "", "status": "warn",
            "hint": "A non-zero offset that was never measured is a guess, and "
                    "it displaces this camera relative to every other one.",
        })
    return checks


def worst_status(checks: list[dict]) -> str:
    for s in ("bad", "warn", "unset", "ok"):
        if any(c["status"] == s for c in checks):
            return s
    return "ok"
