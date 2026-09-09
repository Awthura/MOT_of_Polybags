#!/usr/bin/env python3
"""
Streamer — OVGU AMS algorithm phase (the producer).

Plays the session's four synchronous camera videos, runs YOLO on each frame,
projects every detection's foot-point onto the belt with the stored
calibration, and:

  * publishes all cameras' positions to MQTT `/polybags`, and
  * serves each camera's annotated frame as an MJPEG feed at `/stream/<cam>`.

The digital-twin dashboard is a separate, decoupled consumer that subscribes to
the same MQTT topic — this process only *produces*. It also serves the static
dashboard and the combined layout page for convenience.

Run:  python3 algorithm/streamer/app.py
(Start the broker first:  mosquitto -c algorithm/mosquitto.conf)
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_from_directory

ALGO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ALGO_DIR / "core"))

import appconfig            # noqa: E402
import calib                # noqa: E402
import georef               # noqa: E402
from detector import Detector, draw_detections   # noqa: E402
from fusion import FusionTracker                  # noqa: E402
from publisher import Publisher                   # noqa: E402
from track import Tracker                         # noqa: E402

CAM_COLORS = {                    # BGR, kept in sync with the dashboard palette
    "basler_1": (60, 60, 235),
    "basler_2": (60, 210, 60),
    "lucid": (235, 180, 40),
    "rgbd_1_color": (200, 40, 200),
}

# Our lowercase camera keys -> the CamelCase names the recorder used in the
# timestamp CSV filenames (timestamps_<Name>_<session>.csv).
CSV_CAM_NAME = {
    "basler_1": "Basler_1", "basler_2": "Basler_2",
    "lucid": "Lucid", "rgbd_1_color": "RGBD_1", "rgbd_2_color": "RGBD_2",
}


def load_frame_times(video_dir: Path, name: str, session: str):
    """Per-frame unix_time array from timestamps_<Cam>_<session>.csv, or None.

    These are the camera's own device timestamps, the only thing that makes a
    real (fusion-grade) cross-camera alignment possible. Recordings made before
    the recorder logged them have no CSV -> None -> caller falls back to the
    approximate normalized timeline.
    """
    import csv as _csv
    p = video_dir / f"timestamps_{CSV_CAM_NAME.get(name, name)}_{session}.csv"
    if not p.exists():
        return None
    times = []
    with open(p, newline="") as fh:
        for row in _csv.DictReader(fh):
            times.append(float(row["unix_time"]))
    return np.asarray(times, float) if times else None


class Stream:
    """One camera: its frames (pre-loaded for random access), calibration, tracker.

    The recordings are tagged a nominal 15 fps but were captured at different
    real rates, and each camera has a different frame count for the same event.
    To play them in sync we address frames by **normalized progress** p in
    [0,1): frame p*N maps to the same real instant p*D in every camera (they
    started and stopped together, so share the event duration D). That needs
    random access, but these MJPEG AVIs refuse to seek — so we decode every
    frame once at start-up and keep them as JPEG bytes (~100 KB each), decoding
    on demand. A whole session is a few hundred frames, well within memory.
    """

    def __init__(self, name: str, path: Path, cam: "calib.Camera", tracker: Tracker):
        self.name = name
        self.cam = cam
        self.tracker = tracker
        self.jpegs: list[bytes] = []
        c = cv2.VideoCapture(str(path))
        while True:
            ok, f = c.read()
            if not ok:
                break
            ok2, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok2:
                self.jpegs.append(buf.tobytes())
        c.release()
        self.count = len(self.jpegs)
        self.times = None          # per-frame time in seconds; set by the engine

    def set_times(self, times) -> None:
        self.times = np.asarray(times, float)[:self.count]

    def frame_at_time(self, t: float):
        """Decoded frame whose timestamp is nearest t (seconds), and its index.

        `self.times` is sorted, so a binary search finds the bracketing frames
        and we keep the closer one. This is the sync primitive: feed the same t
        to every camera and each returns its frame at that shared instant.
        """
        if self.count == 0:
            return None, -1
        i = int(np.searchsorted(self.times, t))
        if i <= 0:
            i = 0
        elif i >= self.count:
            i = self.count - 1
        elif abs(self.times[i - 1] - t) <= abs(self.times[i] - t):
            i -= 1
        f = cv2.imdecode(np.frombuffer(self.jpegs[i], np.uint8), cv2.IMREAD_COLOR)
        return f, i


class Engine:
    """Background loop: read → detect → project → publish, one tick per frame."""

    def __init__(self, cfg):
        self.cfg = cfg
        # Resolve the active scenario (video source + session + cameras).
        sc_name = cfg.get("scenario")
        scenarios = cfg.get("scenarios") or {}
        sc = scenarios.get(sc_name)
        if sc is None:
            raise ValueError(f"unknown scenario {sc_name!r}; "
                             f"choices: {list(scenarios)}")
        self.scenario = sc_name
        self.scenario_label = sc.get("label", sc_name)
        self.session = sc["session"]
        self.cameras = sc["cameras"]
        self.video_dir = _resolve(cfg, sc["video_dir"])
        print(f"[engine] scenario '{sc_name}' ({self.scenario_label}) "
              f"session={self.session}")
        self.fps = float(cfg["playback"]["fps"])
        self.georef, self.map_png = _display_map(cfg)

        d = cfg["detector"]
        # `model` is a key into `models` (seg|detect), or a literal .pt path.
        mkey = d.get("model") or d.get("model_path")
        mpath = (d.get("models") or {}).get(mkey, mkey)
        print(f"[engine] detector: {mkey}  ({mpath})")
        self.detector = Detector(_resolve(cfg, mpath),
                                 d["imgsz"], d["conf"], d["iou"], d["device"])
        # Which pixel of a detection is projected onto the belt: bbox centre
        # (default) or foot. Centre is steadier for flat bags than the bottom,
        # which carries view-dependent parallax.
        self.ref_point = d.get("reference_point", "center")
        self.draw_masks = bool(d.get("draw_masks", True))   # False = plain bboxes

        # Conveyor bounds (world mm): projections outside are dropped as off-belt.
        b = cfg.get("belt_bounds") or {}
        self.belt_x = b.get("x_mm", [-1e9, 1e9])
        self.belt_y = b.get("y_mm", [-1e9, 1e9])

        results_dir = cfg.path("results_dir")
        m = cfg["mqtt"]
        self.streams: dict[str, Stream] = {}
        for name in self.cameras:
            # Video files are named to match the calibration, so each camera
            # simply uses its own results/<name>.json — no swaps to worry about.
            try:
                cam = calib.load_camera(name, results_dir)
            except (FileNotFoundError, ValueError) as e:
                print(f"[engine] skipping {name}: {e}")
                continue
            vid = self.video_dir / f"{name}_1280x720_{self.session}.avi"
            if not vid.exists():
                print(f"[engine] no video for {name}: {vid.name}")
                continue
            self.streams[name] = Stream(name, vid, cam, Tracker(ttl_s=m["track_ttl_s"]))
        for n, st in self.streams.items():
            print(f"[engine] {n}: {st.count} frames")
        print(f"[engine] streaming {list(self.streams)}")

        # Shared playback timeline. All cameras share the event duration D (they
        # started/stopped together), so playing over a common period keeps them
        # synced: at progress p, camera i shows frame p*N_i = real instant p*D.
        # period = longest camera / event_fps gives ~real-time; `speed` scales it.
        pb = cfg["playback"]
        self.event_fps = float(pb.get("event_fps", 15))
        self.speed = float(pb.get("speed", 1.0))
        self.sync_offset = pb.get("sync_offset") or {}   # per-camera seconds

        # Prefer real device timestamps (true, fusion-grade sync). Only if every
        # camera has a CSV; otherwise fall back to the normalized timeline, which
        # assumes the cameras started/stopped together and paced uniformly.
        raw = {}
        have_all = bool(self.streams)
        for name, st in self.streams.items():
            t = load_frame_times(self.video_dir, name, self.session)
            if t is None or len(t) < st.count:
                have_all = False
                break
            raw[name] = t[:st.count]
        if have_all:
            t0 = min(float(v[0]) for v in raw.values())
            for name, st in self.streams.items():
                st.set_times(raw[name] - t0)          # seconds since global start
            self.duration = max(float(st.times[-1]) for st in self.streams.values())
            self.sync_mode = "timestamp"
        else:
            maxc = max((st.count for st in self.streams.values()), default=1)
            self.duration = max(maxc / self.event_fps, 1e-3)
            for st in self.streams.values():
                st.set_times(np.arange(st.count) * self.duration / max(st.count, 1))
            self.sync_mode = "normalized"
        print(f"[engine] sync mode: {self.sync_mode}  (loop {self.duration:.1f}s)")
        self._t_start = time.time()

        self.pub = Publisher(m["host"], m["port"], m["topic"])
        self.pub.connect()

        # Fusion layer: dedup overlapping cameras + global re-ID across the gap.
        fz = cfg.get("fusion") or {}
        self.fusion_enabled = bool(fz.get("enabled", True))
        self.fused_topic = m.get("fused_topic", m["topic"] + "_fused")
        self.fusion = FusionTracker(**{k: fz[k] for k in (
            "dedup_gate_mm", "assoc_gate_mm", "fifo_window_s", "set2_entry_y_mm",
            "min_hits", "active_ttl_s", "departed_ttl_s") if k in fz})

        self._latest: dict[str, bytes] = {}   # latest annotated JPEG per camera
        self._polybags: list[dict] = []       # latest RAW per-camera detections
        self._fused: list[dict] = []          # latest FUSED objects (global IDs)
        self._lock = threading.Lock()
        self._running = False
        self._tick = 0
        # MJPEG re-send rate for the feeds. The frames themselves are produced by
        # the inference loop (overlay drawn on the inferred frame), so the feed's
        # real update rate is the inference rate; this just paces the HTTP stream.
        self.display_fps = float(pb.get("display_fps", 15))
        # Publish on a fixed cadence, decoupled from the (irregular, inference-
        # bound) frame loop, so the dashboard receives a steady stream and dots
        # never expire in the gap between two slow ticks. Holding the last known
        # positions between inferences keeps the map stable rather than flickering.
        self.publish_hz = float(m.get("publish_hz", 6))

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._publish_loop, daemon=True).start()

    def _clock(self, name: str) -> float:
        """This camera's position on the shared timeline right now (seconds)."""
        base = (time.time() - self._t_start) * self.speed
        return (base + self.sync_offset.get(name, 0.0)) % self.duration

    def _publish_loop(self):
        interval = 1.0 / self.publish_hz
        while self._running:
            with self._lock:
                polybags = list(self._polybags)
                fused = list(self._fused)
            t_ms = int(time.time() * 1000)
            self.pub.publish({"t": t_ms, "frame": self._tick,
                              "session": self.session, "polybags": polybags})
            if self.fusion_enabled:
                self.pub.publish({"t": t_ms, "frame": self._tick,
                                  "session": self.session, "polybags": fused},
                                 topic=self.fused_topic)
            time.sleep(interval)

    def stop(self):
        self._running = False

    def latest_jpeg(self, name: str) -> bytes | None:
        with self._lock:
            return self._latest.get(name)

    def _loop(self):
        interval = 1.0 / self.fps
        while self._running:
            t0 = time.time()
            polybags = []
            for name, st in self.streams.items():
                frame, _idx = st.frame_at_time(self._clock(name))
                if frame is None:
                    continue
                boxes = self.detector.detect(frame)
                refs = [b.center_px() if self.ref_point == "center" else b.foot_px()
                        for b in boxes]
                world = st.cam.feet_to_world(refs) if refs else []
                # Drop anything projected off the conveyor before it is tracked,
                # published, or drawn — off-belt points are almost all noise.
                keep = [(b, r, (float(w[0]), float(w[1])))
                        for b, r, w in zip(boxes, refs, world)
                        if self.belt_x[0] <= w[0] <= self.belt_x[1]
                        and self.belt_y[0] <= w[1] <= self.belt_y[1]]
                tagged = st.tracker.update(
                    [(w[0], w[1], b.conf) for b, r, w in keep], t0)

                # Draw on the SAME frame we inferred -> the overlay always sits
                # on the bag, even at high speed (no display/inference lag).
                vis = draw_detections(frame, [b for b, r, w in keep],
                                      color=CAM_COLORS.get(name, (60, 220, 60)),
                                      ref=self.ref_point, masks=self.draw_masks)
                for (tid, x, y, c), (b, (rx, ry), w) in zip(tagged, keep):
                    cv2.putText(vis, f"#{tid}", (int(rx) + 6, int(ry) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    polybags.append({
                        "cam": name, "id": int(tid),
                        "x_mm": round(float(x), 1), "y_mm": round(float(y), 1),
                        "conf": round(float(c), 2), "metric": st.cam.metric})
                ok_enc, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok_enc:
                    with self._lock:
                        self._latest[name] = buf.tobytes()

            # Fuse this tick's raw detections into global objects (dedup + re-ID).
            fused = self.fusion.update(polybags, t0) if self.fusion_enabled else []
            # Hand both to the steady publisher; do not publish here (that would
            # inherit this loop's irregular timing).
            with self._lock:
                self._polybags = polybags
                self._fused = fused
            self._tick += 1
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)


def _resolve(cfg, rel: str) -> Path:
    """Resolve a config path (relative to algorithm/) to an absolute path."""
    return (ALGO_DIR / rel).resolve()


def _display_map(cfg):
    """The map + georef the dashboard draws on: (georef_dict, map_png_path).

    `display.map == "dashboard"` uses the clean synthetic belt with a
    by-construction georef; anything else falls back to the scan map and its
    stored meta.json.
    """
    disp = cfg.get("display") or {}
    if disp.get("map", "workspace") == "dashboard":
        mp = _resolve(cfg, disp["map_png"])
        im = cv2.imread(str(mp))
        h, w = im.shape[:2]
        gr = {"mm_per_px": float(disp["mm_per_px"]),
              "origin_px": [float(disp["origin_px"][0]), float(disp["origin_px"][1])],
              "image_size": [w, h], "notes": "clean synthetic belt (dashboard map)"}
        return gr, mp
    return georef.load_georef(cfg.path("workspace_meta")), cfg.path("map_png")


# ── Flask app ────────────────────────────────────────────────────────────────

def create_app(cfg) -> Flask:
    app = Flask(__name__)
    engine = Engine(cfg)
    engine.start()

    streamer_static = ALGO_DIR / "streamer" / "static"
    dashboard_dir = ALGO_DIR / "dashboard"
    map_png = engine.map_png

    @app.route("/")
    def index():
        if (streamer_static / "combined.html").exists():
            return send_from_directory(streamer_static, "combined.html")
        return ("<h1>AMS streamer</h1><p>Feeds at /stream/&lt;cam&gt;, "
                "dashboard at <a href='/dashboard/'>/dashboard/</a>.</p>")

    @app.route("/stream/<cam>")
    def stream(cam):
        def gen():
            while True:
                jpg = engine.latest_jpeg(cam)
                if jpg is None:
                    time.sleep(0.05)
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(1.0 / engine.display_fps)
        if cam not in engine.streams:
            return ("unknown camera", 404)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/config")
    def api_config():
        m = cfg["mqtt"]
        return jsonify({
            "session": engine.session,
            "scenario": engine.scenario,
            "scenario_label": engine.scenario_label,
            "sync_mode": engine.sync_mode,
            "cameras": [{"name": n, "metric": s.cam.metric}
                        for n, s in engine.streams.items()],
            "georef": engine.georef,
            "map_url": "/assets/map.png",
            "mqtt": {"ws_port": m["ws_port"], "topic": m["topic"],
                     "fused_topic": engine.fused_topic,
                     "fusion_enabled": engine.fusion_enabled,
                     "track_ttl_s": m["track_ttl_s"], "host": m["host"]},
        })

    @app.route("/assets/map.png")
    def asset_map():
        return send_from_directory(map_png.parent, map_png.name)

    @app.route("/dashboard/")
    def dashboard_index():
        return send_from_directory(dashboard_dir, "index.html")

    @app.route("/dashboard/<path:relpath>")
    def dashboard_files(relpath):
        return send_from_directory(dashboard_dir, relpath)

    @app.route("/<path:relpath>")
    def streamer_files(relpath):
        return send_from_directory(streamer_static, relpath)

    app.engine = engine
    return app


def main():
    import argparse
    ap = argparse.ArgumentParser(description="AMS digital-twin streamer")
    ap.add_argument("--scenario", help="scenario key from config.yaml (static|single|bulk)")
    ap.add_argument("--model", help="detector model: seg | detect (or a .pt path)")
    ap.add_argument("--port", type=int, help="override server port")
    args = ap.parse_args()

    cfg = appconfig.load()
    if args.scenario:
        cfg._d["scenario"] = args.scenario
    if args.model:
        cfg._d["detector"]["model"] = args.model
    if args.port:
        cfg._d["server"]["port"] = args.port
    app = create_app(cfg)
    s = cfg["server"]
    print(f"[streamer] http://{s['host']}:{s['port']}/  (broker {cfg['mqtt']['host']}:{cfg['mqtt']['port']})")
    app.run(host=s["host"], port=s["port"], threaded=True, debug=False)


if __name__ == "__main__":
    main()
