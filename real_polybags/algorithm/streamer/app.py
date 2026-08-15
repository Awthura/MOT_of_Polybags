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
from publisher import Publisher                   # noqa: E402
from track import Tracker                         # noqa: E402

CAM_COLORS = {                    # BGR, kept in sync with the dashboard palette
    "basler_1": (60, 60, 235),
    "basler_2": (60, 210, 60),
    "lucid": (235, 180, 40),
    "rgbd_1_color": (200, 40, 200),
}


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

    def frame_at(self, p: float):
        """Decoded frame at normalized progress p in [0,1), and its index."""
        if self.count == 0:
            return None, -1
        idx = min(int(p * self.count), self.count - 1)
        f = cv2.imdecode(np.frombuffer(self.jpegs[idx], np.uint8), cv2.IMREAD_COLOR)
        return f, idx


class Engine:
    """Background loop: read → detect → project → publish, one tick per frame."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.session = cfg["session"]
        self.fps = float(cfg["playback"]["fps"])
        self.georef, self.map_png = _display_map(cfg)

        d = cfg["detector"]
        self.detector = Detector(_resolve(cfg, d["model_path"]),
                                 d["imgsz"], d["conf"], d["iou"], d["device"])
        # Which pixel of a detection is projected onto the belt: bbox centre
        # (default) or foot. Centre is steadier for flat bags than the bottom,
        # which carries view-dependent parallax.
        self.ref_point = d.get("reference_point", "center")

        # Conveyor bounds (world mm): projections outside are dropped as off-belt.
        b = cfg.get("belt_bounds") or {}
        self.belt_x = b.get("x_mm", [-1e9, 1e9])
        self.belt_y = b.get("y_mm", [-1e9, 1e9])

        results_dir = cfg.path("results_dir")
        overrides = cfg.get("calib_overrides") or {}
        m = cfg["mqtt"]
        self.streams: dict[str, Stream] = {}
        for name in cfg["cameras"]:
            calib_name = overrides.get(name, name)   # e.g. basler_1 video -> basler_2 calib
            try:
                cam = calib.load_camera(calib_name, results_dir)
            except (FileNotFoundError, ValueError) as e:
                print(f"[engine] skipping {name}: {e}")
                continue
            vid = cfg.path("video_dir") / f"{name}_1280x720_{self.session}.avi"
            if not vid.exists():
                print(f"[engine] no video for {name}: {vid.name}")
                continue
            if calib_name != name:
                print(f"[engine] {name} video uses {calib_name} calibration (override)")
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
        self.max_count = max((st.count for st in self.streams.values()), default=1)
        self.period = max(self.max_count / self.event_fps, 1e-3)
        self._t_start = time.time()

        self.pub = Publisher(m["host"], m["port"], m["topic"])
        self.pub.connect()

        self._latest: dict[str, bytes] = {}
        self._polybags: list[dict] = []      # latest inference result, republished steadily
        self._lock = threading.Lock()
        self._running = False
        self._tick = 0
        # Publish on a fixed cadence, decoupled from the (irregular, inference-
        # bound) frame loop, so the dashboard receives a steady stream and dots
        # never expire in the gap between two slow ticks. Holding the last known
        # positions between inferences keeps the map stable rather than flickering.
        self.publish_hz = float(m.get("publish_hz", 6))

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._publish_loop, daemon=True).start()

    def _publish_loop(self):
        interval = 1.0 / self.publish_hz
        while self._running:
            with self._lock:
                polybags = list(self._polybags)
            self.pub.publish({"t": int(time.time() * 1000), "frame": self._tick,
                              "session": self.session, "polybags": polybags})
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
            # Shared timeline position (seconds); each camera adds its fine offset.
            base = (t0 - self._t_start) * self.speed
            polybags = []
            for name, st in self.streams.items():
                p = ((base + self.sync_offset.get(name, 0.0)) % self.period) / self.period
                frame, _idx = st.frame_at(p)
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

                vis = draw_detections(frame.copy(), [b for b, r, w in keep],
                                      color=CAM_COLORS.get(name, (60, 220, 60)),
                                      ref=self.ref_point)
                for (tid, x, y, c), (b, (rx, ry), w) in zip(tagged, keep):
                    cv2.putText(vis, f"#{tid}", (int(rx) + 6, int(ry) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    polybags.append({
                        "cam": name, "id": int(tid),
                        "x_mm": round(float(x), 1), "y_mm": round(float(y), 1),
                        "conf": round(float(c), 2), "metric": st.cam.metric})
                ok_enc, buf = cv2.imencode(".jpg", vis,
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok_enc:
                    with self._lock:
                        self._latest[name] = buf.tobytes()

            # Hand this tick's positions to the steady publisher; do not publish
            # here (that would inherit this loop's irregular timing).
            with self._lock:
                self._polybags = polybags
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
                time.sleep(1.0 / 15)
        if cam not in engine.streams:
            return ("unknown camera", 404)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/config")
    def api_config():
        m = cfg["mqtt"]
        return jsonify({
            "session": cfg["session"],
            "cameras": [{"name": n, "metric": s.cam.metric}
                        for n, s in engine.streams.items()],
            "georef": engine.georef,
            "map_url": "/assets/map.png",
            "mqtt": {"ws_port": m["ws_port"], "topic": m["topic"],
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
    cfg = appconfig.load()
    app = create_app(cfg)
    s = cfg["server"]
    print(f"[streamer] http://{s['host']}:{s['port']}/  (broker {cfg['mqtt']['host']}:{cfg['mqtt']['port']})")
    app.run(host=s["host"], port=s["port"], threaded=True, debug=False)


if __name__ == "__main__":
    main()
