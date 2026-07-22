"""
macOS variant of record_basler_lucid_rgbd.py (supervisor's original, untouched,
lives alongside this file).

Only the Lucid camera path is different: Lucid's own Arena SDK (arena_api) has
no macOS build at all. Lucid cameras are standard GigE Vision devices though,
so this uses Aravis (vendor-neutral GigE Vision/GenICam library, available via
Homebrew) instead. Basler (pypylon) and RGBD (pyrealsense2, via conda-forge)
both have real macOS support and are used completely unchanged from the
original script.

Setup (one-time):
    brew install aravis pygobject3
    conda activate ams
    pip install pygobject
    conda install -c conda-forge pyrealsense2 -y
    pip install pypylon

Run (every session — DYLD_LIBRARY_PATH is required for Aravis to find
Homebrew's GLib/GObject libraries from inside the conda env):
    export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
    conda activate ams
    python record_basler_lucid_rgbd_macos.py

NOTE: the Aravis-based Lucid path below is written from the documented Aravis
0.8 Python API but has not been validated against a real Lucid camera yet
(no hardware access during development). If it errors on-site, the most
likely fixes are: adjusting the pixel-format string tried first (BGR8/RGB8/
BayerRG8), or the exposure/gain auto-mode enum (Aravis.Auto.OFF vs a string
value) — paste the traceback and it's a quick fix.
"""

import cv2
import numpy as np
from datetime import datetime
import time
import threading
import ctypes

# ── SDK imports ───────────────────────────────────────────────────────────────
try:
    from pypylon import pylon
    BASLER_AVAILABLE = True
except ImportError:
    print("WARNING: pypylon not installed — Basler disabled")
    BASLER_AVAILABLE = False

try:
    import gi
    gi.require_version('Aravis', '0.8')
    from gi.repository import Aravis
    Aravis.update_device_list()
    LUCID_AVAILABLE = True
    print("Aravis (GigE Vision) loaded OK — used for Lucid on macOS")
except Exception as e:
    print(f"WARNING: Lucid disabled - {type(e).__name__}: {e}")
    LUCID_AVAILABLE = False

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    print("Intel RealSense SDK loaded OK")
except ImportError:
    REALSENSE_AVAILABLE = False
    print("INFO: pyrealsense2 not installed — RealSense-specific RGBD disabled")

try:
    import pyk4a
    KINECT_AVAILABLE = True
    print("Azure Kinect SDK loaded OK")
except ImportError:
    KINECT_AVAILABLE = False
    print("INFO: pyk4a not installed — Kinect-specific RGBD disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Base worker (identical to the original script)
# ─────────────────────────────────────────────────────────────────────────────
class CameraWorker:
    def __init__(self, name, width, height, fps, duration, output_file):
        self.name         = name
        self.width        = width
        self.height       = height
        self.fps          = fps
        self.duration     = duration
        self.output_file  = output_file
        self.out          = None
        self.frame_count  = 0
        self.incomplete   = 0
        self.running      = False
        self.thread       = None
        self.latest_frame = None
        self.lock         = threading.Lock()
        self.error        = None
        self.start_time   = None
        self.barrier      = None

    def set_barrier(self, barrier):
        self.barrier = barrier

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def join(self):
        if self.thread:
            self.thread.join(timeout=10)

    def _init_writer(self):
        fourcc   = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter(
            self.output_file, fourcc, self.fps, (self.width, self.height)
        )
        if not self.out.isOpened():
            raise RuntimeError(f"Could not open VideoWriter: {self.output_file}")

    def _write_frame(self, frame):
        if frame is not None and frame.size > 0:
            self.out.write(frame)
            self.frame_count += 1
            with self.lock:
                self.latest_frame = frame.copy()

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def elapsed(self):
        return time.time() - self.start_time if self.start_time else 0

    def release(self):
        if self.out:
            self.out.release()

    def _run(self):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# RGBDWorker — unchanged from the original script (pyrealsense2 now installed
# via conda-forge, so the 'realsense' backend works as originally written)
# ─────────────────────────────────────────────────────────────────────────────
class RGBDWorker(CameraWorker):
    def __init__(self, name, index, backend, width, height, fps,
                 duration, color_file, depth_file=None):
        super().__init__(name, width, height, fps, duration, color_file)
        self.index        = index
        self.backend      = backend
        self.depth_file   = depth_file
        self.depth_out    = None
        self.latest_depth = None

    def _init_depth_writer(self):
        if self.depth_file is None:
            return
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.depth_out = cv2.VideoWriter(
            self.depth_file, fourcc, self.fps, (self.width, self.height)
        )
        if not self.depth_out.isOpened():
            print(f"[{self.name}] WARNING: Could not open depth VideoWriter: {self.depth_file}")
            self.depth_out = None

    def _write_depth(self, depth_mm):
        if self.depth_out is None or depth_mm is None:
            return
        norm = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        colour = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        self.depth_out.write(colour)
        with self.lock:
            self.latest_depth = depth_mm.copy()

    def release(self):
        super().release()
        if self.depth_out:
            self.depth_out.release()

    def _run(self):
        try:
            if self.backend == 'realsense':
                self._run_realsense()
            elif self.backend == 'kinect':
                self._run_kinect()
            else:
                self._run_opencv()
        except Exception as e:
            self.error = str(e)
            print(f"[{self.name}] ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.release()
            print(f"[{self.name}] Done | Frames: {self.frame_count} | "
                  f"Incomplete: {self.incomplete}")

    def _run_realsense(self):
        pipeline = rs.pipeline()
        cfg      = rs.config()

        ctx   = rs.context()
        devs  = ctx.query_devices()
        if len(devs) == 0:
            raise RuntimeError(f"[{self.name}] No RealSense devices found")
        serial = devs[self.index % len(devs)].get_info(rs.camera_info.serial_number)
        cfg.enable_device(serial)

        cfg.enable_stream(rs.stream.color, self.width, self.height,
                          rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height,
                          rs.format.z16,  self.fps)

        align    = rs.align(rs.stream.color)
        profile  = pipeline.start(cfg)
        dev_name = profile.get_device().get_info(rs.camera_info.name)
        print(f"[{self.name}] RealSense connected: {dev_name} (S/N: {serial})")

        for _ in range(30):
            pipeline.wait_for_frames()

        self._init_writer()
        self._init_depth_writer()
        print(f"[{self.name}] Ready — waiting for sync...")

        if self.barrier:
            self.barrier.wait(timeout=30)

        self.start_time = time.time()
        print(f"[{self.name}] Recording → {self.output_file}")

        try:
            while self.running:
                if self.elapsed() >= self.duration:
                    break
                frames      = pipeline.wait_for_frames(timeout_ms=5000)
                aligned     = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()

                if not color_frame or not depth_frame:
                    self.incomplete += 1
                    continue

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())

                self._write_frame(color)
                self._write_depth(depth)
        finally:
            pipeline.stop()

    def _run_kinect(self):
        from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS as K4AFPS

        fps_map = {15: K4AFPS.FPS_15, 30: K4AFPS.FPS_30, 5: K4AFPS.FPS_5}
        k4a = PyK4A(Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=fps_map.get(self.fps, K4AFPS.FPS_15),
            synchronized_images_only=True,
        ), device_id=self.index)
        k4a.start()
        print(f"[{self.name}] Azure Kinect device {self.index} started")

        for _ in range(20):
            k4a.get_capture()

        self._init_writer()
        self._init_depth_writer()
        print(f"[{self.name}] Ready — waiting for sync...")

        if self.barrier:
            self.barrier.wait(timeout=30)

        self.start_time = time.time()
        print(f"[{self.name}] Recording → {self.output_file}")

        try:
            while self.running:
                if self.elapsed() >= self.duration:
                    break
                cap = k4a.get_capture()
                if cap.color is None or cap.depth is None:
                    self.incomplete += 1
                    continue
                color = cap.color[:, :, :3]
                if color.shape[1] != self.width or color.shape[0] != self.height:
                    color = cv2.resize(color, (self.width, self.height))
                depth = cap.transformed_depth
                if depth is not None and \
                   (depth.shape[1] != self.width or depth.shape[0] != self.height):
                    depth = cv2.resize(depth, (self.width, self.height),
                                       interpolation=cv2.INTER_NEAREST)
                self._write_frame(color)
                self._write_depth(depth)
        finally:
            k4a.stop()

    def _run_opencv(self):
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"[{self.name}] Cannot open VideoCapture index {self.index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        for _ in range(10):
            cap.read()

        self._init_writer()
        print(f"[{self.name}] OpenCV VideoCapture({self.index}) ready — waiting for sync...")

        if self.barrier:
            self.barrier.wait(timeout=30)

        self.start_time = time.time()
        print(f"[{self.name}] Recording → {self.output_file}")

        try:
            while self.running:
                if self.elapsed() >= self.duration:
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.incomplete += 1
                    time.sleep(0.01)
                    continue
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                self._write_frame(frame)
        finally:
            cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# BaslerWorker — unchanged from the original script
# ─────────────────────────────────────────────────────────────────────────────
class BaslerWorker(CameraWorker):
    def __init__(self, device_info, index, width, height, fps, duration, output_file):
        super().__init__(f"Basler_{index}", width, height, fps, duration, output_file)
        self.device_info       = device_info
        self.precreated_camera = None

    def _run(self):
        camera = None
        try:
            tlf = pylon.TlFactory.GetInstance()

            if self.precreated_camera is not None:
                camera = self.precreated_camera
                print(f"[{self.name}] Using pre-initialized camera")
            else:
                camera = pylon.InstantCamera(tlf.CreateDevice(self.device_info))
                camera.Open()

            self.width  = min(self.width,  camera.Width.Max)
            self.height = min(self.height, camera.Height.Max)
            offset_x    = ((camera.Width.Max  - self.width)  // 2) & ~1
            offset_y    = ((camera.Height.Max - self.height) // 2) & ~1
            camera.Width.Value   = self.width
            camera.Height.Value  = self.height
            camera.OffsetX.Value = offset_x
            camera.OffsetY.Value = offset_y

            pixel_fmt = 'BayerRG8'
            try:
                camera.PixelFormat.Value = pixel_fmt
                print(f"[{self.name}] PixelFormat = {pixel_fmt}")
            except Exception as e:
                pixel_fmt = camera.PixelFormat.Value
                print(f"[{self.name}] PixelFormat fallback = {pixel_fmt} ({e})")

            try:
                camera.ExposureAuto.Value = 'Off'
                exp_us = max(camera.ExposureTime.Min,
                             min(camera.ExposureTime.Max, 13000.0))
                camera.ExposureTime.Value = exp_us
                print(f"[{self.name}] Exposure = {exp_us:.0f} us")
            except: pass

            try:
                camera.GainAuto.Value = 'Off'
                camera.Gain.Value     = camera.Gain.Min
            except: pass

            wb_locked = False
            try:
                camera.BalanceWhiteAuto.Value = 'Once'
                print(f"[{self.name}] BalanceWhiteAuto = Once "
                      f"(will lock after 1 s of recording)")
            except Exception as e:
                print(f"[{self.name}] BalanceWhiteAuto skipped: {e}")
                wb_locked = True

            try:
                camera.AcquisitionFrameRateEnable.Value = True
                camera.AcquisitionFrameRate.Value       = float(self.fps)
            except: pass

            try: camera.GevSCPSPacketSize.Value = 1400
            except: pass
            try: camera.GevSCPD.Value = 8000
            except: pass

            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat  = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self._init_writer()
            print(f"[{self.name}] Ready — waiting for sync...")

            if self.barrier:
                self.barrier.wait(timeout=30)

            self.start_time = time.time()
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print(f"[{self.name}] Recording → {self.output_file}")

            while self.running and camera.IsGrabbing():
                if self.elapsed() >= self.duration:
                    break

                grab = camera.RetrieveResult(
                    3000, pylon.TimeoutHandling_ThrowException)

                if not grab.GrabSucceeded():
                    self.incomplete += 1
                    grab.Release()
                    continue

                converted = converter.Convert(grab)
                frame = converted.Array
                grab.Release()
                self._write_frame(frame)

                if not wb_locked and self.elapsed() >= 1.0:
                    try:
                        camera.BalanceWhiteAuto.Value = 'Off'
                        wb = {}
                        for ch in ('Red', 'Green', 'Blue'):
                            camera.BalanceRatioSelector.Value = ch
                            wb[ch] = camera.BalanceRatio.Value
                        print(f"[{self.name}] WB locked — "
                              f"R={wb['Red']:.3f} G={wb['Green']:.3f} "
                              f"B={wb['Blue']:.3f}")
                    except Exception as e:
                        print(f"[{self.name}] WB lock skipped: {e}")
                    wb_locked = True

        except Exception as e:
            self.error = str(e)
            print(f"[{self.name}] ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            if camera and camera.IsOpen():
                camera.StopGrabbing()
                camera.Close()
            self.release()
            print(f"[{self.name}] Done | Frames: {self.frame_count} | "
                  f"Incomplete: {self.incomplete}")


# ─────────────────────────────────────────────────────────────────────────────
# LucidAravisWorker — NEW: Lucid via Aravis (GigE Vision), not Arena SDK
# ─────────────────────────────────────────────────────────────────────────────
class LucidAravisWorker(CameraWorker):
    def __init__(self, device_id, width, height, fps, duration, output_file):
        super().__init__("Lucid", width, height, fps, duration, output_file)
        self.device_id   = device_id   # Aravis device id string, or None for first found
        self.exposure_us = 10000.0

    def set_exposure(self, value_us):
        """Thread-safe-ish live exposure adjustment (best effort, mirrors the
        original script's +/- key handler)."""
        self.exposure_us = value_us
        if getattr(self, '_camera', None) is not None:
            try:
                self._camera.set_exposure_time(value_us)
                print(f"[{self.name}] Exposure → {value_us:.0f} µs")
            except Exception as e:
                print(f"[{self.name}] set_exposure error: {e}")

    def _run(self):
        camera = None
        stream = None
        try:
            camera = Aravis.Camera.new(self.device_id)
            self._camera = camera
            print(f"[{self.name}] Connected: {camera.get_vendor_name()} "
                  f"{camera.get_model_name()} (S/N: {camera.get_device_serial_number()})")

            # ── Resolution ───────────────────────────────────────────────
            try:
                camera.set_region(0, 0, self.width, self.height)
            except Exception as e:
                print(f"[{self.name}] set_region failed ({e}), using sensor default")
            _, _, w, h = camera.get_region()
            self.width, self.height = w, h
            print(f"[{self.name}] Resolution: {self.width}x{self.height}")

            # ── Pixel format ─────────────────────────────────────────────
            chosen_fmt = None
            for fmt in ('BGR8', 'RGB8'):
                try:
                    camera.set_pixel_format_from_string(fmt)
                    chosen_fmt = fmt
                    break
                except Exception:
                    continue
            if chosen_fmt is None:
                chosen_fmt = camera.get_pixel_format_as_string()
            print(f"[{self.name}] Pixel format: {chosen_fmt}")

            # ── Manual exposure / gain ─────────────────────────────────────
            try:
                camera.set_exposure_time_auto(Aravis.Auto.OFF)
                camera.set_exposure_time(self.exposure_us)
                print(f"[{self.name}] Manual exposure: {self.exposure_us:.0f} µs")
            except Exception as e:
                print(f"[{self.name}] WARNING: could not set manual exposure: {e}")
            try:
                camera.set_gain_auto(Aravis.Auto.OFF)
                camera.set_gain(0)
                print(f"[{self.name}] Gain set to minimum")
            except Exception:
                pass

            # ── FPS ──────────────────────────────────────────────────────
            try:
                camera.set_frame_rate_enable(True)
                camera.set_frame_rate(float(self.fps))
                print(f"[{self.name}] FPS: {self.fps}")
            except Exception:
                pass

            # ── GigE tuning (mirrors GevSCPSPacketSize/GevSCPD in the
            # original script's Basler/Lucid workers) ────────────────────
            try:
                camera.gv_set_packet_size(1400)
            except Exception as e:
                print(f"[{self.name}] packet size skipped: {e}")
            try:
                camera.gv_set_packet_delay(60000)
            except Exception as e:
                print(f"[{self.name}] packet delay skipped: {e}")

            # ── Stream setup ─────────────────────────────────────────────
            stream = camera.create_stream(None, None)
            payload = camera.get_payload()
            for _ in range(10):
                stream.push_buffer(Aravis.Buffer.new_allocate(payload))

            self._init_writer()
            print(f"[{self.name}] Ready — waiting for sync...")

            if self.barrier:
                self.barrier.wait(timeout=30)

            camera.start_acquisition()
            self.start_time = time.time()
            print(f"[{self.name}] Recording → {self.output_file}")

            while self.running:
                if self.elapsed() >= self.duration:
                    break

                buf = stream.timeout_pop_buffer(2_000_000)  # 2s timeout, in µs
                if buf is None:
                    self.incomplete += 1
                    continue
                if buf.get_status() != Aravis.BufferStatus.SUCCESS:
                    self.incomplete += 1
                    stream.push_buffer(buf)
                    continue

                try:
                    bw   = buf.get_image_width()
                    bh   = buf.get_image_height()
                    data = buf.get_data()
                    arr  = np.frombuffer(data, dtype=np.uint8)

                    if chosen_fmt == 'BGR8':
                        frame = arr.reshape(bh, bw, 3).copy()
                    elif chosen_fmt == 'RGB8':
                        frame = cv2.cvtColor(arr.reshape(bh, bw, 3), cv2.COLOR_RGB2BGR)
                    elif 'Bayer' in chosen_fmt:
                        frame = cv2.cvtColor(arr.reshape(bh, bw), cv2.COLOR_BAYER_BG2BGR)
                    else:
                        frame = cv2.cvtColor(arr.reshape(bh, bw), cv2.COLOR_GRAY2BGR)

                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                    self._write_frame(frame)

                    if self.frame_count in (1, 10, 50) or self.frame_count % 100 == 0:
                        print(f"[{self.name}] frame #{self.frame_count} written "
                              f"({self.elapsed():.1f}s elapsed)")
                except Exception as e:
                    print(f"[{self.name}] frame conversion error: {e}")
                finally:
                    stream.push_buffer(buf)

        except Exception as e:
            self.error = str(e)
            print(f"[{self.name}] ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            try:
                camera.stop_acquisition()
            except Exception:
                pass
            self.release()
            print(f"[{self.name}] Done | Frames: {self.frame_count} | "
                  f"Incomplete: {self.incomplete}")


# ─────────────────────────────────────────────────────────────────────────────
# Display — identical to the original script
# ─────────────────────────────────────────────────────────────────────────────
def display_loop(workers, duration):
    PREVIEW_W = 480
    PREVIEW_H = 270
    COLS      = 3
    EXP_STEP  = 0.20

    lucid_worker = next((w for w in workers if isinstance(w, LucidAravisWorker)), None)
    last_good    = {w.name: None for w in workers}

    print("[Display] Waiting for all cameras to start recording...")
    while True:
        if all(w.start_time is not None or w.error is not None for w in workers):
            break
        waiting = np.full((PREVIEW_H * 2, PREVIEW_W * COLS, 3), 30, dtype=np.uint8)
        cv2.putText(waiting, "Waiting for cameras...",
                    (10, PREVIEW_H), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (200, 200, 200), 2)
        cv2.imshow('Multi-Camera Recording  |  Q=stop  +/-=Lucid exposure', waiting)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            for w in workers:
                w.stop()
            cv2.destroyAllWindows()
            return
        time.sleep(0.05)

    started    = [w.start_time for w in workers if w.start_time is not None]
    loop_start = min(started) if started else time.time()
    print("[Display] All cameras live — recording started.")

    while True:
        elapsed = time.time() - loop_start
        if elapsed >= duration:
            break

        tiles = []
        for w in workers:
            f = w.get_latest_frame()
            if f is not None:
                last_good[w.name] = f
            else:
                f = last_good[w.name]

            if f is not None:
                tile = cv2.resize(f, (PREVIEW_W, PREVIEW_H))
            else:
                tile = np.full((PREVIEW_H, PREVIEW_W, 3), 50, dtype=np.uint8)
                cv2.putText(tile, f"{w.name} init...",
                            (10, PREVIEW_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            fps_now = w.frame_count / max(w.elapsed(), 0.001)
            label   = f"{w.name} | {w.frame_count}f | {fps_now:.1f}fps"
            color   = (0, 0, 255) if w.error else (0, 255, 0)
            cv2.putText(tile, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(tile, f"{int(elapsed)}s / {duration}s", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            if isinstance(w, LucidAravisWorker):
                cv2.putText(tile, f"Exp: {w.exposure_us:.0f}us  [+/-]",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            if isinstance(w, RGBDWorker) and w.latest_depth is not None:
                with w.lock:
                    d = w.latest_depth
                cy, cx = d.shape[0] // 2, d.shape[1] // 2
                depth_mm = int(d[cy, cx])
                cv2.putText(tile, f"Depth@ctr: {depth_mm}mm",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)

            tiles.append(tile)

        blank = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
        while len(tiles) % COLS != 0:
            tiles.append(blank)

        rows = []
        for i in range(0, len(tiles), COLS):
            rows.append(np.hstack(tiles[i:i + COLS]))
        grid = np.vstack(rows)

        cv2.imshow('Multi-Camera Recording  |  Q=stop  +/-=Lucid exposure', grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stop requested by user")
            for w in workers:
                w.stop()
            break

        if lucid_worker is not None:
            if key in (ord('+'), ord('=')):
                lucid_worker.set_exposure(lucid_worker.exposure_us * (1 + EXP_STEP))
            elif key == ord('-'):
                lucid_worker.set_exposure(lucid_worker.exposure_us * (1 - EXP_STEP))

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# RGBD device discovery — unchanged from the original script
# ─────────────────────────────────────────────────────────────────────────────
def _find_rgbd_backends():
    MAX_RGBD = 2
    found    = []

    if REALSENSE_AVAILABLE and len(found) < MAX_RGBD:
        ctx  = rs.context()
        devs = ctx.query_devices()
        for i, dev in enumerate(devs):
            if len(found) >= MAX_RGBD:
                break
            name = dev.get_info(rs.camera_info.name)
            sn   = dev.get_info(rs.camera_info.serial_number)
            found.append({'backend': 'realsense', 'index': i,
                          'label': f"RealSense {name} (S/N:{sn})"})
            print(f"  RGBD[{len(found)}] RealSense: {name} S/N {sn}")

    if KINECT_AVAILABLE and len(found) < MAX_RGBD:
        try:
            import pyk4a
            num_k4a = pyk4a.connected_device_count()
            for i in range(num_k4a):
                if len(found) >= MAX_RGBD:
                    break
                found.append({'backend': 'kinect', 'index': i,
                              'label': f"Azure Kinect #{i}"})
                print(f"  RGBD[{len(found)}] Azure Kinect device {i}")
        except Exception as e:
            print(f"  Kinect enumeration failed: {e}")

    if len(found) < MAX_RGBD:
        for idx in range(4):
            if len(found) >= MAX_RGBD:
                break
            claimed = [d['index'] for d in found if d['backend'] == 'opencv']
            if idx in claimed:
                continue
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                found.append({'backend': 'opencv', 'index': idx,
                              'label': f"OpenCV cam {idx}"})
                print(f"  RGBD[{len(found)}] OpenCV VideoCapture({idx})")
            cap.release()

    return found[:MAX_RGBD]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def record_all_cameras(duration_seconds=90, width=1280, height=720, fps=15):
    """
    Records simultaneously from:
      • Up to 2 RGBD cameras   (RealSense via pyrealsense2 / Kinect / OpenCV fallback)
      • Up to 2 Basler cameras (GigE, via pypylon)
      • 1  Lucid camera        (GigE, via Aravis — not Arena SDK, see module docstring)
    """
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    workers      = []

    # ── 1. RGBD cameras ──────────────────────────────────────────────────────
    print("\n=== Enumerating RGBD cameras ===")
    rgbd_devs = _find_rgbd_backends()
    print(f"Found {len(rgbd_devs)} RGBD device(s)")

    for i, dev in enumerate(rgbd_devs, start=1):
        name       = f"RGBD_{i}"
        color_file = f"rgbd_{i}_color_{width}x{height}_{timestamp}.avi"
        depth_file = (f"rgbd_{i}_depth_{width}x{height}_{timestamp}.avi"
                      if dev['backend'] != 'opencv' else None)
        w = RGBDWorker(
            name      = name,
            index     = dev['index'],
            backend   = dev['backend'],
            width     = width,
            height    = height,
            fps       = fps,
            duration  = duration_seconds,
            color_file= color_file,
            depth_file= depth_file,
        )
        workers.append(w)
        print(f"[{name}] Prepared: {dev['label']} → {color_file}"
              + (f" + {depth_file}" if depth_file else ""))

    # ── 2. Lucid via Aravis ──────────────────────────────────────────────────
    if LUCID_AVAILABLE:
        print("\n=== Enumerating GigE Vision devices (Aravis) for Lucid ===")
        Aravis.update_device_list()
        n = Aravis.get_n_devices()
        print(f"  Aravis sees {n} device(s):")
        lucid_id = None
        for i in range(n):
            dev_id = Aravis.get_device_id(i)
            vendor = Aravis.get_device_vendor(i) if hasattr(Aravis, 'get_device_vendor') else '?'
            model  = Aravis.get_device_model(i) if hasattr(Aravis, 'get_device_model') else '?'
            print(f"    [{i}] id={dev_id} vendor={vendor} model={model}")
            if lucid_id is None and 'lucid' in f"{vendor}{model}".lower():
                lucid_id = dev_id
        if lucid_id is None and n > 0:
            # Fall back to first device if vendor string doesn't match "lucid"
            # (some Lucid models report vendor as "Lucid Vision Labs" with
            # different casing/spacing — check the printed list above and set
            # lucid_id manually here if this picks the wrong camera).
            lucid_id = Aravis.get_device_id(0)
            print(f"  WARNING: no device matched 'lucid' by name — defaulting to device 0 ({lucid_id})")
        if lucid_id is not None:
            workers.append(LucidAravisWorker(
                lucid_id, width, height, fps, duration_seconds,
                f"lucid_{width}x{height}_{timestamp}.avi"
            ))
        else:
            print("[Lucid] WARNING: no GigE Vision device found — skipping")

    # ── 3. Basler cameras ─────────────────────────────────────────────────────
    if BASLER_AVAILABLE:
        print("\n=== Initializing Basler cameras ===")
        tlf   = pylon.TlFactory.GetInstance()
        infos = [d for d in tlf.EnumerateDevices()
                 if 'basler' in d.GetVendorName().lower()]
        print(f"Found {len(infos)} Basler camera(s)")

        for i, info in enumerate(infos[:2]):
            camera = pylon.InstantCamera(tlf.CreateDevice(info))
            camera.Open()
            print(f"[Basler_{i+1}] Pre-initialized: "
                  f"{info.GetModelName()} (S/N: {info.GetSerialNumber()})")
            workers.append(BaslerWorker(
                info, i + 1, width, height, fps, duration_seconds,
                f"basler_{i+1}_{width}x{height}_{timestamp}.avi"
            ))
            workers[-1].precreated_camera = camera

    if not workers:
        print("ERROR: No cameras found!")
        return

    barrier = threading.Barrier(len(workers))
    for w in workers:
        w.set_barrier(barrier)

    print(f"\nAll {len(workers)} cameras pre-initialized.")
    print(f"Starting simultaneous recording for {duration_seconds}s...\n")

    for w in workers:
        w.running = True
        w.thread  = threading.Thread(target=w._run, daemon=True)
        w.thread.start()

    display_loop(workers, duration_seconds)

    for w in workers:
        w.stop()
    for w in workers:
        w.join()

    print(f"\n{'='*60}")
    print("All recordings complete!")
    for w in workers:
        fps_actual = w.frame_count / max(w.elapsed(), 0.001)
        extra = ""
        if isinstance(w, RGBDWorker) and w.depth_file:
            extra = f"  depth→ {w.depth_file}"
        print(f"  {w.name:12} | {w.frame_count:5} frames | "
              f"{fps_actual:.1f} fps | {w.incomplete} incomplete | "
              f"{w.output_file}{extra}")
    print(f"{'='*60}")


if __name__ == '__main__':
    record_all_cameras(
        duration_seconds=50,
        width=1280,
        height=720,
        fps=5
    )
