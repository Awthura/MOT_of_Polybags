"""
Combined 5-camera recorder for macOS: 2x Basler (pypylon) + 1x Lucid (Aravis,
not Arena SDK — no macOS support there) + 2x RealSense RGBD (pyrealsense2).

Each camera connects/configures itself independently, then reports success or
failure before waiting on a shared "go" event — a camera that fails to
connect is skipped, not allowed to block the others (see CameraWorker's
setup_done/go_event, replacing an earlier threading.Barrier design that
caused one failed camera to time out and take every other camera down with
it). Cameras that succeed are released to start recording at the same
wall-clock moment — this is software/start-time synchronization, not
hardware-triggered frame-by-frame sync. Each camera then free-runs at its own
target FPS; --fps sets the SAME target FPS for all of them.

Setup: see CAPTURE_INSTRUCTIONS.md. Requires pypylon, pygobject (+ Aravis via
Homebrew), pyrealsense2 (conda-forge), opencv-python, numpy.

Run:
    export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
    conda activate ams
    python record_all_5_cameras_macos.py --fps 15 --duration 90 --verbose

RealSense needs `sudo` on macOS (USB power-state permission, unrelated to
Basler/Lucid which are GigE) — if running the full 5-camera set, launch with:
    sudo -E $(which python) record_all_5_cameras_macos.py --fps 15
"""

import argparse
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
    print("INFO: pyrealsense2 not installed — RealSense disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Base worker
# ─────────────────────────────────────────────────────────────────────────────
class CameraWorker:
    def __init__(self, name, width, height, fps, duration, output_file):
        self.name         = name
        self.width        = width
        self.height       = height
        self.fps          = fps          # target fps (from --fps)
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
        # Setup and the actual synchronized start are decoupled: each worker
        # signals setup_done (success or failure) independently, so a camera
        # that fails to connect doesn't block the ones that did — replaces
        # a fixed-size threading.Barrier, which would time out (and take
        # every other camera down with it) if even one worker never reached
        # it.
        self.setup_done   = threading.Event()
        self.setup_ok     = False
        self.go_event     = None

    def set_go_event(self, go_event):
        self.go_event = go_event

    def mark_ready(self, ok):
        if not self.setup_done.is_set():
            self.setup_ok = ok
            self.setup_done.set()

    def wait_for_go(self, timeout=35):
        """Call after setup succeeds. Returns False if the go signal never
        arrives (e.g. another camera's setup took too long)."""
        self.mark_ready(True)
        if self.go_event is None:
            return True
        return self.go_event.wait(timeout=timeout)

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

    def actual_fps(self):
        return self.frame_count / max(self.elapsed(), 0.001)

    def release(self):
        if self.out:
            self.out.release()

    def _run(self):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# BaslerWorker — same logic as record_basler_lucid_10_macos.py (verified
# working against real hardware), fps now comes from the shared --fps value.
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
            except Exception:
                wb_locked = True

            try:
                camera.AcquisitionFrameRateEnable.Value = True
                camera.AcquisitionFrameRate.Value       = float(self.fps)
                print(f"[{self.name}] Target FPS: {self.fps}")
            except: pass

            try: camera.GevSCPSPacketSize.Value = 1400
            except: pass
            try: camera.GevSCPD.Value = 5000
            except: pass

            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat  = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self._init_writer()
            print(f"[{self.name}] Ready — waiting for sync...")

            if not self.wait_for_go():
                print(f"[{self.name}] Timed out waiting for other cameras — skipping")
                return

            self.start_time = time.time()
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print(f"[{self.name}] Recording → {self.output_file}")

            while self.running and camera.IsGrabbing():
                if self.elapsed() >= self.duration:
                    break

                grab = camera.RetrieveResult(3000, pylon.TimeoutHandling_ThrowException)

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
                    except Exception:
                        pass
                    wb_locked = True

        except Exception as e:
            self.error = str(e)
            self.mark_ready(False)
            print(f"[{self.name}] ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            if camera and camera.IsOpen():
                camera.StopGrabbing()
                camera.Close()
            self.release()
            print(f"[{self.name}] Done | Frames: {self.frame_count} | "
                  f"Incomplete: {self.incomplete} | Actual FPS: {self.actual_fps():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# LucidWorker — Aravis-based, verified working against real hardware
# (2 Basler + 1 Lucid, 772/1081/1339 frames recorded successfully).
# ─────────────────────────────────────────────────────────────────────────────
class LucidWorker(CameraWorker):
    def __init__(self, device_id, width, height, fps, duration, output_file):
        super().__init__("Lucid", width, height, fps, duration, output_file)
        self.device_id   = device_id
        self._camera     = None
        self._exp_min    = 10.0
        self._exp_max    = 1_000_000.0
        self.exposure_us = 10000.0

    def set_exposure(self, value_us):
        value_us = max(self._exp_min, min(self._exp_max, value_us))
        self.exposure_us = value_us
        if self._camera is not None:
            try:
                self._camera.set_exposure_time(value_us)
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

            try:
                camera.set_region(0, 0, self.width, self.height)
            except Exception as e:
                print(f"[{self.name}] set_region failed ({e}), using sensor default")
            _, _, w, h = camera.get_region()
            self.width, self.height = w, h

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

            try:
                camera.set_exposure_time_auto(Aravis.Auto.OFF)
                bounds = camera.get_exposure_time_bounds()
                self._exp_min, self._exp_max = bounds
                self.exposure_us = max(self._exp_min, min(self._exp_max, 10000.0))
                camera.set_exposure_time(self.exposure_us)
            except Exception as e:
                print(f"[{self.name}] WARNING: could not set manual exposure: {e}")
            try:
                camera.set_gain_auto(Aravis.Auto.OFF)
                camera.set_gain(0)
            except Exception:
                pass

            try:
                camera.set_frame_rate_enable(True)
                camera.set_frame_rate(float(self.fps))
                print(f"[{self.name}] Target FPS: {self.fps}")
            except Exception:
                pass

            try:
                camera.gv_set_packet_size(1400)
            except Exception:
                pass
            try:
                camera.gv_set_packet_delay(40000)
            except Exception:
                pass

            time.sleep(0.5)
            stream = camera.create_stream(None, None)
            payload = camera.get_payload()
            for _ in range(10):
                stream.push_buffer(Aravis.Buffer.new_allocate(payload))

            self._init_writer()
            print(f"[{self.name}] Ready — waiting for sync...")

            if not self.wait_for_go():
                print(f"[{self.name}] Timed out waiting for other cameras — skipping")
                return

            camera.start_acquisition()

            drain_until = time.time() + 0.5
            while time.time() < drain_until:
                buf = stream.try_pop_buffer()
                if buf is None:
                    break
                stream.push_buffer(buf)

            self.start_time = time.time()
            print(f"[{self.name}] Recording → {self.output_file}")

            while self.running:
                if self.elapsed() >= self.duration:
                    break

                buf = stream.timeout_pop_buffer(5_000_000)
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
                except Exception as e:
                    print(f"[{self.name}] frame conversion error: {e}")
                    stream.push_buffer(buf)
                    continue

                stream.push_buffer(buf)
                self._write_frame(frame)

        except Exception as e:
            self.error = str(e)
            self.mark_ready(False)
            print(f"[{self.name}] ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            try:
                camera.stop_acquisition()
            except Exception:
                pass
            self.release()
            print(f"[{self.name}] Done | Frames: {self.frame_count} | "
                  f"Incomplete: {self.incomplete} | Actual FPS: {self.actual_fps():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# RGBDWorker — RealSense via pyrealsense2, with the retry-safe pipeline.start()
# fix (fresh pipeline/config per attempt — reusing one after a failed start()
# leaves the USB handle half-claimed and every retry then fails with
# "UVC device is already opened!"). Reports setup success/failure via
# mark_ready()/wait_for_go() like the other worker types.
# ─────────────────────────────────────────────────────────────────────────────
class RGBDWorker(CameraWorker):
    def __init__(self, index, width, height, fps, duration, color_file, depth_file):
        super().__init__(f"RGBD_{index + 1}", width, height, fps, duration, color_file)
        self.index      = index
        self.depth_file = depth_file
        self.depth_out  = None

    def _init_depth_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.depth_out = cv2.VideoWriter(
            self.depth_file, fourcc, self.fps, (self.width, self.height)
        )
        if not self.depth_out.isOpened():
            print(f"[{self.name}] WARNING: could not open depth VideoWriter")
            self.depth_out = None

    def _write_depth(self, depth_mm):
        if self.depth_out is None or depth_mm is None:
            return
        norm = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        colour = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        self.depth_out.write(colour)

    def release(self):
        super().release()
        if self.depth_out:
            self.depth_out.release()

    def _run(self):
        pipeline = None
        try:
            ctx  = rs.context()
            devs = ctx.query_devices()
            if len(devs) <= self.index:
                raise RuntimeError(f"No RealSense device at index {self.index}")
            serial = devs[self.index].get_info(rs.camera_info.serial_number)

            # Retry with a fresh pipeline/config each attempt — see module
            # docstring / record_real_sense_dual.py fix for why reusing one
            # object across retries causes "UVC device is already opened!".
            profile  = None
            last_err = None
            for attempt in range(4):
                pipeline = rs.pipeline()
                config   = rs.config()
                config.enable_device(serial)
                # pyrealsense2's binding requires an int-like framerate —
                # a bare Python float (e.g. from --fps 10.0) is rejected.
                rs_fps = int(round(self.fps))
                config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, rs_fps)
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, rs_fps)
                try:
                    profile = pipeline.start(config)
                    break
                except RuntimeError as e:
                    last_err = e
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                    wait_s = 2.0 * (attempt + 1)
                    print(f"[{self.name}] pipeline.start() failed ({e}), "
                          f"retrying in {wait_s:.1f}s (attempt {attempt + 1}/4)")
                    time.sleep(wait_s)
            if profile is None:
                raise RuntimeError(f"pipeline.start() failed after 4 attempts: {last_err}")

            dev_name = profile.get_device().get_info(rs.camera_info.name)
            print(f"[{self.name}] RealSense connected: {dev_name} (S/N: {serial}) "
                  f"Target FPS: {self.fps}")

            align = rs.align(rs.stream.color)
            for _ in range(30):
                pipeline.wait_for_frames()

            self._init_writer()
            self._init_depth_writer()
            print(f"[{self.name}] Ready — waiting for sync...")

            if not self.wait_for_go():
                print(f"[{self.name}] Timed out waiting for other cameras — skipping")
                return

            self.start_time = time.time()
            print(f"[{self.name}] Recording → {self.output_file}")

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

        except Exception as e:
            self.error = str(e)
            self.mark_ready(False)
            print(f"[{self.name}] ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            self.release()
            print(f"[{self.name}] Done | Frames: {self.frame_count} | "
                  f"Incomplete: {self.incomplete} | Actual FPS: {self.actual_fps():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────
def display_loop(workers, duration, verbose):
    PREVIEW_W, PREVIEW_H, COLS = 480, 270, 3
    EXP_STEP = 0.20

    lucid_worker = next((w for w in workers if isinstance(w, LucidWorker)), None)
    last_good    = {w.name: None for w in workers}
    last_report  = time.time()

    print("[Display] Waiting for all cameras to start recording...")
    while True:
        if all(w.start_time is not None or w.error is not None for w in workers):
            break
        waiting = np.full((PREVIEW_H * 2, PREVIEW_W * COLS, 3), 30, dtype=np.uint8)
        cv2.putText(waiting, "Waiting for cameras...", (10, PREVIEW_H),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        cv2.imshow('5-Camera Recording  |  Q=stop  +/-=Lucid exposure', waiting)
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

        if verbose and time.time() - last_report >= 5.0:
            stats = "  ".join(f"{w.name}={w.actual_fps():.1f}fps" for w in workers)
            print(f"[Verbose] t={elapsed:.0f}s  {stats}")
            last_report = time.time()

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
                cv2.putText(tile, f"{w.name} init...", (10, PREVIEW_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            color = (0, 0, 255) if w.error else (0, 255, 0)
            cv2.putText(tile, f"{w.name} | {w.frame_count}f | {w.actual_fps():.1f}fps",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(tile, f"{int(elapsed)}s / {duration}s", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            if isinstance(w, LucidWorker):
                cv2.putText(tile, f"Exp: {w.exposure_us:.0f}us [+/-]", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 2)

            tiles.append(tile)

        blank = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
        while len(tiles) % COLS != 0:
            tiles.append(blank)
        rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
        grid = np.vstack(rows)

        cv2.imshow('5-Camera Recording  |  Q=stop  +/-=Lucid exposure', grid)

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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def record_all_cameras(duration_seconds, width, height, fps, verbose):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workers   = []

    # ── 1. Lucid via Aravis (init first, matching the original GigE-ordering
    # convention) ─────────────────────────────────────────────────────────────
    if LUCID_AVAILABLE:
        print("\n=== Enumerating GigE Vision devices (Aravis) for Lucid ===")
        Aravis.update_device_list()
        n = Aravis.get_n_devices()
        print(f"  Aravis sees {n} device(s):")
        lucid_id = None
        for i in range(n):
            dev_id = Aravis.get_device_id(i)
            vendor = Aravis.get_device_vendor(i)
            model  = Aravis.get_device_model(i)
            print(f"    [{i}] id={dev_id} vendor={vendor} model={model}")
            if lucid_id is None and 'lucid' in f"{vendor}{model}".lower():
                lucid_id = dev_id
        if lucid_id is not None:
            workers.append(LucidWorker(
                lucid_id, width, height, fps, duration_seconds,
                f"lucid_{width}x{height}_{timestamp}.avi"
            ))
        else:
            print("[Lucid] WARNING: no GigE Vision Lucid device found — skipping")

    # ── 2. Basler ─────────────────────────────────────────────────────────────
    if BASLER_AVAILABLE:
        print("\n=== Initializing Basler cameras ===")
        tlf   = pylon.TlFactory.GetInstance()
        infos = [d for d in tlf.EnumerateDevices() if 'basler' in d.GetVendorName().lower()]
        print(f"Found {len(infos)} Basler camera(s)")
        for i, info in enumerate(infos[:2]):
            camera = pylon.InstantCamera(tlf.CreateDevice(info))
            camera.Open()
            print(f"[Basler_{i+1}] Pre-initialized: {info.GetModelName()} (S/N: {info.GetSerialNumber()})")
            workers.append(BaslerWorker(
                info, i + 1, width, height, fps, duration_seconds,
                f"basler_{i+1}_{width}x{height}_{timestamp}.avi"
            ))
            workers[-1].precreated_camera = camera

    # ── 3. RealSense RGBD ─────────────────────────────────────────────────────
    if REALSENSE_AVAILABLE:
        print("\n=== Enumerating RealSense cameras ===")
        ctx  = rs.context()
        devs = ctx.query_devices()
        print(f"Found {len(devs)} RealSense camera(s)")
        for i, dev in enumerate(devs[:2]):
            sn = dev.get_info(rs.camera_info.serial_number)
            print(f"[RGBD_{i+1}] Prepared: S/N {sn}")
            workers.append(RGBDWorker(
                i, width, height, fps, duration_seconds,
                f"rgbd_{i+1}_color_{width}x{height}_{timestamp}.avi",
                f"rgbd_{i+1}_depth_{width}x{height}_{timestamp}.avi",
            ))

    if not workers:
        print("ERROR: No cameras found!")
        return

    # Each worker does its own device connection/configuration inside _run(),
    # then reports success/failure via mark_ready() before waiting on this
    # shared go_event. A camera that fails setup marks itself not-ready and
    # returns — it does NOT block the others (unlike a fixed-size
    # threading.Barrier, which requires every worker to arrive or times out
    # taking every other camera down with it).
    go_event = threading.Event()
    for w in workers:
        w.set_go_event(go_event)

    print(f"\nConnecting to {len(workers)} camera(s), target FPS: {fps}...")
    for w in workers:
        w.running = True
        w.thread  = threading.Thread(target=w._run, daemon=True)
        w.thread.start()

    print("Waiting for all cameras to finish setup (up to 35s each)...")
    for w in workers:
        if not w.setup_done.wait(timeout=35):
            print(f"[{w.name}] setup did not report back in time — treating as failed")
            w.mark_ready(False)

    ready_workers  = [w for w in workers if w.setup_ok]
    failed_workers = [w for w in workers if not w.setup_ok]

    if failed_workers:
        print(f"\nWARNING: {len(failed_workers)} camera(s) failed setup and will be "
              f"skipped: {[w.name for w in failed_workers]}")
    if not ready_workers:
        print("ERROR: no cameras successfully initialized — aborting")
        return

    print(f"{len(ready_workers)}/{len(workers)} camera(s) ready — "
          f"starting synchronized recording for {duration_seconds}s...\n")
    go_event.set()

    display_loop(ready_workers, duration_seconds, verbose)

    for w in workers:
        w.stop()
    for w in workers:
        w.join()

    print(f"\n{'='*70}")
    print(f"All recordings complete! (target FPS: {fps})")
    for w in workers:
        extra = f"  depth→ {w.depth_file}" if isinstance(w, RGBDWorker) else ""
        print(f"  {w.name:10} | {w.frame_count:5} frames | "
              f"{w.actual_fps():5.1f} fps actual | {w.incomplete} incomplete | "
              f"{w.output_file}{extra}")
    print(f"{'='*70}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Synchronized 5-camera recorder (Basler + Lucid + RealSense)")
    ap.add_argument('--fps', type=float, default=15.0, help="Target FPS for ALL cameras")
    ap.add_argument('--duration', type=int, default=90, help="Recording duration in seconds")
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--verbose', action='store_true', help="Print per-camera actual FPS every 5s")
    args = ap.parse_args()

    record_all_cameras(
        duration_seconds=args.duration,
        width=args.width,
        height=args.height,
        fps=args.fps,
        verbose=args.verbose,
    )
