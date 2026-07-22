"""
macOS variant of record_basler_lucid_10.py (original untouched, alongside this
file). Only the Lucid path differs: Lucid's own Arena SDK (arena_api) has no
macOS build at all (Windows/Linux/Linux-ARM only). Lucid cameras are standard
GigE Vision devices though, so this uses Aravis (vendor-neutral GigE Vision
library, via Homebrew) instead. BaslerWorker and all GigE bandwidth-tuning
values/comments are unchanged from the original.

Setup (one-time, see CAPTURE_INSTRUCTIONS.md for full detail):
    brew install aravis pygobject3
    conda activate ams
    pip install pygobject pypylon

Run (every session):
    export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
    conda activate ams
    python record_basler_lucid_10_macos.py

NOTE: the Aravis-based Lucid capture loop is written from the documented
Aravis 0.8 Python API (import + device enumeration verified working) but has
not been validated against a real Lucid camera end-to-end. If it errors once
connected to hardware, paste the traceback — likely a quick fix (pixel-format
string or auto-exposure enum), not a redesign.
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
            try: camera.GevSCPD.Value = 5000
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
# LucidWorker — NEW: same class name and constructor shape the rest of this
# script expects, but backed by Aravis (GigE Vision) instead of Arena SDK.
# GigE bandwidth-tuning values (packet size 1400, GevSCPD 40000ns) are carried
# over unchanged from the original script's comments/reasoning.
# ─────────────────────────────────────────────────────────────────────────────
class LucidWorker(CameraWorker):
    def __init__(self, device_id, width, height, fps, duration, output_file):
        super().__init__("Lucid", width, height, fps, duration, output_file)
        self.device_id   = device_id   # Aravis device id string
        self._camera     = None        # set in _run; used by set_exposure()
        self._exp_min    = 10.0
        self._exp_max    = 1_000_000.0
        self.exposure_us = 10000.0

    def set_exposure(self, value_us):
        """Thread-safe-ish live exposure adjustment. Call from display loop."""
        value_us = max(self._exp_min, min(self._exp_max, value_us))
        self.exposure_us = value_us
        if self._camera is not None:
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

            # ── Manual exposure — start at a safe dim value ───────────────
            try:
                camera.set_exposure_time_auto(Aravis.Auto.OFF)
                self._exp_min = getattr(camera, 'get_exposure_time_bounds', lambda: (10.0, 1_000_000.0))()[0] \
                    if hasattr(camera, 'get_exposure_time_bounds') else self._exp_min
                self._exp_max = getattr(camera, 'get_exposure_time_bounds', lambda: (10.0, 1_000_000.0))()[1] \
                    if hasattr(camera, 'get_exposure_time_bounds') else self._exp_max
                self.exposure_us = max(self._exp_min, min(self._exp_max, 10000.0))
                camera.set_exposure_time(self.exposure_us)
                print(f"[{self.name}] Manual exposure: {self.exposure_us:.0f} µs "
                      f"(range {self._exp_min:.0f}–{self._exp_max:.0f} µs)")
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

            # ── Stream tuning — same bandwidth budget reasoning as the
            # original: 3 cameras sharing one 1 Gbps NIC, GevSCPD spaces
            # Lucid's packets so its bursts don't collide with the two
            # Baslers. Packet size 1400 bytes, 40 µs inter-packet delay. ───
            try:
                camera.gv_set_packet_size(1400)
                print(f"[{self.name}] GevSCPSPacketSize = 1400")
            except Exception as e:
                print(f"[{self.name}] packet size skipped: {e}")
            try:
                camera.gv_set_packet_delay(40000)   # 40 µs, matches original
                print(f"[{self.name}] GevSCPD = 40000 ns (40 µs)")
            except Exception as e:
                print(f"[{self.name}] packet delay skipped: {e}")

            # ── Stream setup ─────────────────────────────────────────────
            # Small stagger: let the two Baslers start streaming first so
            # their first burst of packets clears before Lucid joins.
            time.sleep(0.5)
            stream = camera.create_stream(None, None)
            payload = camera.get_payload()
            for _ in range(10):
                stream.push_buffer(Aravis.Buffer.new_allocate(payload))

            self._init_writer()
            print(f"[{self.name}] Stream ready, waiting for sync...")

            if self.barrier:
                self.barrier.wait(timeout=30)

            camera.start_acquisition()

            # ── Drain stale pre-barrier frames ────────────────────────────
            drain_until = time.time() + 0.5
            drained = 0
            while time.time() < drain_until:
                buf = stream.try_pop_buffer()
                if buf is None:
                    break
                stream.push_buffer(buf)
                drained += 1
            if drained:
                print(f"[{self.name}] Drained {drained} stale pre-barrier buffer(s)")

            self.start_time = time.time()
            print(f"[{self.name}] *** CAPTURE LOOP STARTED *** → {self.output_file}")

            # ── Capture loop ─────────────────────────────────────────────
            while self.running:
                if self.elapsed() >= self.duration:
                    break

                buf = stream.timeout_pop_buffer(5_000_000)  # 5s timeout, in µs
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
                if self.frame_count in (1, 10, 50) or self.frame_count % 100 == 0:
                    print(f"[{self.name}] frame #{self.frame_count} written "
                          f"({self.elapsed():.1f}s elapsed)")

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
# Display — unchanged from the original script
# ─────────────────────────────────────────────────────────────────────────────
def display_loop(workers, duration):
    PREVIEW_W   = 640
    PREVIEW_H   = 360
    EXP_STEP    = 0.20

    lucid_worker = next((w for w in workers if isinstance(w, LucidWorker)), None)
    last_good = {w.name: None for w in workers}

    print("[Display] Waiting for all cameras to start recording...")
    while True:
        if all(w.start_time is not None or w.error is not None for w in workers):
            break
        waiting = np.full((PREVIEW_H, PREVIEW_W, 3), 30, dtype=np.uint8)
        cv2.putText(waiting, "Waiting for cameras...",
                    (10, PREVIEW_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 2)
        cv2.imshow('Multi-Camera Recording  |  Q=stop  +/-=Lucid exposure', waiting)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            for w in workers:
                w.stop()
            cv2.destroyAllWindows()
            return
        time.sleep(0.05)

    started = [w.start_time for w in workers if w.start_time is not None]
    loop_start = min(started) if started else time.time()
    print("[Display] All cameras live — recording started.")

    while True:
        elapsed = time.time() - loop_start
        if elapsed >= duration:
            break

        frames = []
        for w in workers:
            f = w.get_latest_frame()

            if f is not None:
                last_good[w.name] = f
            else:
                f = last_good[w.name]

            if f is not None:
                preview = cv2.resize(f, (PREVIEW_W, PREVIEW_H))
            else:
                preview = np.full((PREVIEW_H, PREVIEW_W, 3),
                                  50, dtype=np.uint8)
                cv2.putText(preview, f"{w.name} initializing...",
                            (10, PREVIEW_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (200, 200, 200), 2)

            fps_now = w.frame_count / max(w.elapsed(), 0.001)
            label   = f"{w.name} | {w.frame_count}f | {fps_now:.1f}fps"
            color   = (0, 0, 255) if w.error else (0, 255, 0)
            cv2.putText(preview, label,
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
            cv2.putText(preview, f"{int(elapsed)}s / {duration}s",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)

            if isinstance(w, LucidWorker):
                exp_label = f"Exp: {w.exposure_us:.0f} us  [+/-] to adjust"
                cv2.putText(preview, exp_label,
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 200, 255), 2)

            frames.append(preview)

        if len(frames) == 3:
            top    = np.hstack([frames[0], frames[1]])
            pad    = np.zeros((PREVIEW_H, PREVIEW_W // 2, 3), dtype=np.uint8)
            bottom = np.hstack([pad, frames[2], pad])
            grid   = np.vstack([top, bottom])
        elif len(frames) == 2:
            grid = np.hstack(frames)
        elif len(frames) == 1:
            grid = frames[0]
        else:
            grid = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)

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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def record_all_cameras(duration_seconds=90, width=1280, height=720, fps=15):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workers   = []

    # ── Lucid via Aravis (still goes first, matching the original script's
    # ordering — kept as a conservative default even though the specific
    # Arena-SDK-vs-pypylon GigE conflict this guarded against doesn't
    # necessarily apply to Aravis's own stack) ───────────────────────────────
    if LUCID_AVAILABLE:
        print("Enumerating GigE Vision devices (Aravis) for Lucid...")
        Aravis.update_device_list()
        n = Aravis.get_n_devices()
        print(f"  Aravis sees {n} device(s) total:")
        lucid_id = None
        for i in range(n):
            dev_id = Aravis.get_device_id(i)
            vendor = Aravis.get_device_vendor(i) if hasattr(Aravis, 'get_device_vendor') else '?'
            model  = Aravis.get_device_model(i) if hasattr(Aravis, 'get_device_model') else '?'
            print(f"    [{i}] id={dev_id} vendor={vendor} model={model}")
            if lucid_id is None and 'lucid' in f"{vendor}{model}".lower():
                lucid_id = dev_id
        if lucid_id is None and n > 0:
            lucid_id = Aravis.get_device_id(0)
            print(f"  WARNING: no device matched 'lucid' by name — defaulting to device 0 ({lucid_id})")

        if lucid_id is not None:
            workers.append(LucidWorker(
                lucid_id, width, height, fps, duration_seconds,
                f"lucid_{width}x{height}_{timestamp}.avi"
            ))
        else:
            print("[Lucid] WARNING: no GigE Vision device found — skipping")

    # ── Pre-initialize Basler cameras (AFTER Lucid is open) ──────────────────
    if BASLER_AVAILABLE:
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
        print(f"  {w.name:12} | {w.frame_count:5} frames | "
              f"{fps_actual:.1f} fps | {w.incomplete} incomplete | "
              f"{w.output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    record_all_cameras(
        duration_seconds=90,
        width=1280,
        height=720,
        fps=15
    )
