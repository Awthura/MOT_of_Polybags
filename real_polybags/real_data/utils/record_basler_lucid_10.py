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
    from arena_api.system import system as ArenaSystem
    LUCID_AVAILABLE = True
    print("Lucid Arena SDK loaded OK")
except Exception as e:
    print(f"WARNING: Lucid disabled - {type(e).__name__}: {e}")
    LUCID_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Base worker
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
# BaslerWorker
# ─────────────────────────────────────────────────────────────────────────────
class BaslerWorker(CameraWorker):
    def __init__(self, device_info, index, width, height, fps, duration, output_file):
        super().__init__(f"Basler_{index}", width, height, fps, duration, output_file)
        self.device_info       = device_info
        self.precreated_camera = None   # set by record_all_cameras if pre-opened

    def _run(self):
        camera = None
        try:
            tlf = pylon.TlFactory.GetInstance()

            # Use pre-created camera if available, else open fresh
            if self.precreated_camera is not None:
                camera = self.precreated_camera
                print(f"[{self.name}] Using pre-initialized camera")
            else:
                camera = pylon.InstantCamera(tlf.CreateDevice(self.device_info))
                camera.Open()

            # ── Resolution ───────────────────────────────────────────────
            self.width  = min(self.width,  camera.Width.Max)
            self.height = min(self.height, camera.Height.Max)
            offset_x    = ((camera.Width.Max  - self.width)  // 2) & ~1
            offset_y    = ((camera.Height.Max - self.height) // 2) & ~1
            camera.Width.Value   = self.width
            camera.Height.Value  = self.height
            camera.OffsetX.Value = offset_x
            camera.OffsetY.Value = offset_y

            # ── Pixel format ─────────────────────────────────────────────
            # Always use BayerRG8 so we control demosaicing + white balance
            # through the converter. BGR8 skips the camera's on-chip colour
            # processing and can produce the green tint seen on some models.
            pixel_fmt = 'BayerRG8'
            try:
                camera.PixelFormat.Value = pixel_fmt
                print(f"[{self.name}] PixelFormat = {pixel_fmt}")
            except Exception as e:
                pixel_fmt = camera.PixelFormat.Value
                print(f"[{self.name}] PixelFormat fallback = {pixel_fmt} ({e})")

            # ── Exposure ─────────────────────────────────────────────────
            try:
                camera.ExposureAuto.Value = 'Off'
                exp_us = max(camera.ExposureTime.Min,
                             min(camera.ExposureTime.Max, 13000.0))
                camera.ExposureTime.Value = exp_us
                print(f"[{self.name}] Exposure = {exp_us:.0f} us")
            except: pass

            # ── Gain ─────────────────────────────────────────────────────
            try:
                camera.GainAuto.Value = 'Off'
                camera.Gain.Value     = camera.Gain.Min
            except: pass

            # ── White balance — Once, locks during first second of capture ─
            # Do NOT call StartGrabbing here — camera is pre-opened and the
            # barrier hasn't fired yet. Set Once now; the camera converges
            # during the first ~15 live frames, then we freeze the ratios.
            wb_locked = False
            try:
                camera.BalanceWhiteAuto.Value = 'Once'
                print(f"[{self.name}] BalanceWhiteAuto = Once "
                      f"(will lock after 1 s of recording)")
            except Exception as e:
                print(f"[{self.name}] BalanceWhiteAuto skipped: {e}")
                wb_locked = True  # nothing to lock later

            # ── FPS ──────────────────────────────────────────────────────
            try:
                camera.AcquisitionFrameRateEnable.Value = True
                camera.AcquisitionFrameRate.Value       = float(self.fps)
            except: pass

            # ── GigE tuning ──────────────────────────────────────────────
            try: camera.GevSCPSPacketSize.Value = 1400
            except: pass
            try: camera.GevSCPD.Value = 5000
            except: pass

            # ── Converter: Bayer → BGR ────────────────────────────────────
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat  = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self._init_writer()
            print(f"[{self.name}] Ready — waiting for sync...")

            # ── Barrier ──────────────────────────────────────────────────
            if self.barrier:
                self.barrier.wait(timeout=30)

            # ── Start AFTER barrier ───────────────────────────────────────
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

                # Always convert so Bayer demosaicing + WB ratios are applied
                converted = converter.Convert(grab)
                frame = converted.Array
                grab.Release()
                self._write_frame(frame)

                # Lock WB after 1 second — ratios have converged by then
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
# LucidWorker — ctypes zero-copy capture, BGR8, manual exposure
# ─────────────────────────────────────────────────────────────────────────────

class LucidWorker(CameraWorker):
    def __init__(self, arena_system, device, width, height, fps, duration, output_file):
        super().__init__("Lucid", width, height, fps, duration, output_file)
        self.arena_system  = arena_system  # already created
        self.device        = device        # already opened
        self._nodemap      = None          # set in _run; used by set_exposure()
        self._exp_min      = 10.0          # µs — updated after camera connects
        self._exp_max      = 1_000_000.0   # µs — updated after camera connects
        self.exposure_us   = 10000.0       # current manual exposure (µs)

    def set_exposure(self, value_us):
        """Thread-safe live exposure adjustment. Call from display loop."""
        value_us = max(self._exp_min, min(self._exp_max, value_us))
        self.exposure_us = value_us
        if self._nodemap is not None:
            try:
                self._nodemap['ExposureTime'].value = value_us
                print(f"[{self.name}] Exposure → {value_us:.0f} µs")
            except Exception as e:
                print(f"[{self.name}] set_exposure error: {e}")

    def _run(self):
        device = self.device
        try:
            nodemap = device.nodemap
            model   = nodemap['DeviceModelName'].value
            serial  = nodemap['DeviceSerialNumber'].value
            print(f"[{self.name}] Connected: {model} (S/N: {serial})")

            # ── Resolution ───────────────────────────────────────────────
            self.width  = min(self.width,  nodemap['Width'].max)
            self.height = min(self.height, nodemap['Height'].max)
            nodemap['Width'].value  = self.width
            nodemap['Height'].value = self.height
            print(f"[{self.name}] Resolution: {self.width}x{self.height}")

            # ── Reset ────────────────────────────────────────────────────
            try:    nodemap['TriggerMode'].value     = 'Off'
            except: pass
            try:    nodemap['AcquisitionMode'].value = 'Continuous'
            except: pass

            # ── Pixel format ─────────────────────────────────────────────
            # BGR8 is not universally supported; fall back to RGB8 or the
            # camera default if it raises. chosen_fmt drives conversion below.
            chosen_fmt = None
            for fmt in ('BGR8', 'RGB8'):
                try:
                    nodemap['PixelFormat'].value = fmt
                    chosen_fmt = fmt
                    break
                except Exception:
                    continue
            if chosen_fmt is None:
                chosen_fmt = nodemap['PixelFormat'].value   # camera default
            print(f"[{self.name}] Pixel format: {chosen_fmt}")

            # ── Manual exposure — start at a safe dim value ───────────────
            # Auto exposure is disabled so the display-loop +/- keys work.
            # We read the camera's min/max so set_exposure() can clamp correctly.
            try:
                nodemap['ExposureAuto'].value = 'Off'
                self._exp_min    = nodemap['ExposureTime'].min
                self._exp_max    = nodemap['ExposureTime'].max
                # Start at 10 ms (10 000 µs); user can adjust up/down live
                self.exposure_us = max(self._exp_min, min(self._exp_max, 10000.0))
                nodemap['ExposureTime'].value = self.exposure_us
                print(f"[{self.name}] Manual exposure: {self.exposure_us:.0f} µs "
                      f"(range {self._exp_min:.0f}–{self._exp_max:.0f} µs)")
            except Exception as e:
                print(f"[{self.name}] WARNING: could not set manual exposure: {e}")
            try:
                nodemap['GainAuto'].value = 'Off'
                nodemap['Gain'].value     = nodemap['Gain'].min
                print(f"[{self.name}] Gain set to minimum")
            except: pass
            # Store nodemap reference so set_exposure() can reach it from any thread
            self._nodemap = nodemap

            # ── FPS ──────────────────────────────────────────────────────
            try:
                if 'AcquisitionFrameRateEnable' in nodemap.feature_names:
                    nodemap['AcquisitionFrameRateEnable'].value = True
                if 'AcquisitionFrameRate' in nodemap.feature_names:
                    nodemap['AcquisitionFrameRate'].value = float(self.fps)
                print(f"[{self.name}] FPS: {self.fps}")
            except: pass

            # ── Stream tuning ─────────────────────────────────────────────
            # 3 cameras share ONE 1 Gbps NIC.
            # 1280x720 BGR8 @ 15fps = ~264 Mbps per camera → 792 Mbps total.
            # Without spacing the Lucid bursts packets that collide with the
            # two Baslers and every buffer arrives incomplete → 0 frames.
            #
            # GevSCPD (inter-packet delay) = time in ns between packets.
            # With 3 cameras: space Lucid packets so it uses ~1/3 of bandwidth.
            # Packet size 1400 bytes → ~11200 bits.
            # Target: leave ~330 Mbps for Lucid = 30 µs per packet gap minimum.
            # We use 40000 ns (40 µs) to be conservative.
            try:
                nodemap['GevSCPSPacketSize'].value = 1400
                print(f"[{self.name}] GevSCPSPacketSize = 1400")
            except Exception as e:
                print(f"[{self.name}] GevSCPSPacketSize skipped: {e}")
            try:
                nodemap['GevSCPD'].value = 40000   # 40 µs inter-packet delay
                print(f"[{self.name}] GevSCPD = 40000 ns (40 µs)")
            except Exception as e:
                print(f"[{self.name}] GevSCPD skipped: {e}")
            try:
                tl = device.tl_stream_nodemap
                tl['StreamBufferHandlingMode'].value = 'OldestFirst'
                tl['StreamPacketResendEnable'].value  = True
                print(f"[{self.name}] StreamBufferHandlingMode = OldestFirst, resend ON")
            except Exception as e:
                print(f"[{self.name}] stream nodemap skipped: {e}")

            # ── Start stream, then wait for barrier ───────────────────────
            # Small stagger: let the two Baslers start streaming first so
            # their first burst of packets clears before Lucid joins.
            time.sleep(0.5)
            device.start_stream(30)
            print(f"[{self.name}] Stream started, waiting for sync...")

            self._init_writer()

            # ── Barrier ───────────────────────────────────────────────────
            if self.barrier:
                self.barrier.wait(timeout=30)

            # ── Drain stale pre-barrier frames ────────────────────────────
            # Pull frames that arrived during init/barrier and discard them
            # so the capture loop starts with a live frame, not stale ones.
            # Use a short but nonzero timeout — timeout=0 is not guaranteed
            # to raise on an empty queue in all Arena SDK versions and can
            # block the thread indefinitely.
            drain_until = time.time() + 0.5   # drain window: 500 ms max
            drained = 0
            while time.time() < drain_until:
                try:
                    stale = device.get_buffer(timeout=50)   # 50 ms — short but safe
                    device.requeue_buffer(stale)
                    drained += 1
                except Exception:
                    break   # queue empty — done draining
            if drained:
                print(f"[{self.name}] Drained {drained} stale pre-barrier buffer(s)")

            self.start_time = time.time()
            print(f"[{self.name}] *** CAPTURE LOOP STARTED *** → {self.output_file}")

            # ── Capture loop ─────────────────────────────────────────────
            while self.running:
                if self.elapsed() >= self.duration:
                    break

                try:
                    buffer = device.get_buffer(timeout=5000)  # 5 s — accommodates GevSCPD spacing
                except Exception as e:
                    print(f"[{self.name}] buffer error: {e}")
                    continue

                if buffer.is_incomplete:
                    self.incomplete += 1
                    device.requeue_buffer(buffer)
                    continue

                # Zero-copy read via ctypes pointer.
                # .copy() owns the data so we can safely requeue before writing.
                try:
                    pdata = ctypes.cast(buffer.pdata,
                                        ctypes.POINTER(ctypes.c_ubyte))
                    arr   = np.ctypeslib.as_array(
                                pdata,
                                shape=(buffer.height * buffer.width * 3,))
                    frame = arr.reshape(buffer.height, buffer.width, 3).copy()
                    if chosen_fmt == 'RGB8':
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"[{self.name}] frame conversion error: {e}")
                    device.requeue_buffer(buffer)
                    continue

                # Requeue AFTER copy — buffer is returned to SDK immediately
                device.requeue_buffer(buffer)
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
                device.stop_stream()
                self.arena_system.destroy_device(device)
            except: pass
            self.release()
            print(f"[{self.name}] Done | Frames: {self.frame_count} | "
                  f"Incomplete: {self.incomplete}")


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────
def display_loop(workers, duration):
    PREVIEW_W   = 640
    PREVIEW_H   = 360
    EXP_STEP    = 0.20   # 20 % change per keypress

    # Find the Lucid worker once so keypress handling is O(1)
    lucid_worker = next((w for w in workers if isinstance(w, LucidWorker)), None)

    # Keep last good frame per worker
    last_good = {w.name: None for w in workers}

    # ── Wait until ALL workers have crossed the barrier and set start_time ──
    # Without this the display clock races ahead of the capture threads and
    # the loop can expire before a single frame is recorded.
    print("[Display] Waiting for all cameras to start recording...")
    while True:
        if all(w.start_time is not None or w.error is not None for w in workers):
            break
        # Show a waiting screen so the window appears immediately
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

    # Derive a shared start time from the earliest worker that actually started
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

            # Keep last good frame if current is None
            if f is not None:
                last_good[w.name] = f
            else:
                f = last_good[w.name]

            if f is not None:
                preview = cv2.resize(f, (PREVIEW_W, PREVIEW_H))
            else:
                # Gray placeholder with camera name
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

            # Show live exposure value on the Lucid tile
            if isinstance(w, LucidWorker):
                exp_label = f"Exp: {w.exposure_us:.0f} us  [+/-] to adjust"
                cv2.putText(preview, exp_label,
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 200, 255), 2)

            frames.append(preview)

        # Layout
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

        # Live Lucid exposure adjustment: + increases, - decreases by 20 %
        if lucid_worker is not None:
            if key in (ord('+'), ord('=')):   # '=' so no Shift needed
                lucid_worker.set_exposure(lucid_worker.exposure_us * (1 + EXP_STEP))
            elif key == ord('-'):
                lucid_worker.set_exposure(lucid_worker.exposure_us * (1 - EXP_STEP))

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def record_all_cameras(duration_seconds=90, width=1280, height=720, fps=15):

    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    workers      = []
    arena_system = None  # keep alive for full session

    # ── IMPORTANT: Lucid MUST be initialized BEFORE pypylon opens Basler ──────
    # Arena SDK and pypylon both touch the GigE network stack. If pypylon opens
    # the Basler cameras first, the Arena SDK's subsequent create_device() call
    # for the Lucid camera conflicts with the shared GigE transport layer and
    # hangs or silently fails. Lucid goes first.
    if LUCID_AVAILABLE:
        print("Initializing Lucid (Arena SDK) — must happen before Basler opens...")
        arena_system = ArenaSystem

        # Give the SDK a moment to enumerate all GigE devices cleanly
        time.sleep(1.0)
        all_infos = arena_system.device_infos
        print(f"  Arena SDK sees {len(all_infos)} device(s) total:")
        for d in all_infos:
            print(f"    vendor={d.get('vendor','?')}  model={d.get('model','?')}  "
                  f"serial={d.get('serial','?')}  ip={d.get('ip','?')}")

        def is_lucid(d):
            return any(k in str(dict(d)).lower()
                       for k in ['lucid', 'atlas', 'triton', 'phoenix', 'helios'])

        lucid_infos = [d for d in all_infos if is_lucid(d)]
        print(f"  Lucid device(s) matched: {len(lucid_infos)}")

        if lucid_infos:
            lucid_device = arena_system.create_device([lucid_infos[0]])
            if isinstance(lucid_device, list):
                lucid_device = lucid_device[0]
            model  = lucid_device.nodemap['DeviceModelName'].value
            serial = lucid_device.nodemap['DeviceSerialNumber'].value
            print(f"[Lucid] Pre-initialized: {model} (S/N: {serial})")

            workers.append(LucidWorker(
                arena_system, lucid_device,
                width, height, fps, duration_seconds,
                f"lucid_{width}x{height}_{timestamp}.avi"
            ))
        else:
            print("[Lucid] WARNING: no Lucid device matched — skipping")

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

    # ── Barrier ───────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    record_all_cameras(
        duration_seconds=90,
        width=1280,
        height=720,
        fps=15          # safe for 3 cameras on 1 GigE
    )
