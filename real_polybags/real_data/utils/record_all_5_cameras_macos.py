"""
Combined multi-camera recorder for macOS: 2x Basler (pypylon) + 1x Lucid
(Aravis, not Arena SDK — no macOS support there) + RealSense RGBD
(pyrealsense2).

Defaults to a 4-camera rig (2 Basler + 1 Lucid + 1 RGBD), matching the current
hardware after one RealSense was removed. Pass --max-realsense 2 if a second
unit is refitted, but see CAPTURE_INSTRUCTIONS.md issue 2 first: the two D435s
interfere at device-open time on a shared USB controller.

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

Run (from the output directory, e.g. raw_recordings/):
    sudo /opt/anaconda3/envs/ams/bin/python ../utils/record_all_5_cameras_macos.py \
        --fps 15 --duration 90 --verbose

RealSense needs `sudo` on macOS (USB power-state permission; Basler/Lucid are
GigE and don't). DYLD_LIBRARY_PATH is NOT needed any more and should not be
set — Aravis now resolves its libraries from symlinks inside the conda env,
which is what makes it survive sudo (dyld strips DYLD_* for privileged
processes). See CAPTURE_INSTRUCTIONS.md "Known issues" 1.

Timing caveat: the .avi files carry the TARGET fps in their header, so any
camera that runs below target plays back too fast (Lucid at 8.5 fps against a
15 fps target plays 1.76x fast). Each run writes timestamps_<camera>_*.csv and
recording_metadata_*.json with the real per-frame capture times — use those for
anything temporal, not the video timebase. fix_video_timing.py can remux the
files to their true rate losslessly.

Known hardware issue: two RealSense units on one USB controller interfere at
device-open time; use --max-realsense 1 for a reliable take. See
CAPTURE_INSTRUCTIONS.md "Known issues" 2.
"""

import argparse
import json
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
        self.end_time     = None   # set on capture-loop exit; freezes elapsed()
        self.frame_times  = []     # host wall-clock time per written frame
        self.device_times = []     # camera's own timestamp per frame (may be None)
        # Unit/epoch of device_times, set by each worker once it knows. Recorded
        # verbatim rather than normalised, because the SDKs disagree: Aravis
        # reports nanoseconds on the device clock, pylon reports device ticks
        # whose frequency is model-dependent, and librealsense reports
        # milliseconds in one of several domains. Converting blindly here would
        # silently produce wrong numbers; downstream tooling is told the domain
        # and converts explicitly.
        self.ts_domain    = None
        # Setup and the actual synchronized start are decoupled: each worker
        # signals setup_done (success or failure) independently, so a camera
        # that fails to connect doesn't block the ones that did — replaces
        # a fixed-size threading.Barrier, which would time out (and take
        # every other camera down with it) if even one worker never reached
        # it.
        self.setup_done   = threading.Event()
        self.setup_ok     = False
        self.go_event     = None
        # Safety net only — the coordinator is expected to always signal
        # go_event. This MUST stay comfortably larger than the coordinator's
        # total setup budget: a worker whose go-wait expires *before* the
        # coordinator has finished deciding will exit having recorded nothing,
        # even though it connected fine. That is exactly what happened when
        # this timeout and the coordinator's were both 35s — the workers
        # started their countdown first (at their own ready moment) and so
        # lost the race by a few seconds, producing a "4/5 cameras ready"
        # run in which all 4 reported Frames: 0. set_go_event() derives it.
        self.go_timeout   = 300.0

    def set_go_event(self, go_event, go_timeout=None):
        self.go_event = go_event
        if go_timeout is not None:
            self.go_timeout = go_timeout

    def mark_ready(self, ok):
        if not self.setup_done.is_set():
            self.setup_ok = ok
            self.setup_done.set()

    def wait_for_go(self, timeout=None):
        """Call after setup succeeds. Returns False if the go signal never
        arrives (e.g. another camera's setup took too long)."""
        self.mark_ready(True)
        if self.go_event is None:
            return True
        return self.go_event.wait(
            timeout=self.go_timeout if timeout is None else timeout)

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

    def _write_frame(self, frame, device_ts=None):
        """Write a frame and record when it was captured.

        Two clocks are recorded per frame, because neither alone is enough:

        `host` (time.time()) is sampled here, i.e. AFTER the frame has crossed
        the wire, been format-converted and been handed to the VideoWriter. It
        therefore bundles transfer + conversion latency into what is supposed
        to be a capture time, and that latency differs per camera and varies
        frame to frame. It is a shared reference across cameras (one host
        records all of them) but a noisy one.

        `device_ts` is the camera's OWN timestamp for the exposure, supplied by
        the caller because only each worker knows its SDK's call. It is precise
        but sits on that camera's own clock, so it is not directly comparable
        across cameras.

        Pairing them is what makes ~10-30ms cross-camera alignment possible
        without PTP: the device clock gives precision, the host clock gives a
        common origin to map the device clocks onto. Both are written to
        timestamps_<camera>_<run>.csv.
        """
        if frame is not None and frame.size > 0:
            self.out.write(frame)
            self.frame_times.append(time.time())
            self.device_times.append(device_ts)
            self.frame_count += 1
            with self.lock:
                self.latest_frame = frame.copy()

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def elapsed(self):
        """Seconds spent recording. Freezes once the worker stops.

        Previously this always used a live time.time(), so the clock kept
        running after the capture loop exited while frame_count stayed fixed.
        The end-of-run summary prints after teardown and thread joins, several
        seconds later, so every camera's fps was reported meaningfully lower
        there than on its own "Done" line — e.g. a camera that actually hit
        14.9 fps (149 frames in 10s) was summarised as 12.6.
        """
        if not self.start_time:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def mark_finished(self):
        if self.end_time is None:
            self.end_time = time.time()

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

                # Read the device timestamp BEFORE Release() — the grab result
                # is returned to the pool there and must not be touched after.
                try:
                    dev_ts = grab.TimeStamp
                    if self.ts_domain is None:
                        # pylon reports device ticks; the tick frequency is
                        # model-dependent (and changes when PTP is enabled), so
                        # record the raw value and let downstream convert.
                        self.ts_domain = "basler_device_ticks"
                except Exception:
                    dev_ts = None

                converted = converter.Convert(grab)
                frame = converted.Array
                grab.Release()
                self._write_frame(frame, device_ts=dev_ts)

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
            self.mark_finished()   # freeze the fps clock before teardown
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
    def __init__(self, device_id, width, height, fps, duration, output_file,
                 packet_size=1400, packet_delay_ns=40000):
        super().__init__("Lucid", width, height, fps, duration, output_file)
        self.device_id   = device_id
        self.packet_size = packet_size
        self.packet_delay_ns = packet_delay_ns
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

            # GigE transport tuning — the main lever on Lucid's frame rate.
            #
            # Lucid reproducibly captured exactly 85 frames in 10s (8.5 fps)
            # against a 15 fps target, identically across four runs — a hard
            # configuration ceiling, not jitter. The inter-packet delay is the
            # prime suspect:
            #
            #   1280x720 BGR8 = 2,764,800 B / 1400 B per packet = ~1975 packets
            #   1975 packets x 40us delay = 79 ms/frame => 12.7 fps ceiling,
            #   before any actual transmission or processing time.
            #
            # Packet delay exists to stop multiple GigE cameras from bursting
            # into a shared switch and dropping frames, so it can't simply be
            # zeroed on a 5-camera rig — but 40us is worth tuning against
            # measured frame rate. Both values are CLI-exposed for that.
            #
            # These previously sat behind bare `except: pass`, so a silent
            # failure was indistinguishable from success; now the outcome and
            # the resulting theoretical ceiling are printed.
            try:
                camera.gv_set_packet_size(self.packet_size)
            except Exception as e:
                print(f"[{self.name}] gv_set_packet_size({self.packet_size}) failed: {e}")
            try:
                camera.gv_set_packet_delay(self.packet_delay_ns)
            except Exception as e:
                print(f"[{self.name}] gv_set_packet_delay({self.packet_delay_ns}) failed: {e}")

            time.sleep(0.5)
            stream = camera.create_stream(None, None)
            payload = camera.get_payload()

            try:
                ps  = camera.gv_get_packet_size()
                pd  = camera.gv_get_packet_delay()
                pkts = -(-payload // ps) if ps else 0        # ceil division
                delay_s = pkts * (pd / 1e9) if pd else 0.0
                ceiling = (1.0 / delay_s) if delay_s > 0 else float('inf')
                print(f"[{self.name}] GigE: packet_size={ps}B delay={pd}ns "
                      f"payload={payload}B (~{pkts} packets/frame)")
                print(f"[{self.name}] delay-implied fps ceiling: "
                      f"{ceiling:.1f} (target {self.fps})")
                if ceiling < self.fps:
                    print(f"[{self.name}] WARNING: packet delay alone caps this "
                          f"below target — lower --lucid-packet-delay or raise "
                          f"--lucid-packet-size")
            except Exception as e:
                print(f"[{self.name}] could not read back GigE settings: {e}")
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
                    # Device timestamp must be read while we still hold the
                    # buffer — it goes back to the stream pool on push_buffer().
                    try:
                        dev_ts = buf.get_timestamp()
                        if self.ts_domain is None:
                            self.ts_domain = "aravis_device_ns"
                    except Exception:
                        dev_ts = None

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
                self._write_frame(frame, device_ts=dev_ts)

        except Exception as e:
            self.error = str(e)
            self.mark_ready(False)
            print(f"[{self.name}] ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.mark_finished()   # freeze the fps clock before teardown
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
    # Depth is requested from the sensor at `depth_width`x`depth_height`, which
    # is deliberately NOT tied to the colour resolution: _run() aligns depth to
    # the colour stream (rs.align(rs.stream.color)), so the aligned depth frame
    # — and therefore the written depth video — comes out at colour resolution
    # regardless of what the sensor streamed. That makes the depth stream size a
    # free lever on USB bandwidth.
    #
    # It matters because two D435s on one USB controller at 1280x720 depth +
    # colour is roughly 138 MB/s of payload, and one camera loses the bus:
    # "Frame didn't arrive within 5000" during the warmup. 848x480 is the D435's
    # native depth resolution and saves ~15 MB/s per camera for no loss in
    # output — depth detail beyond the sensor's real resolution was interpolated
    # anyway.
    def __init__(self, index, width, height, fps, duration, color_file, depth_file,
                 depth_width=None, depth_height=None):
        super().__init__(f"RGBD_{index + 1}", width, height, fps, duration, color_file)
        self.index        = index
        self.depth_file   = depth_file
        self.depth_out    = None
        # Default to the colour resolution — i.e. the original behaviour.
        # Lowering it was tried as a fix for the dual-RealSense failure and the
        # evidence contradicted it (848x480 run: both cameras failed;
        # 1280x720 run: one camera recorded 152 frames), so it is NOT the
        # default. Kept as a tuning knob only.
        self.depth_width  = depth_width or width
        self.depth_height = depth_height or height

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
                # Depth at its own (smaller) resolution to save USB bandwidth —
                # align.process() resamples it to colour resolution anyway, so
                # the written depth video is unaffected. See class docstring.
                config.enable_stream(rs.stream.depth, self.depth_width,
                                     self.depth_height, rs.format.z16, rs_fps)
                try:
                    profile = pipeline.start(config)
                    break
                except RuntimeError as e:
                    last_err = e
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                    if attempt == 3:
                        # Don't sleep after the final attempt — the old code
                        # burned 8s here before raising, for no gain, and that
                        # alone pushed this worker's setup past the
                        # coordinator's budget.
                        print(f"[{self.name}] pipeline.start() failed ({e}) "
                              f"on final attempt 4/4")
                        break
                    wait_s = 1.5 * (attempt + 1)   # 1.5 + 3.0 + 4.5 = 9s total
                    print(f"[{self.name}] pipeline.start() failed ({e}), "
                          f"retrying in {wait_s:.1f}s (attempt {attempt + 1}/4)")
                    time.sleep(wait_s)
            if profile is None:
                raise RuntimeError(f"pipeline.start() failed after 4 attempts: {last_err}")

            dev_name = profile.get_device().get_info(rs.camera_info.name)
            print(f"[{self.name}] RealSense connected: {dev_name} (S/N: {serial}) "
                  f"Target FPS: {self.fps}")

            # Ask librealsense to report frame timestamps already mapped onto
            # the host clock (GLOBAL_TIME domain). This is the one camera on the
            # rig that can do the device->host clock mapping for us; without it
            # timestamps come back on HARDWARE_CLOCK, an arbitrary device epoch
            # that can't be compared to anything else.
            try:
                for s in profile.get_device().query_sensors():
                    if s.supports(rs.option.global_time_enabled):
                        s.set_option(rs.option.global_time_enabled, 1)
                print(f"[{self.name}] global_time enabled (device clock mapped to host)")
            except Exception as e:
                print(f"[{self.name}] could not enable global_time: {e}")

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

                try:
                    dev_ts = color_frame.get_timestamp()   # ms
                    if self.ts_domain is None:
                        self.ts_domain = str(color_frame.get_frame_timestamp_domain())
                except Exception:
                    dev_ts = None

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())

                self._write_frame(color, device_ts=dev_ts)
                self._write_depth(depth)

        except Exception as e:
            self.error = str(e)
            self.mark_ready(False)
            print(f"[{self.name}] ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.mark_finished()   # freeze the fps clock before teardown
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
    # Grid sized to the actual camera count rather than a fixed 3 columns, so
    # no blank padding tiles appear. With the rig now at 4 cameras (2 Basler +
    # 1 Lucid + 1 RGBD) a fixed 3-column layout padded to 6 tiles, i.e. two
    # dead black panels. Near-square: 1->1x1, 2->2x1, 4->2x2 exactly,
    # 5->3x2 (one blank), 6->3x2 exact.
    import math
    PREVIEW_W, PREVIEW_H = 480, 270
    EXP_STEP = 0.20

    N    = max(len(workers), 1)
    COLS = math.ceil(math.sqrt(N))
    ROWS = math.ceil(N / COLS)

    lucid_worker = next((w for w in workers if isinstance(w, LucidWorker)), None)
    last_good    = {w.name: None for w in workers}
    last_report  = time.time()

    print("[Display] Waiting for all cameras to start recording...")
    while True:
        if all(w.start_time is not None or w.error is not None for w in workers):
            break
        waiting = np.full((PREVIEW_H * ROWS, PREVIEW_W * COLS, 3), 30, dtype=np.uint8)
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
def record_all_cameras(duration_seconds, width, height, fps, verbose,
                       max_realsense=1, depth_width=None, depth_height=None,
                       lucid_packet_size=1400, lucid_packet_delay=40000):
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
                f"lucid_{width}x{height}_{timestamp}.avi",
                packet_size=lucid_packet_size,
                packet_delay_ns=lucid_packet_delay,
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
    # `max_realsense` exists because a RealSense in a bad USB state can
    # SIGSEGV inside libusb (null deref in darwin_submit_transfer, reached via
    # librealsense's usb_messenger_libusb::control_transfer). That kills the
    # whole process, taking the GigE cameras down with it, and it cannot be
    # caught from Python. Capping the count lets a session proceed with the
    # cameras that do work instead of losing everything to one flaky unit.
    if REALSENSE_AVAILABLE and max_realsense > 0:
        print("\n=== Enumerating RealSense cameras ===")
        ctx  = rs.context()
        devs = ctx.query_devices()
        print(f"Found {len(devs)} RealSense camera(s)")
        if len(devs) > max_realsense:
            print(f"  (using only the first {max_realsense} — --max-realsense)")
        for i, dev in enumerate(devs[:max_realsense]):
            sn = dev.get_info(rs.camera_info.serial_number)
            print(f"[RGBD_{i+1}] Prepared: S/N {sn}")
            workers.append(RGBDWorker(
                i, width, height, fps, duration_seconds,
                f"rgbd_{i+1}_color_{width}x{height}_{timestamp}.avi",
                f"rgbd_{i+1}_depth_{width}x{height}_{timestamp}.avi",
                depth_width=depth_width, depth_height=depth_height,
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
    # One wall-clock budget for the whole setup phase, shared across all
    # cameras — deliberately not per-camera. With a per-camera timeout the
    # worst case was 35s x n_cameras, during which cameras that were already
    # connected sat waiting and then gave up. Workers get a go-wait timeout
    # derived from this budget so they cannot expire before this thread has
    # decided whether to start.
    SETUP_BUDGET = 45.0

    go_event = threading.Event()
    for w in workers:
        w.set_go_event(go_event, go_timeout=SETUP_BUDGET + 60.0)

    print(f"\nConnecting to {len(workers)} camera(s), target FPS: {fps}...")
    for w in workers:
        w.running = True
        w.thread  = threading.Thread(target=w._run, daemon=True)
        w.thread.start()

    print(f"Waiting for all cameras to finish setup "
          f"({SETUP_BUDGET:.0f}s total budget)...")
    setup_deadline = time.time() + SETUP_BUDGET
    for w in workers:
        remaining = max(0.0, setup_deadline - time.time())
        if not w.setup_done.wait(timeout=remaining):
            print(f"[{w.name}] setup did not report back in time — treating as failed")
            w.mark_ready(False)

    ready_workers  = [w for w in workers if w.setup_ok]
    failed_workers = [w for w in workers if not w.setup_ok]

    if failed_workers:
        print(f"\nWARNING: {len(failed_workers)} camera(s) failed setup and will be "
              f"skipped: {[w.name for w in failed_workers]}")
    if not ready_workers:
        print("ERROR: no cameras successfully initialized — aborting")
        go_event.set()  # unblock any worker still parked in wait_for_go()
        return

    print(f"{len(ready_workers)}/{len(workers)} camera(s) ready — "
          f"starting synchronized recording for {duration_seconds}s...\n")
    go_event.set()

    display_loop(ready_workers, duration_seconds, verbose)

    for w in workers:
        w.stop()
    for w in workers:
        w.join()

    # ── Timing metadata ───────────────────────────────────────────────────────
    # Written because the .avi files alone are not a faithful time record: the
    # container stores one fixed fps (the target), so any camera that ran below
    # target plays back too fast and its frame indices don't map linearly to
    # real time. Anything temporal downstream should use these files, not the
    # video's own timebase.
    meta = {
        "timestamp": timestamp,
        "target_fps": fps,
        "requested_duration_s": duration_seconds,
        "width": width, "height": height,
        "depth_stream": [depth_width or width, depth_height or height],
        "cameras": {},
    }
    for w in workers:
        ts = w.frame_times
        drift = None
        if len(ts) > 1:
            # Max deviation of actual frame times from a perfectly even
            # target-rate grid — how far the camera strayed from uniform pacing.
            span = ts[-1] - ts[0]
            ideal = span / (len(ts) - 1)
            drift = max(abs((ts[i] - ts[0]) - i * ideal) for i in range(len(ts)))
        meta["cameras"][w.name] = {
            "color_file": w.output_file,
            "depth_file": getattr(w, "depth_file", None),
            "frames": w.frame_count,
            "incomplete": w.incomplete,
            "actual_fps": round(w.actual_fps(), 3),
            "recorded_s": round(w.elapsed(), 3),
            "start_unix": w.start_time,
            "end_unix": w.end_time,
            "playback_speed_error": (round(w.actual_fps() / fps, 3) if fps else None),
            "max_pacing_drift_s": (round(drift, 4) if drift is not None else None),
            "error": w.error,
        }
        meta["cameras"][w.name]["timestamp_domain"] = w.ts_domain
        meta["cameras"][w.name]["device_timestamps_present"] = bool(
            [d for d in w.device_times if d is not None])
        if ts:
            ts_file = f"timestamps_{w.name}_{timestamp}.csv"
            dev = w.device_times
            with open(ts_file, "w") as f:
                f.write("frame_index,host_unix_time,host_seconds_from_start,"
                        "device_timestamp,device_delta\n")
                d0 = next((d for d in dev if d is not None), None)
                for i, t in enumerate(ts):
                    d = dev[i] if i < len(dev) else None
                    # device_delta is relative to this camera's own first frame,
                    # which makes the column readable without knowing the tick
                    # rate or epoch. Absolute device_timestamp is kept verbatim
                    # so nothing is lost to the conversion.
                    dd = "" if (d is None or d0 is None) else f"{d - d0}"
                    ds = "" if d is None else f"{d}"
                    f.write(f"{i},{t:.6f},{t - ts[0]:.6f},{ds},{dd}\n")
            meta["cameras"][w.name]["timestamps_file"] = ts_file

    meta_file = f"recording_metadata_{timestamp}.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*70}")
    print(f"All recordings complete! (target FPS: {fps})")
    for w in workers:
        extra = f"  depth→ {w.depth_file}" if isinstance(w, RGBDWorker) else ""
        speed = w.actual_fps() / fps if fps else 1.0
        # Flag files whose playback speed is materially wrong, so a camera
        # silently running slow doesn't quietly corrupt downstream timing.
        warn = "" if 0.95 <= speed <= 1.05 else f"  <- PLAYS {1/speed:.2f}x TOO FAST"
        print(f"  {w.name:10} | {w.frame_count:5} frames | "
              f"{w.actual_fps():5.1f} fps actual | {w.incomplete} incomplete | "
              f"{w.output_file}{extra}{warn}")
    print(f"{'='*70}")
    print(f"Timing metadata → {meta_file}")
    print("Use the timestamps_*.csv files for any temporal analysis; the .avi "
          "timebase is the target fps, not the real one.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Synchronized 5-camera recorder (Basler + Lucid + RealSense)")
    ap.add_argument('--fps', type=float, default=15.0, help="Target FPS for ALL cameras")
    ap.add_argument('--duration', type=int, default=90, help="Recording duration in seconds")
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--verbose', action='store_true', help="Print per-camera actual FPS every 5s")
    ap.add_argument('--max-realsense', type=int, default=1, choices=(0, 1, 2),
                    help="How many RealSense cameras to use (default 1, matching the "
                         "current 4-camera rig: 2 Basler + 1 Lucid + 1 RGBD). Raise to "
                         "2 only if a second unit is refitted — note the two D435s "
                         "interfere at device-open time on a shared USB controller "
                         "(see CAPTURE_INSTRUCTIONS.md issue 2), and a wedged unit can "
                         "SIGSEGV inside libusb, which cannot be caught from Python "
                         "and takes the Basler/Lucid cameras down with it.")
    ap.add_argument('--depth-width', type=int, default=None,
                    help="Depth STREAM width (default: same as --width). Independent "
                         "of --width because depth is aligned to the colour stream, so "
                         "the written depth video is at colour resolution regardless. "
                         "848 is the D435's native depth resolution and cuts USB "
                         "bandwidth ~15MB/s per camera. NOTE: this did NOT fix the "
                         "dual-RealSense failure — that is open-time contention, not "
                         "bandwidth — so it is a tuning knob, not a default.")
    ap.add_argument('--depth-height', type=int, default=None,
                    help="Depth STREAM height (default: same as --height). See "
                         "--depth-width.")
    ap.add_argument('--lucid-packet-size', type=int, default=1400,
                    help="Lucid GigE packet size in bytes (default 1400). Larger "
                         "means fewer packets per frame, so less total inter-packet "
                         "delay. Needs a matching network MTU: 1400 is safe on a "
                         "standard 1500-MTU link; ~8000 needs jumbo frames enabled "
                         "on the NIC AND the switch.")
    ap.add_argument('--lucid-packet-delay', type=int, default=40000,
                    help="Lucid GigE inter-packet delay in NANOSECONDS (default "
                         "40000 = 40us). This is the main throttle on Lucid's frame "
                         "rate: ~1975 packets/frame x 40us = 79ms, a 12.7 fps "
                         "ceiling. It exists to stop multiple GigE cameras "
                         "saturating a shared switch, so reduce it while watching "
                         "the 'incomplete' frame count for dropped packets.")
    args = ap.parse_args()

    record_all_cameras(
        duration_seconds=args.duration,
        width=args.width,
        height=args.height,
        fps=args.fps,
        verbose=args.verbose,
        max_realsense=args.max_realsense,
        depth_width=args.depth_width,
        depth_height=args.depth_height,
        lucid_packet_size=args.lucid_packet_size,
        lucid_packet_delay=args.lucid_packet_delay,
    )
