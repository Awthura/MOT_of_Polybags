"""
Frame sources — OVGU AMS calibration tool.

The UI needs frames from somewhere, and "somewhere" differs by situation:

- **live camera** in the lab (Basler / Lucid / RealSense),
- **a folder of images** already captured,
- **synthetic** — a virtual camera rendering the board through a known
  projection.

The synthetic source is not a toy. It makes the entire live-capture workflow —
streaming, detection overlay, shot capture, pose coverage, calibration —
runnable and verifiable with no hardware attached, and because its true `K` is
known, the result can be checked rather than merely inspected. Building the UI
against it means the lab session starts with something already proven to work,
instead of debugging a web app next to a running conveyor.

All sources expose the same tiny interface:

    src.read()  -> BGR ndarray, or None if no frame is available
    src.close()
"""

from __future__ import annotations

import glob
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class FrameSource:
    name = "base"
    description = ""

    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @property
    def truth(self) -> dict | None:
        """Known ground truth, if this source has any (synthetic only)."""
        return None

    def device_info(self) -> dict | None:
        """Which physical unit this session is actually reading from.

        Matters wherever more than one camera of the same kind can be on the
        rig at once — two Baslers, potentially two RealSense units. A camera
        *name* like "basler_2" is just a label the operator typed in; the
        serial number is what the hardware itself reports, and it is what
        lets a calibration be tied to one specific physical camera rather than
        to whichever one happened to answer first. `None` where a source has
        no such identity to report (synthetic, folder).
        """
        return None


# ── Synthetic ────────────────────────────────────────────────────────────────

class SyntheticSource(FrameSource):
    """A virtual camera viewing the printed board through a known projection.

    Poses drift continuously so the operator can 'move the board around' exactly
    as they would in the lab, and the pose-coverage feedback behaves realistically.
    """
    name = "synthetic"
    description = "Virtual camera with known K/D — for testing without hardware"

    def __init__(self, spec, image_size=(1280, 720), seed: int = 0):
        from board import render_board, MM_PER_INCH
        import extrinsics as extr

        self.spec = spec
        self.image_size = image_size
        self._extr = extr
        self._rng = np.random.default_rng(seed)
        self._px_per_mm = 6.0
        self._board_img = np.array(render_board(spec, dpi=int(self._px_per_mm * MM_PER_INCH)))

        self.K_true = np.array([[1100.0, 0.0, 645.0],
                                [0.0, 1095.0, 355.0],
                                [0.0, 0.0, 1.0]])
        self.D_true = np.array([-0.28, 0.12, 0.001, -0.0008, 0.0])
        self._t0 = time.time()

    @property
    def truth(self) -> dict:
        return {"K": self.K_true.tolist(), "D": self.D_true.ravel().tolist(),
                "fx": float(self.K_true[0, 0]), "fy": float(self.K_true[1, 1]),
                "cx": float(self.K_true[0, 2]), "cy": float(self.K_true[1, 2])}

    def _pose(self):
        """Smoothly varying pose — several incommensurate periods so the motion
        does not repeat and coverage builds up naturally as the user waits."""
        t = time.time() - self._t0
        rx = np.radians(30 * np.sin(t * 0.37))
        ry = np.radians(30 * np.sin(t * 0.29 + 1.1))
        rz = np.radians(20 * np.sin(t * 0.19 + 2.0))
        R = (cv2.Rodrigues(np.array([0, 0, rz]))[0]
             @ cv2.Rodrigues(np.array([0, ry, 0]))[0]
             @ cv2.Rodrigues(np.array([rx, 0, 0]))[0])
        rvec = cv2.Rodrigues(R)[0]

        d = 900.0 * (1.0 + 0.25 * np.sin(t * 0.23))
        tx = 0.40 * d * np.sin(t * 0.31 + 0.5)
        ty = 0.26 * d * np.sin(t * 0.41 + 1.7)
        tvec = np.array([[tx - self.spec.width_mm / 2],
                         [ty - self.spec.height_mm / 2], [d]])
        return rvec, tvec

    def read(self):
        rvec, tvec = self._pose()
        w, h = self.image_size
        H = self._extr.homography_from_pose(self.K_true, rvec, tvec)
        H_inv = np.linalg.inv(H)

        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        pts = np.stack([xs.ravel(), ys.ravel()], -1).reshape(-1, 1, 2)
        ideal = cv2.undistortPoints(pts, self.K_true, self.D_true, P=self.K_true)
        mm = cv2.perspectiveTransform(ideal, H_inv).reshape(-1, 2)
        map_x = (mm[:, 0] * self._px_per_mm).reshape(h, w).astype(np.float32)
        map_y = (mm[:, 1] * self._px_per_mm).reshape(h, w).astype(np.float32)

        img = cv2.remap(self._board_img, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=120)
        img = np.clip(img.astype(np.float32) + self._rng.normal(0, 2.0, img.shape),
                      0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# ── Folder ───────────────────────────────────────────────────────────────────

class FolderSource(FrameSource):
    """Cycles through images in a directory — for calibrating from a set that
    was already captured, or re-running a calibration after the fact."""
    name = "folder"
    description = "Images already on disk"

    def __init__(self, folder: str, loop: bool = True):
        self.folder = Path(folder)
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        self.files = sorted(f for e in exts for f in glob.glob(str(self.folder / e)))
        if not self.files:
            raise FileNotFoundError(f"no images found in {folder}")
        self.i = 0
        self.loop = loop

    def read(self):
        if self.i >= len(self.files):
            if not self.loop:
                return None
            self.i = 0
        img = cv2.imread(self.files[self.i])
        self.current_file = self.files[self.i]
        self.i += 1
        return img


# ── Live cameras ─────────────────────────────────────────────────────────────
# Deliberately thin. Each mirrors the connection logic already proven in
# real_data/utils/record_all_5_cameras_macos.py rather than inventing a second
# way to open the same hardware. Untested against cameras — they were offline
# when this was written — so failures are surfaced rather than swallowed.

def list_realsense_devices() -> list[dict]:
    """Every RealSense currently visible over USB, serial first.

    A calibration session should never guess which of two D435 units it is
    talking to, so this exists to populate a picker rather than let
    `RealSenseSource` fall back to whichever the driver enumerates first.
    """
    import pyrealsense2 as rs
    ctx = rs.context()
    out = []
    for dev in ctx.query_devices():
        out.append({
            "serial": dev.get_info(rs.camera_info.serial_number),
            "model": dev.get_info(rs.camera_info.name),
        })
    return out


class RealSenseSource(FrameSource):
    name = "realsense"
    description = "Intel RealSense D435 (colour stream)"

    def __init__(self, width=1280, height=720, fps=15, serial: str | None = None):
        import pyrealsense2 as rs
        self._rs = rs
        devices = list_realsense_devices()
        if not devices:
            raise RuntimeError("no RealSense device found")
        if serial:
            if serial not in [d["serial"] for d in devices]:
                found = ", ".join(d["serial"] for d in devices)
                raise RuntimeError(
                    f"no RealSense with serial {serial!r} — connected: {found}")
        elif len(devices) > 1:
            # Recorded rigs run up to two D435 units (rgbd_1_color,
            # rgbd_2_color). Picking devices[0] here would calibrate whichever
            # one enumerates first, filed under whatever camera name the
            # operator typed — silently wrong the moment those two disagree.
            found = ", ".join(f"{d['serial']} ({d['model']})" for d in devices)
            raise RuntimeError(
                f"{len(devices)} RealSense devices connected ({found}) — "
                f"specify which one by serial rather than guessing")
        else:
            serial = devices[0]["serial"]

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        # int, not float: pybind11 rejects a float framerate here.
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(fps))
        self.profile = self.pipeline.start(cfg)
        self.serial = serial
        self.model = next((d["model"] for d in devices if d["serial"] == serial), None)

    def factory_intrinsics(self) -> dict | None:
        """The D435 is factory-calibrated — this is independent ground truth to
        check our own result against, which no other camera on the rig offers."""
        try:
            vs = self.profile.get_stream(self._rs.stream.color).as_video_stream_profile()
            i = vs.get_intrinsics()
            return {"fx": i.fx, "fy": i.fy, "cx": i.ppx, "cy": i.ppy,
                    "width": i.width, "height": i.height,
                    "model": str(i.model), "coeffs": list(i.coeffs)}
        except Exception:
            return None

    def device_info(self) -> dict | None:
        return {"serial": self.serial, "model": self.model}

    def read(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        c = frames.get_color_frame()
        return np.asanyarray(c.get_data()) if c else None

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def list_basler_devices() -> list[dict]:
    """Every Basler GigE camera currently visible on the network, serial first.

    This rig has *two* Basler units. `record_all_5_cameras_macos.py` names
    them `basler_1` / `basler_2` purely by `EnumerateDevices()` order — there
    is no serial pinned to either name at the recorder level either. That
    makes the serial number the only thing that actually identifies which
    physical camera a calibration came from; the name typed into this tool is
    just a label. Exists so a picker can show serials rather than the app
    guessing which device index the operator meant.
    """
    from pypylon import pylon
    tlf = pylon.TlFactory.GetInstance()
    return [{"serial": d.GetSerialNumber(), "model": d.GetModelName()}
            for d in tlf.EnumerateDevices() if "basler" in d.GetVendorName().lower()]


class BaslerSource(FrameSource):
    name = "basler"
    description = "Basler GigE via pypylon"

    def __init__(self, serial: str | None = None):
        from pypylon import pylon
        self._pylon = pylon
        tlf = pylon.TlFactory.GetInstance()
        infos = [d for d in tlf.EnumerateDevices()
                if "basler" in d.GetVendorName().lower()]
        if not infos:
            raise RuntimeError("no Basler cameras found")
        if serial:
            matches = [d for d in infos if d.GetSerialNumber() == serial]
            if not matches:
                found = ", ".join(d.GetSerialNumber() for d in infos)
                raise RuntimeError(
                    f"no Basler camera with serial {serial!r} — "
                    f"connected: {found}")
            info = matches[0]
        elif len(infos) > 1:
            # Two Baslers on this rig, disambiguated by nothing but
            # EnumerateDevices() order — which is also all the recorder uses
            # to tell basler_1 from basler_2. Silently taking infos[0] here
            # would calibrate whichever camera enumerates first and file it
            # under whatever name the operator happened to type in, with
            # nothing to catch the two disagreeing.
            found = ", ".join(f"{d.GetSerialNumber()} ({d.GetModelName()})"
                              for d in infos)
            raise RuntimeError(
                f"{len(infos)} Basler cameras connected ({found}) — specify "
                f"which one by serial rather than guessing")
        else:
            info = infos[0]

        self.camera = pylon.InstantCamera(tlf.CreateDevice(info))
        self.camera.Open()
        self.serial = info.GetSerialNumber()
        self.model = info.GetModelName()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed

    def device_info(self) -> dict | None:
        return {"serial": self.serial, "model": self.model}

    def read(self):
        grab = self.camera.RetrieveResult(3000, self._pylon.TimeoutHandling_Return)
        if not grab or not grab.GrabSucceeded():
            return None
        img = self.converter.Convert(grab).Array
        grab.Release()
        return img

    def close(self):
        try:
            self.camera.StopGrabbing()
            self.camera.Close()
        except Exception:
            pass


class LucidSource(FrameSource):
    name = "lucid"
    description = "Lucid Triton GigE via Aravis"

    # GenICam PixelFormat strings we can hand to OpenCV, best first. Triton
    # color units otherwise come up in whatever format was last persisted on
    # the device (often Mono8) — if we don't set one explicitly, read() has
    # no reliable way to tell "true Mono8" apart from "raw undebayered Bayer
    # plane", and both are a single byte/pixel.
    _PREFERRED_FORMATS = [
        "BGR8", "RGB8", "BayerRG8", "BayerGB8", "BayerGR8", "BayerBG8",
    ]
    # GenICam pattern -> OpenCV constant, via the four-letter aliases, which
    # spell out the full first-two-rows tile and are therefore unambiguous:
    # GenICam BayerRG8 means a tile of R G / G B = "RGGB". Do NOT use the
    # letter-for-letter two-letter names (COLOR_BayerRG2BGR etc.) — OpenCV's
    # legacy two-letter names refer to the pattern at a DIFFERENT tile corner
    # than GenICam's, so that obvious-looking mapping swaps red and blue on
    # every frame. Verified empirically; the recorders in real_data/utils
    # carry the same fix and the full derivation.
    _BAYER_TO_CV = {
        "BayerRG8": cv2.COLOR_BAYER_RGGB2BGR,   # == COLOR_BAYER_BG2BGR (46)
        "BayerGB8": cv2.COLOR_BAYER_GBRG2BGR,   # == COLOR_BAYER_GR2BGR (49)
        "BayerGR8": cv2.COLOR_BAYER_GRBG2BGR,   # == COLOR_BAYER_GB2BGR (47)
        "BayerBG8": cv2.COLOR_BAYER_BGGR2BGR,   # == COLOR_BAYER_RG2BGR (48)
    }

    def __init__(self, device_id: str | None = None):
        import gi
        gi.require_version("Aravis", "0.8")
        from gi.repository import Aravis
        self._Aravis = Aravis
        Aravis.update_device_list()
        if device_id is None:
            for i in range(Aravis.get_n_devices()):
                vendor = Aravis.get_device_vendor(i) or ""
                model = Aravis.get_device_model(i) or ""
                if "lucid" in (vendor + model).lower():
                    device_id = Aravis.get_device_id(i)
                    break
        if device_id is None:
            raise RuntimeError("no Lucid device found")
        self.device_id = device_id
        self.camera = Aravis.Camera.new(device_id)
        self.pixel_format = self._configure_pixel_format()
        try:
            self.camera.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)
            self.stream = self.camera.create_stream(None, None)
            payload = self.camera.get_payload()
            for _ in range(10):
                self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
            self.camera.start_acquisition()
        except Exception as e:
            # "Controller privilege required for streaming control" — GigE
            # Vision allows exactly one controlling process per camera. Name
            # the actual fix rather than surfacing a bare GVCP error.
            raise RuntimeError(
                f"could not start streaming from the Lucid ({e}) — another "
                f"process probably holds control of this camera. Close any "
                f"other recorder/viewer session (including a stale app.py or "
                f"ArenaView on another machine) and retry.") from e

    def _configure_pixel_format(self) -> str:
        try:
            available = list(self.camera.dup_available_pixel_formats_as_strings())
        except Exception:
            available = []
        for fmt in self._PREFERRED_FORMATS:
            if available and fmt not in available:
                continue
            try:
                self.camera.set_pixel_format_from_string(fmt)
                return fmt
            except Exception:
                # A write can fail even for a format the camera supports:
                # GigE Vision grants CONTROLLER access to exactly one process,
                # and everyone else is read-only. Diagnosed live on this rig
                # (TRI032S-C): every PixelFormat write returned access-denied
                # while another process held the camera. Fall through and use
                # whatever format is actually active — read() decodes Bayer
                # correctly either way.
                continue
        # Nothing writable (or genuinely mono-only) — read back what the
        # camera is actually in rather than assuming.
        active = self.camera.get_pixel_format_as_string()
        print(f"[lucid] could not set any pixel format — camera is in "
              f"'{active}' (another process may hold GigE controller access)")
        return active

    def device_info(self) -> dict | None:
        # Only one physical Lucid is on this rig, found by vendor/model
        # substring rather than a picked serial — reported here anyway so a
        # saved calibration still carries something to cross-check if that
        # ever changes.
        return {"serial": self.device_id, "model": None,
                "pixel_format": self.pixel_format}

    def read(self):
        buf = self.stream.timeout_pop_buffer(2_000_000)
        if buf is None:
            return None
        try:
            if buf.get_status() != self._Aravis.BufferStatus.SUCCESS:
                return None
            w, h = buf.get_image_width(), buf.get_image_height()
            arr = np.frombuffer(buf.get_data(), dtype=np.uint8)
            fmt = self.pixel_format
            if fmt == "BGR8":
                img = arr[: h * w * 3].reshape(h, w, 3).copy()
            elif fmt == "RGB8":
                rgb = arr[: h * w * 3].reshape(h, w, 3)
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            elif fmt in self._BAYER_TO_CV:
                bayer = arr[: h * w].reshape(h, w)
                img = cv2.cvtColor(bayer, self._BAYER_TO_CV[fmt])
            else:
                mono = arr[: h * w].reshape(h, w).copy()
                img = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
            return img
        finally:
            self.stream.push_buffer(buf)

    def close(self):
        try:
            self.camera.stop_acquisition()
        except Exception:
            pass


AVAILABLE = {
    "synthetic": SyntheticSource,
    "folder": FolderSource,
    "realsense": RealSenseSource,
    "basler": BaslerSource,
    "lucid": LucidSource,
}


_DEVICE_LISTERS = {"basler": list_basler_devices, "realsense": list_realsense_devices}


def probe() -> list[dict]:
    """Which sources can actually be used right now, and why not if not.

    Where more than one physical unit of a kind can be on the rig at once —
    both Baslers, potentially two RealSense — also enumerates the connected
    devices by serial, so the UI can offer a picker instead of the source
    silently guessing which camera was meant. Enumeration failures (SDK
    present but nothing plugged in, USB not permissioned yet) are swallowed
    here: they surface properly once a session actually tries to open one.
    """
    out = [
        {"id": "synthetic", "label": "Synthetic camera (no hardware)",
         "available": True, "note": "Known K/D — use to test the whole flow",
         "devices": []},
        {"id": "folder", "label": "Folder of images",
         "available": True, "note": "Point at a directory of captures",
         "devices": []},
    ]
    for sid, label, mod in (("realsense", "Intel RealSense D435", "pyrealsense2"),
                            ("basler", "Basler GigE", "pypylon"),
                            ("lucid", "Lucid Triton (Aravis)", "gi")):
        devices = []
        try:
            __import__(mod)
            ok, note = True, "SDK present — camera must be connected"
            lister = _DEVICE_LISTERS.get(sid)
            if lister:
                try:
                    devices = lister()
                    if len(devices) > 1:
                        note = f"{len(devices)} connected — pick one by serial"
                    elif len(devices) == 1:
                        note = f"1 connected: S/N {devices[0]['serial']}"
                    else:
                        note = "SDK present — no device found"
                except Exception:
                    pass
        except ImportError:
            ok, note = False, f"{mod} not installed"
        out.append({"id": sid, "label": label, "available": ok, "note": note,
                    "devices": devices})
    return out
