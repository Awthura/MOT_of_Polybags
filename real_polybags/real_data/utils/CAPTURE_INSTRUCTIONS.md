# Multi-camera capture — setup & run instructions

## Scripts

- `record_basler_lucid_rgbd.py` — supervisor's original (Windows/Linux only;
  uses Arena SDK for Lucid, which has no macOS build).
- `record_basler_lucid_rgbd_macos.py` — macOS variant, 2 Basler + 1 Lucid +
  2 RGBD, Lucid via Aravis instead of Arena SDK. Not yet tested against real
  RealSense hardware end-to-end (RGBD path is `pyrealsense2`, unchanged from
  the original).
- `record_basler_lucid_10_macos.py` — macOS, 2 Basler + 1 Lucid only (no
  RGBD). **Verified working against real hardware** — 772/1081/1339 frames
  recorded successfully across Lucid/Basler_1/Basler_2 in one full run.
- `record_real_sense_dual.py` — RealSense-only, 2 cameras, split out
  separately since RealSense needs `sudo` on macOS and Basler/Lucid don't.
  Includes a retry-with-fresh-pipeline fix for the "failed to set power
  state" / multi-camera USB bug (see Known Issues below).
- `record_all_5_cameras_macos.py` — **current combined script**, all 5
  cameras, `--fps`/`--duration`/`--width`/`--height`/`--verbose` CLI args.
  Uses a per-camera setup-then-go-signal mechanism instead of a fixed-size
  barrier, so one camera failing to connect no longer blocks/kills the
  others — verified in isolation with simulated workers, not yet run
  end-to-end with all 5 real cameras.

**Use `_macos` scripts on the MacBook. Use the originals on Windows/Linux.**

## One-time environment setup (macOS, `ams` conda env)

```bash
brew install aravis pygobject3

conda activate ams
pip install pypylon
pip install pygobject
conda install -c conda-forge pyrealsense2 -y
```

| Camera | SDK used on macOS | Status |
|---|---|---|
| Basler ×2 | `pypylon` | **Verified working** against real hardware |
| Lucid | Aravis (not Arena SDK) | **Verified working** against real hardware |
| RGBD ×2 | `pyrealsense2` (conda-forge) | Verified import/enumeration; multi-camera "power state" issue, see below |

## Every session (macOS)

Aravis needs Homebrew's GLib/GObject libraries visible from inside the conda
env — required every new shell, for any script that touches Lucid:

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
```

Consider adding that line to `~/.zshrc` if recording sessions are frequent.

## Running

```bash
conda activate ams
mkdir -p /Users/awthura/OVGU/AMS/real_polybags/real_data/raw_recordings
cd /Users/awthura/OVGU/AMS/real_polybags/real_data/raw_recordings

# Basler + Lucid only (no sudo needed — both are GigE, not USB):
python ../utils/record_basler_lucid_10_macos.py

# All 5 cameras, target FPS + verbose per-camera FPS reporting:
sudo -E /opt/anaconda3/envs/ams/bin/python ../utils/record_all_5_cameras_macos.py --fps 15 --duration 90 --verbose
```

Always run from `raw_recordings/` (or any dedicated folder) so output `.avi`
files land there, not in `utils/`.

`record_all_5_cameras_macos.py`'s `--fps`/`--duration`/`--width`/`--height`
replace what used to be hardcoded in the older scripts' `__main__` blocks —
no file editing needed for those on the combined script.

## Known issues

### 1. RealSense needs `sudo` — but `sudo` breaks Aravis/Lucid

Since macOS Monterey, `pyrealsense2` needs root to claim the USB device and
set its power state ("failed to set power state" without `sudo`). But when
running under `sudo -E`, Aravis fails to load GLib/GObject even with
`DYLD_LIBRARY_PATH` exported — macOS's dynamic linker strips `DYLD_*`
environment variables for privilege-elevated processes; this is not
something `sudo`'s `env_keep` config can override, since it's dyld's own
behavior, not sudo's environment handling.

**Fix to try** (needs your password, hence not done automatically): symlink
the needed libraries into `/usr/local/lib`, which is one of dyld's built-in
default fallback search paths — checked regardless of `DYLD_LIBRARY_PATH` or
privilege level, so it should survive `sudo`:

```bash
sudo mkdir -p /usr/local/lib
sudo ln -sf /opt/homebrew/lib/libglib-2.0.0.dylib /usr/local/lib/
sudo ln -sf /opt/homebrew/lib/libgobject-2.0.0.dylib /usr/local/lib/
sudo ln -sf /opt/homebrew/opt/glib/lib/libgirepository-2.0.0.dylib /usr/local/lib/
```

Then retest:
```bash
sudo -E /opt/anaconda3/envs/ams/bin/python -c "
import gi
gi.require_version('Aravis', '0.8')
from gi.repository import Aravis
print('Aravis OK under sudo')
"
```

If it still fails, paste the exact error — there may be one or two more
libraries in the dependency chain to symlink the same way (check with
`otool -L` on the failing library).

**Workaround if the above doesn't pan out**: run Basler+Lucid (no sudo) and
RealSense (sudo) as two separate recording sessions instead of one combined
5-camera run — you lose cross-camera start-time sync between the two groups,
but each group individually works today.

### 2. RealSense: two cameras, second one fails to start

Known upstream librealsense/macOS bug ("failed to set power state" or "UVC
device is already opened!"), especially over a shared USB hub. Mitigated in
`record_real_sense_dual.py` and `record_all_5_cameras_macos.py`'s `RGBDWorker`
with a retry loop (4 attempts, fresh `pipeline`/`config` each time, 2s+
backoff) — this helps with the intermittent timing version of the bug, but
if the two cameras share a hub, separate USB ports (or a powered hub) is the
more fundamental fix. **Still open**: confirm whether both RealSense cameras
are on the same hub or separate direct ports.

If a run gets interrupted (`Ctrl+C`) partway through RealSense initialization,
the USB device can get stuck in a bad state — **physically unplug and replug**
both RealSense cables before the next attempt if you see immediate failures
even on the first camera.

## Network requirement

Basler and Lucid are both GigE Vision (Ethernet) cameras — the MacBook's
Ethernet interface needs to be on the same subnet as the cameras (static IP or
DHCP within their range). If a script reports 0 Basler/Lucid devices found,
check the physical network connection and IP configuration before assuming a
software problem.
