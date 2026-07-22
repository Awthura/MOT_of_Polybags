# Multi-camera capture — setup & run instructions

Two scripts here:
- `record_basler_lucid_rgbd.py` — supervisor's original. Uses `pypylon` (Basler),
  `arena_api` (Lucid, Arena SDK), `pyrealsense2`/`pyk4a` (RGBD). Windows/Linux only
  for the Lucid path — Arena SDK has no macOS build.
- `record_basler_lucid_rgbd_macos.py` — macOS variant. Basler and RGBD logic are
  identical to the original (both have real macOS support once their SDKs are
  installed). Only the Lucid path differs: it uses **Aravis** (vendor-neutral
  GigE Vision library) instead of Arena SDK, since Lucid cameras are standard
  GigE Vision devices and Aravis can drive them directly on macOS.

**Use the `_macos` version on the MacBook. Use the original on Windows/Linux.**

## One-time environment setup (macOS, `ams` conda env)

```bash
brew install aravis pygobject3

conda activate ams
pip install pypylon
pip install pygobject
conda install -c conda-forge pyrealsense2 -y
```

All three were verified to import and enumerate devices (0 found, since not
connected to hardware during setup) — see below for what's confirmed vs. not:

| Camera | SDK used on macOS | Status |
|---|---|---|
| Basler ×2 | `pypylon` | Verified import; native Apple Silicon wheel |
| RGBD ×2 | `pyrealsense2` (conda-forge) | Verified import; native Apple Silicon build |
| Lucid | Aravis (not Arena SDK) | Import verified; **capture loop not yet tested against real hardware** |

If the Lucid path errors once connected to the real camera, it's most likely a
pixel-format string (tries `BGR8` then `RGB8`) or the exposure/gain auto-mode
call (`Aravis.Auto.OFF`) needing a small adjustment — not a redesign.

## Every session (macOS)

Aravis needs Homebrew's GLib/GObject libraries visible from inside the conda
env — this must be set every new shell:

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
```

Consider adding that line to `~/.zshrc` if recording sessions are frequent.

## Running

```bash
conda activate ams
mkdir -p /Users/awthura/OVGU/AMS/real_polybags/real_data/raw_recordings
cd /Users/awthura/OVGU/AMS/real_polybags/real_data/raw_recordings
python ../utils/record_basler_lucid_rgbd_macos.py
```

Run from `raw_recordings/` (or any dedicated folder) so the output `.avi`
files land there rather than in `utils/`.

Duration/resolution/fps are hardcoded in the `__main__` block at the bottom of
whichever script you run (`duration_seconds=50, width=1280, height=720, fps=5`
by default) — edit those values directly before running if you need a
different session length.

## Network requirement

Basler and Lucid are both GigE Vision (Ethernet) cameras — the MacBook's
Ethernet interface needs to be on the same subnet as the cameras (static IP or
DHCP within their range). If the script reports 0 Basler/Lucid devices found,
check the physical network connection and IP configuration before assuming a
software problem.
