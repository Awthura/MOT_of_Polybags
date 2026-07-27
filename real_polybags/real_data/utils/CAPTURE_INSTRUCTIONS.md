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
| RGBD ×2 | `pyrealsense2` (conda-forge) | Needs `sudo` (USB power state). The old sudo-vs-Lucid conflict is resolved — see issue 1. Multi-camera hub issue still open, see issue 2 |

## Every session (macOS)

**Nothing to export any more.** This used to require
`export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH` in every new
shell, which was also what made `sudo` and Lucid mutually exclusive (see
"Resolved" below). That is fixed permanently by symlinks inside the conda
env — do **not** set `DYLD_LIBRARY_PATH`; it is no longer needed, and
`sudo -E`'s `-E` is now redundant too.

If the conda env is ever rebuilt, re-create the symlinks:

```bash
cd /opt/anaconda3/envs/ams/lib
for L in libglib-2.0.0 libgobject-2.0.0 libgio-2.0.0 libgmodule-2.0.0 \
         libgirepository-2.0.0 libaravis-0.8.0; do
  ln -sf "/opt/homebrew/lib/$L.dylib" "$L.dylib"
done
```

## Running

```bash
conda activate ams
mkdir -p /Users/awthura/OVGU/AMS/real_polybags/real_data/raw_recordings
cd /Users/awthura/OVGU/AMS/real_polybags/real_data/raw_recordings

# Basler + Lucid only (no sudo needed — both are GigE, not USB):
python ../utils/record_basler_lucid_10_macos.py

# All 5 cameras, target FPS + verbose per-camera FPS reporting:
sudo /opt/anaconda3/envs/ams/bin/python ../utils/record_all_5_cameras_macos.py --fps 15 --duration 90 --verbose
```

Do a short `--duration 10` smoke run first to confirm all 5 cameras come up
before committing to a real take.

Note: under `sudo` the output `.avi` files are owned by `root`. Fix after a
session with:

```bash
sudo chown "$USER" *.avi
```

Always run from `raw_recordings/` (or any dedicated folder) so output `.avi`
files land there, not in `utils/`.

`record_all_5_cameras_macos.py`'s `--fps`/`--duration`/`--width`/`--height`
replace what used to be hardcoded in the older scripts' `__main__` blocks —
no file editing needed for those on the combined script.

## Known issues

### 1. ~~RealSense needs `sudo` — but `sudo` breaks Aravis/Lucid~~ — RESOLVED (2026-07-27)

`pyrealsense2` does need root to claim the USB device and set its power state
("failed to set power state" without `sudo` — the Basler cameras are
unaffected because they're GigE/Ethernet, not USB, which is why a non-sudo
run gets *past* Basler init and dies on RealSense). The old blocker was that
Aravis/Lucid needed `DYLD_LIBRARY_PATH`, and dyld strips `DYLD_*` for
privilege-elevated processes.

**The earlier `/usr/local/lib` "fix to try" does not work, and the reason it
can't is worth recording.** Those symlinks were created but are never
consulted. Tracing the actual failure with `DYLD_PRINT_LIBRARIES=1` shows
the real mechanism is not an absolute-path version clash but a **bare-name
`dlopen` from the GI typelib**:

```
GLib-GIRepository-WARNING: Failed to load shared library 'libglib-2.0.0.dylib'
  referenced by the typelib: dlopen(libglib-2.0.0.dylib, 0x0009): tried:
  'libglib-2.0.0.dylib' (no such file),
  '/opt/anaconda3/envs/ams/bin/../lib/libglib-2.0.0.dylib' (no such file),
  '/usr/lib/libglib-2.0.0.dylib' (no such file, not in dyld cache)
```

`/usr/local/lib` is simply not in that effective search list — so symlinking
there accomplishes nothing. But note the second path dyld tries:
**`<env>/bin/../lib/`, i.e. the conda env's own `lib/`**. That directory *is*
searched natively, with no env var and regardless of privilege level.

**Actual fix**: symlink the libraries into `/opt/anaconda3/envs/ams/lib/`
(command in "Every session" above). The env had no `libg*`/`libaravis` files
of its own, so nothing is shadowed. Verified: Aravis enumerates all 3 GigE
devices (both Baslers + `Lucid Vision Labs-TRI032S-C-232700105`) with **no
`DYLD_*` variable set at all**, which is precisely why it now survives
`sudo` — there is no longer any variable for dyld to strip.

Supporting detail that makes this safe: `/opt/anaconda3/envs/ams/bin/python3.11`
is adhoc-signed (`flags=0x2`) with no hardened-runtime flag, no
library-validation entitlement, and is not setuid — so dyld honours its own
default search paths for it under root exactly as it does for a normal user.

Consequence: the old "run Basler+Lucid and RealSense as two separate
sessions, losing cross-group start-time sync" workaround is no longer
needed — a single combined 5-camera `sudo` run is the supported path.

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
