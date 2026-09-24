# Multi-camera capture — setup & run instructions

## Scripts

- `record_basler_lucid_rgbd.py` — **Windows/Linux script** (uses Arena SDK for
  Lucid, which has no macOS build). Updated 2026-07-27 to carry the same fixes
  as the macOS combined script: setup-then-go-signal sync instead of a
  fixed-size `threading.Barrier` (one camera failing to connect no longer
  raises BrokenBarrierError and loses the whole recording), frozen fps clock,
  and a `--fps`/`--duration`/`--width`/`--height`/`--max-rgbd` CLI in place of
  hardcoded values. See "Windows" below.
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

## Windows

```bash
pip install pypylon pyrealsense2
# Lucid: install the Arena SDK + arena_api Python wheel from Lucid's website
python record_basler_lucid_rgbd.py --fps 15 --duration 90
```

**Confirmed working on Windows against all 4 cameras (2026-07-28.)**

Both scripts now default to the **4-camera rig** — 2 Basler + 1 Lucid + 1 RGBD
— after one RealSense was removed from the mount. Raise `--max-rgbd` (Windows)
or `--max-realsense` (macOS) to 2 only if a second unit is refitted, and read
issue 2 first.

Two behaviours worth knowing:

- **A generic USB/laptop webcam will NOT be used as an RGBD camera** unless you
  pass `--allow-webcam-fallback`. The fallback exists for RGBD-ish devices that
  only appear as generic UVC (e.g. Orbbec), but on a fixed rig it silently
  masked a real failure: if the D435 failed to enumerate, a webcam quietly took
  its slot and the run looked successful. Off by default, so that slot is now
  reported as failed instead.
- **The preview grid is sized to the camera count** (near-square) rather than a
  fixed 3 columns, so 4 cameras tile 2x2 with no dead panels.

Feature parity with the macOS script: resilient sync, frozen fps clock,
per-frame host + device timestamps, `recording_metadata_*.json`,
playback-speed warnings, and `--lucid-packet-size`/`--lucid-packet-delay`/
`--depth-width`/`--depth-height` tuning.

No `sudo`/admin needed — the macOS RealSense USB permission problem
(see Known Issues 1) is macOS-specific.

`--max-rgbd 1` excludes one RGBD unit; use it when a camera is being removed
from the rig, or when one is in a bad USB state and destabilising the others.

The two RealSense units in this rig interfere with each other at device-open
time when they share a USB controller — see Known Issues 2. If that shows up on
Windows too, `--max-rgbd 1` is the workaround, and putting the cameras on
separate controllers is the fix.

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

Known upstream librealsense/macOS bug ("failed to set power state", "No device
connected", or "UVC device is already opened!"). Mitigated in
`record_real_sense_dual.py` and `record_all_5_cameras_macos.py`'s `RGBDWorker`
with a retry loop (4 attempts, fresh `pipeline`/`config` each time), but
**never fully solved — treat 2x RealSense as unreliable on this rig.**

**Confirmed 2026-07-27: both cameras are on the SAME USB controller**
(`system_profiler SPUSBDataType` Location IDs `0x01210000` and `0x01220000`).

**It is contention at device-open time, not bandwidth.** Measured across two
runs 25 minutes apart:

| depth stream | result |
|---|---|
| 1280x720 | RGBD_2 recorded 152 frames, RGBD_1 failed |
| 848x480 (~31 MB/s less) | **both** failed |

Reducing bandwidth made it worse, so the bandwidth theory is wrong. What the
logs show instead: one camera's `pipeline.start()` retry loop runs while the
other is in its 30-frame warmup, and the warming camera dies with "Frame didn't
arrive within 5000". Corroborating evidence: `reset_realsense.py --skip-reset`
passes **both** cameras every time, because it opens them *sequentially* — they
only fail when opened concurrently.

Workarounds, in order of preference:
1. `--max-realsense 1` (macOS) / `--max-rgbd 1` (Windows) — guarantees a clean
   4-camera take. Losing one RGBD stream beats losing the session.
2. Put the two cameras on **separate USB controllers**. This is the real fix.
3. Run RealSense in a separate process from the GigE cameras, so neither
   contention nor a libusb segfault can affect Basler/Lucid. Not implemented.

### 2b. RealSense wedges after any unclean exit

`kill -9`, `Ctrl+Z`, or a libusb SIGSEGV leaves the D435s wedged: subsequent
runs report "failed to set power state" or "No device connected" **even as
root**. Use the recovery tool instead of replugging cables:

```bash
sudo /opt/anaconda3/envs/ams/bin/python reset_realsense.py
sudo /opt/anaconda3/envs/ams/bin/python reset_realsense.py --skip-reset  # check only
```

It sends `hardware_reset()` (firmware reboot), waits for re-enumeration, then
starts a stream on each camera to prove it actually works — enumeration alone
is not proof, since the failure occurs at `pipeline.start()`.

Two traps that script had to learn, both of which produced false "camera is
broken" verdicts on healthy hardware:
- **Re-enumerating is not being ready.** Streaming ~2s after the devices
  re-appear fails; the firmware needs several more seconds (`--settle`).
- **Use one shared `rs.context()`**, passed to `rs.pipeline(ctx)`. With a fresh
  context per operation, devices enumerate fine in one context while a pipeline
  on another reports "No device connected" for the same serial.

**Prevention**: `Ctrl+C` once and let the `finally` blocks release the
pipelines. Never `Ctrl+Z` — suspending holds the cameras claimed, which is what
produces "No device connected" on the next run.

### 3. Recorded .avi playback speed is wrong for any camera below target fps

Both recorders create their VideoWriter with the **target** fps, because the
real rate isn't known until the run ends. A camera that can't sustain the
target therefore writes a file that plays back too fast. Measured 2026-07-27
on a 10s run at a 15 fps target:

| camera | frames | plays for | appears |
|---|---|---|---|
| Lucid | 85 | 5.7s | **1.76x too fast** |
| Basler_1 | 129 | 8.6s | 1.16x too fast |
| Basler_2 | 150 | 10.0s | correct |
| RGBD | 152 | 10.1s | correct |

This is why Lucid looks like it "records faster" than the RGBD cameras when it
is in fact the slowest on the rig. Frame *content* is unaffected — but frame
index is not proportional to time, which breaks velocity, cross-camera
alignment and MOT.

Both scripts now write `timestamps_<camera>_<run>.csv` (real per-frame capture
times) and `recording_metadata_<run>.json` (measured fps, playback speed error,
pacing drift). **Use those for any temporal analysis, not the video timebase.**
The end-of-run summary flags any camera off by more than 5%.

The CSV carries two clocks per frame, because neither alone is enough:

| column | meaning |
|---|---|
| `host_unix_time` | host clock, sampled after transfer + conversion. Shared across cameras (one host records all), but carries per-camera latency. |
| `device_timestamp` | the camera's own timestamp. Precise, but on that camera's private clock and in SDK-specific units. |
| `device_delta` | device timestamp relative to that camera's first frame — readable without knowing the tick rate. |

`timestamp_domain` in the metadata JSON says how to interpret
`device_timestamp`: `aravis_device_ns` (nanoseconds), `basler_device_ticks`
(model-dependent rate, and it changes if PTP is enabled), `arena_device_ns`, or
a librealsense domain such as `global_time`. Units are recorded verbatim rather
than normalised, because converting blindly would silently produce wrong
numbers.

Pairing the two clocks is what makes ~10–30 ms cross-camera alignment possible
without PTP hardware: the device clock supplies precision, the host clock a
common origin.

### 3b. Measuring how well-synchronized the cameras actually are

```bash
cd raw_recordings
python3 ../utils/measure_sync.py                 # newest run
python3 ../utils/measure_sync.py --json skew.json
```

Reports per-camera pacing, cross-camera skew (p95 and worst-case), and
device-vs-host clock drift in ppm.

**Run this before buying any synchronization hardware.** The key column is
`floor` = 1/(2·fps): the closest a camera can be to an arbitrary instant purely
because of its frame rate. If measured skew is already at that floor, the
cameras are as aligned as their frame rates permit and a PTP switch would change
nothing — the fix is a higher frame rate (issue 4 below). The tool states this
verdict explicitly rather than leaving it to be inferred.

A drift figure above ~100 ppm is worth attention: over a 90 s run that is ~9 ms
of accumulated error, the same order as the skew being measured. Drift is
correctable in software; it does not require hardware either.

To correct existing files (lossless container remux, keeps `.orig.avi`):

```bash
python3 fix_video_timing.py --metadata recording_metadata_<run>.json
python3 fix_video_timing.py --duration 10 --dry-run *.avi   # older recordings
```

### 4. Lucid frame rate is capped by GigE inter-packet delay

Lucid captured exactly 85 frames in 10s (8.5 fps) on four consecutive runs — a
hard configuration ceiling, not jitter. The arithmetic:

```
1280x720 BGR8 = 2,764,800 B / 1400 B per packet  = ~1975 packets/frame
1975 packets x 60us inter-packet delay           = 118.5 ms/frame
                                                 => 8.4 fps ceiling
```

That matches the measured 8.5 fps to within 1%, and it applies *before* any
transmission or processing time. The Windows script sets `GevSCPD = 60000`,
and these values are stored **non-volatile on the camera**, so they persist
across machines and sessions — which is likely why macOS saw 8.5 fps despite
requesting 40us (its `gv_set_packet_delay` sat behind a bare `except: pass`,
so a silent rejection was invisible).

Note the asymmetry: Basler is given 8us in the same script, Lucid 60us.

Both scripts now read back the negotiated values at startup and print the
delay-implied fps ceiling, warning when it caps below target. Tune with
`--lucid-packet-delay` / `--lucid-packet-size` while watching the `incomplete`
count — that counts dropped/incomplete frames, which is what the delay exists
to prevent. **Not yet verified against hardware.**

## Network requirement

Basler and Lucid are both GigE Vision (Ethernet) cameras — the MacBook's
Ethernet interface needs to be on the same subnet as the cameras (static IP or
DHCP within their range). If a script reports 0 Basler/Lucid devices found,
check the physical network connection and IP configuration before assuming a
software problem.
