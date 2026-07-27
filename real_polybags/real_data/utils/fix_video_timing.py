"""
Rewrite recorded .avi files so their playback speed matches real time.

The recorder creates every VideoWriter with the *target* fps, because the real
capture rate isn't known until the run ends. Cameras that can't sustain the
target therefore produce files that play back too fast:

    Lucid     85 frames of a 10s event, 15fps header -> plays in  5.7s (1.76x)
    Basler_1 129 frames of a 10s event, 15fps header -> plays in  8.6s (1.16x)
    Basler_2 150 frames of a 10s event, 15fps header -> plays in 10.0s (correct)

Frame *content* is fine — this only affects the timebase. It matters for
anything temporal (velocity, cross-camera alignment, MOT), and for anyone
eyeballing the footage and concluding one camera "records faster".

This remuxes with the correct frame rate: a container-level rewrite, so it is
fast and completely lossless — no re-encoding, no quality change.

Two ways to determine the true rate, in order of preference:

  1. `--metadata recording_metadata_<timestamp>.json` — written by current
     versions of the recorder, contains the measured actual_fps per camera.
  2. `--duration <seconds>` — for older recordings with no metadata (anything
     from before this was added). True fps = frame_count / duration, using the
     run's requested duration.

Usage:
    # New recordings (preferred)
    python3 fix_video_timing.py --metadata recording_metadata_20260727_142904.json

    # Older recordings — supply the run's --duration
    python3 fix_video_timing.py --duration 10 *.avi

    python3 fix_video_timing.py --duration 10 --dry-run *.avi   # preview only

Originals are preserved as <name>.orig.avi unless --in-place is given.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2


def probe(path):
    """Return (frame_count, header_fps) for a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, fps


def remux(path, true_fps, in_place, dry_run):
    n, hdr = probe(path)
    if not n or n <= 1:
        print(f"  SKIP {os.path.basename(path)}: unreadable or empty")
        return False
    if not true_fps or true_fps <= 0:
        print(f"  SKIP {os.path.basename(path)}: no usable true fps")
        return False

    ratio = hdr / true_fps if true_fps else 1.0
    if 0.98 <= ratio <= 1.02:
        print(f"  OK   {os.path.basename(path)}: {hdr:.2f}fps header vs "
              f"{true_fps:.2f} actual — already correct")
        return False

    print(f"  FIX  {os.path.basename(path)}: {n} frames, header {hdr:.2f}fps -> "
          f"{true_fps:.2f}fps (was playing {ratio:.2f}x too fast)")
    if dry_run:
        return True

    tmp = path + ".remux.avi"
    # -c copy => stream copy, no re-encode. -r before -i reinterprets the input
    # timebase rather than dropping/duplicating frames, so every original frame
    # is preserved exactly; only the declared rate changes.
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-r", f"{true_fps:.6f}", "-i", path, "-c", "copy", tmp]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"       ffmpeg failed: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False

    n2, fps2 = probe(tmp)
    if not n2 or n2 < n * 0.99:
        print(f"       ABORT: output has {n2} frames vs {n} input — keeping original")
        os.remove(tmp)
        return False

    if not in_place:
        shutil.move(path, path.replace(".avi", ".orig.avi"))
    else:
        os.remove(path)
    shutil.move(tmp, path)
    print(f"       -> {n2} frames @ {fps2:.2f}fps")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*', help=".avi files to fix")
    ap.add_argument('--metadata', help="recording_metadata_*.json from the recorder")
    ap.add_argument('--duration', type=float,
                    help="Run duration in seconds, for files without metadata")
    ap.add_argument('--in-place', action='store_true',
                    help="Don't keep .orig.avi backups")
    ap.add_argument('--dry-run', action='store_true', help="Show what would change")
    args = ap.parse_args()

    targets = {}   # path -> true fps

    if args.metadata:
        with open(args.metadata) as f:
            meta = json.load(f)
        base = os.path.dirname(os.path.abspath(args.metadata))
        for cam, info in meta.get("cameras", {}).items():
            fps = info.get("actual_fps")
            if not fps:
                continue
            for key in ("color_file", "depth_file"):
                fn = info.get(key)
                if fn:
                    p = fn if os.path.isabs(fn) else os.path.join(base, fn)
                    if os.path.exists(p):
                        targets[p] = fps
    elif args.duration:
        if not args.files:
            print("ERROR: pass .avi files along with --duration")
            return 1
        for p in args.files:
            n, _ = probe(p)
            if n and n > 1:
                targets[p] = n / args.duration
    else:
        print("ERROR: need --metadata or --duration (see --help)")
        return 1

    if not targets:
        print("Nothing to do — no matching video files found.")
        return 0

    print(f"Checking {len(targets)} file(s)...")
    fixed = sum(remux(p, f, args.in_place, args.dry_run)
                for p, f in sorted(targets.items()))
    print(f"\n{fixed} file(s) "
          f"{'would be' if args.dry_run else ''} corrected.")
    if args.dry_run:
        print("Re-run without --dry-run to apply.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
