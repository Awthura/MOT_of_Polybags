"""
Power-cycle the RealSense cameras in firmware, without physically unplugging.

Why this exists: the D435s in this rig get wedged at the USB level fairly
easily — after a `kill -9` during capture, after a Ctrl+Z suspend, or after a
libusb SIGSEGV. Once wedged, `pipeline.start()` reports either
"failed to set power state" or "No device connected" *even as root*, and the
documented remedy so far was to physically unplug and replug both cables.
That's slow and awkward mid-recording-session.

`rs.device.hardware_reset()` asks the camera's firmware to reboot itself,
which clears the same bad state. The device then drops off the bus and
re-enumerates a few seconds later, so this script waits for it to come back,
gives the firmware time to settle, and then proves it can actually stream.

Two things this script learned the hard way, both of which produced false
"camera is broken" verdicts on hardware that was fine:

1. **Re-enumerating is not the same as being ready.** Streaming ~2s after the
   devices re-appear fails with "No device connected". The firmware needs
   several more seconds (`--settle`).
2. **Use ONE rs.context() for everything.** Creating a fresh context per
   operation — and letting each `rs.pipeline()` make its own private one — is
   unreliable on macOS: devices enumerate fine in one context while a
   pipeline built on a different context reports "No device connected" for the
   very same serial. The context is created once here and passed explicitly to
   `rs.pipeline(ctx)`.

Run this BEFORE a recording session if the previous run crashed or was killed:

    sudo /opt/anaconda3/envs/ams/bin/python ../utils/reset_realsense.py

Needs sudo for the same reason the recorder does — librealsense must claim
the USB device to talk to it at all.

If a camera still doesn't come back after this, then physically replug it —
that's the only remaining step.
"""

import argparse
import time

import pyrealsense2 as rs


def serials(ctx):
    """Serial numbers visible in `ctx`, or [] if enumeration itself fails."""
    try:
        return sorted(d.get_info(rs.camera_info.serial_number)
                      for d in ctx.query_devices())
    except RuntimeError as e:
        print(f"  (enumeration failed: {e})")
        return []


def try_stream(ctx, sn, attempts, gap):
    """Return True if `sn` can start+stop a stream, retrying on transient failure."""
    for attempt in range(1, attempts + 1):
        pipe = rs.pipeline(ctx)          # share the context — see module docstring
        cfg = rs.config()
        cfg.enable_device(sn)
        try:
            pipe.start(cfg)
            try:
                pipe.stop()
            except RuntimeError:
                pass
            print(f"  [{sn}] stream start OK"
                  + (f" (attempt {attempt})" if attempt > 1 else ""))
            return True
        except RuntimeError as e:
            if attempt == attempts:
                print(f"  [{sn}] stream start FAILED after {attempts} attempts: {e}")
                return False
            print(f"  [{sn}] not ready yet ({e}) — retrying {attempt}/{attempts}")
            time.sleep(gap)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wait', type=float, default=25.0,
                    help="Seconds to wait for cameras to re-enumerate (default 25)")
    ap.add_argument('--settle', type=float, default=8.0,
                    help="Extra seconds to wait AFTER cameras re-appear before "
                         "trying to stream (default 8). Re-enumerating is not the "
                         "same as being ready.")
    ap.add_argument('--verify-attempts', type=int, default=4,
                    help="How many times to try starting each stream (default 4)")
    ap.add_argument('--gap', type=float, default=4.0,
                    help="Seconds between stream-start attempts (default 4)")
    ap.add_argument('--skip-reset', action='store_true',
                    help="Don't reset; just verify the cameras can stream. Useful to "
                         "check state without disturbing working cameras.")
    args = ap.parse_args()

    # One context for the whole run. It also picks up device add/remove events,
    # so it stays valid across the reset.
    ctx = rs.context()

    before = serials(ctx)
    print(f"Found {len(before)} RealSense camera(s): {before or '(none)'}")
    if not before:
        print("\nNothing to work with. If cameras are physically connected but not\n"
              "listed, they are wedged badly enough that only a physical replug\n"
              "will clear it.")
        return 1

    if args.skip_reset:
        print("\nSkipping reset (--skip-reset); verifying streams only...")
        ok = all(try_stream(ctx, sn, args.verify_attempts, args.gap) for sn in before)
        print("\nAll cameras streaming — safe to record." if ok
              else "\nAt least one camera won't stream — run without --skip-reset.")
        return 0 if ok else 1

    n_sent = 0
    for dev in ctx.query_devices():
        try:
            sn = dev.get_info(rs.camera_info.serial_number)
        except RuntimeError:
            sn = "<unreadable>"
        try:
            dev.hardware_reset()
            print(f"  [{sn}] hardware_reset() sent")
            n_sent += 1
        except RuntimeError as e:
            # Surfaced rather than swallowed: a device too wedged to accept a
            # reset command is exactly the case that still needs a physical replug.
            print(f"  [{sn}] hardware_reset() FAILED: {e}")

    if not n_sent:
        print("\nNo resets went through — physically replug the cameras.")
        return 1

    print(f"\nWaiting up to {args.wait:.0f}s for re-enumeration...")
    deadline = time.time() + args.wait
    after = []
    while time.time() < deadline:
        time.sleep(1.0)
        after = serials(ctx)
        if len(after) >= len(before):
            break
        print(f"  ...{len(after)}/{len(before)} back")

    print(f"After reset: {len(after)}/{len(before)} present: {after or '(none)'}")
    missing = sorted(set(before) - set(after))
    if missing:
        print(f"STILL MISSING: {missing}")
        print("Physically unplug and replug these, then re-run this script.")
        return 1

    print(f"Letting firmware settle for {args.settle:.0f}s...")
    time.sleep(args.settle)

    print("\nVerifying each camera can actually start a stream...")
    ok = all(try_stream(ctx, sn, args.verify_attempts, args.gap) for sn in after)

    print("\nAll cameras reset and streaming — safe to record." if ok
          else "\nAt least one camera still won't stream — physically replug it.")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
