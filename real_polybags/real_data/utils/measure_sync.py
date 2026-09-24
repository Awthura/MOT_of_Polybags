"""
Measure how well the cameras are actually synchronized.

This turns "are the cameras in sync?" into a number, which is the question that
decides whether any hardware (a PTP-capable switch, trigger wiring) is worth
buying. Run this before spending money.

What it does, given one recording run:

  1. Reports each camera's pacing — actual fps, and how far frame intervals
     stray from perfectly even. A camera that is merely *slow* but *steady* is
     much easier to align than one that is erratic.
  2. Cross-camera skew: for every frame of a reference camera, finds the
     nearest frame in each other camera and reports the distribution of time
     differences. This is the number that matters — it is the residual error
     you would carry into MOT if you align purely in software.
  3. Device-vs-host clock drift: if the offset between a camera's own clock and
     the host clock is constant, the device timestamps are usable directly. If
     it drifts, the clocks are running at different rates and alignment must
     account for it.

Interpreting the result, against the ~10-30 ms target chosen for this project:

  < 30 ms   software alignment is sufficient; no hardware needed.
  30-50 ms  usable, but check whether the worst-case pairs are the ones that
            matter for cross-camera bag hand-off.
  > 50 ms   investigate before building on it. Likely a slow camera (Lucid at
            8.5 fps has 118 ms between frames, so it cannot be closer than
            ~59 ms to an arbitrary instant no matter how good the clocks are).

That last point is worth stating plainly: **the frame rate itself sets a floor
on achievable skew.** A camera running at F fps can be at best 1/(2F) away from
any given moment. Fixing skew below that floor requires raising the frame rate
(see the Lucid packet-delay issue in CAPTURE_INSTRUCTIONS.md, issue 4), not
better clocks.

Usage:
    python3 measure_sync.py                              # newest run in cwd
    python3 measure_sync.py --metadata recording_metadata_20260727_142904.json
    python3 measure_sync.py --reference Basler_2         # pick the reference
    python3 measure_sync.py --json skew.json             # machine-readable out
"""

import argparse
import csv
import glob
import json
import os
import statistics


def load_run(meta_path):
    """Return (metadata dict, {camera: [host_times]}, {camera: [device_times]})."""
    with open(meta_path) as f:
        meta = json.load(f)
    base = os.path.dirname(os.path.abspath(meta_path))
    host, dev = {}, {}
    for cam, info in meta.get("cameras", {}).items():
        ts_file = info.get("timestamps_file")
        if not ts_file:
            continue
        path = ts_file if os.path.isabs(ts_file) else os.path.join(base, ts_file)
        if not os.path.exists(path):
            continue
        h, d = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                # Tolerate the pre-device-timestamp CSV layout, which used
                # "unix_time" rather than "host_unix_time".
                t = row.get("host_unix_time") or row.get("unix_time")
                if t:
                    h.append(float(t))
                raw = row.get("device_timestamp", "")
                d.append(float(raw) if raw not in ("", None) else None)
        if h:
            host[cam] = h
            dev[cam] = d
    return meta, host, dev


def nearest_skews(ref_times, other_times):
    """For each ref frame, the signed time to the nearest frame in `other`."""
    if not ref_times or not other_times:
        return []
    out, j = [], 0
    n = len(other_times)
    for t in ref_times:
        # other_times is monotonically increasing, so advance a single pointer
        # rather than searching from scratch for each reference frame.
        while j + 1 < n and abs(other_times[j + 1] - t) <= abs(other_times[j] - t):
            j += 1
        out.append(other_times[j] - t)
    return out


def describe(values, scale=1000.0):
    """Summarise a list (converted to ms by default)."""
    if not values:
        return None
    v = sorted(x * scale for x in values)
    n = len(v)
    return {
        "n": n,
        "mean": statistics.fmean(v),
        "median": v[n // 2],
        "p95_abs": sorted(abs(x) for x in v)[int(n * 0.95) - 1 if n > 1 else 0],
        "max_abs": max(abs(x) for x in v),
        "min": v[0],
        "max": v[-1],
    }


def fmt(d, unit="ms"):
    if not d:
        return "n/a"
    return (f"mean {d['mean']:+7.1f} | median {d['median']:+7.1f} | "
            f"p95 |{d['p95_abs']:6.1f}| | max |{d['max_abs']:6.1f}| {unit}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--metadata', help="recording_metadata_*.json (default: newest in cwd)")
    ap.add_argument('--reference', help="camera to measure others against "
                                        "(default: the one with most frames)")
    ap.add_argument('--json', dest='json_out', help="write results as JSON")
    args = ap.parse_args()

    meta_path = args.metadata
    if not meta_path:
        cands = sorted(glob.glob("recording_metadata_*.json"))
        if not cands:
            print("ERROR: no recording_metadata_*.json here. Pass --metadata, or "
                  "run from the directory holding the recordings.")
            return 1
        meta_path = cands[-1]
        print(f"Using newest run: {meta_path}\n")

    meta, host, dev = load_run(meta_path)
    if not host:
        print("ERROR: no timestamp CSVs found for this run.")
        print("Runs recorded before per-frame timestamps were added have none —")
        print("re-record with the current script to measure sync.")
        return 1

    target = meta.get("target_fps")
    results = {"metadata": meta_path, "target_fps": target, "cameras": {}, "skew": {}}

    # ── 1. Per-camera pacing ─────────────────────────────────────────────────
    print("=" * 78)
    print("PER-CAMERA PACING")
    print("=" * 78)
    print(f"{'camera':<12}{'frames':>7}{'fps':>7}{'interval ms':>13}{'jitter (sd)':>13}"
          f"{'floor ms':>10}")
    for cam, h in sorted(host.items()):
        if len(h) < 2:
            print(f"{cam:<12}{len(h):>7}  (too few frames)")
            continue
        gaps = [(h[i + 1] - h[i]) * 1000 for i in range(len(h) - 1)]
        fps = (len(h) - 1) / (h[-1] - h[0]) if h[-1] > h[0] else 0
        sd = statistics.pstdev(gaps) if len(gaps) > 1 else 0.0
        # Best achievable distance from an arbitrary instant, purely from rate.
        floor = (1000.0 / fps) / 2 if fps else float('inf')
        print(f"{cam:<12}{len(h):>7}{fps:>7.1f}{statistics.fmean(gaps):>13.1f}"
              f"{sd:>13.1f}{floor:>10.1f}")
        results["cameras"][cam] = {
            "frames": len(h), "fps": round(fps, 3),
            "mean_interval_ms": round(statistics.fmean(gaps), 3),
            "interval_sd_ms": round(sd, 3),
            "skew_floor_ms": round(floor, 3),
        }
    print("\n'floor' = 1/(2*fps): the closest this camera can be to an arbitrary")
    print("instant purely because of its frame rate. Skew cannot beat it.")

    # ── 2. Cross-camera skew ─────────────────────────────────────────────────
    ref = args.reference or max(host, key=lambda c: len(host[c]))
    if ref not in host:
        print(f"\nERROR: reference camera '{ref}' not in this run: {list(host)}")
        return 1

    print()
    print("=" * 78)
    print(f"CROSS-CAMERA SKEW (reference: {ref})")
    print("=" * 78)
    worst = 0.0
    worst_cam = None
    for cam in sorted(host):
        if cam == ref:
            continue
        d = describe(nearest_skews(host[ref], host[cam]))
        print(f"{cam:<12} {fmt(d)}")
        if d:
            results["skew"][f"{ref}->{cam}"] = {k: round(v, 3) for k, v in d.items()}
            if d["p95_abs"] > worst:
                worst, worst_cam = d["p95_abs"], cam

    # ── 3. Device vs host clock ──────────────────────────────────────────────
    print()
    print("=" * 78)
    print("DEVICE CLOCK vs HOST CLOCK")
    print("=" * 78)
    any_dev = False
    for cam in sorted(host):
        d = dev.get(cam) or []
        pairs = [(h, x) for h, x in zip(host[cam], d) if x is not None]
        domain = meta.get("cameras", {}).get(cam, {}).get("timestamp_domain")
        if len(pairs) < 2:
            print(f"{cam:<12} no device timestamps  (domain: {domain})")
            continue
        any_dev = True
        # Device units differ per SDK (ns, model-dependent ticks, ms), so an
        # absolute rate is not comparable across cameras. What IS comparable,
        # and what actually matters, is whether the rate is *constant*: compare
        # the device/host rate over the first half of the run against the
        # second. Equal rates mean a fixed offset, which alignment handles
        # trivially. Diverging rates mean the clocks tick at different speeds,
        # and the error grows with recording length.
        mid = len(pairs) // 2
        def rate(seg):
            (h0, d0), (hN, dN) = seg[0], seg[-1]
            return (dN - d0) / (hN - h0) if hN > h0 else float('nan')
        r1, r2 = rate(pairs[:mid + 1]), rate(pairs[mid:])
        drift_ppm = ((r2 - r1) / r1 * 1e6) if r1 else float('nan')
        overall = rate(pairs)
        # Over a 90s run, 100 ppm = 9 ms of accumulated error — same order as
        # the skew being measured, so it is worth flagging.
        flag = ""
        if drift_ppm == drift_ppm and abs(drift_ppm) > 100:
            flag = f"  <- DRIFT {drift_ppm:+.0f} ppm"
        print(f"{cam:<12} domain={domain or '?':<22} "
              f"rate={overall:.6g} u/s  drift={drift_ppm:+7.1f} ppm{flag}")
        results["cameras"].setdefault(cam, {})["device_host_rate"] = (
            None if overall != overall else round(overall, 9))
        results["cameras"][cam]["clock_drift_ppm"] = (
            None if drift_ppm != drift_ppm else round(drift_ppm, 3))
        results["cameras"][cam]["timestamp_domain"] = domain
    if not any_dev:
        print("\nNo device timestamps in this run. Either it predates the feature,")
        print("or every SDK call failed — check the recorder's startup output.")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    if worst:
        print(f"VERDICT: worst-case p95 skew = {worst:.1f} ms  ({ref} vs {worst_cam})")

        # The decisive question is WHY the skew is what it is. If it is already
        # at the frame-rate floor, the clocks are blameless and no amount of
        # PTP or triggering will help — only a higher frame rate will. Getting
        # this backwards is exactly how one ends up buying a switch that
        # changes nothing.
        floor = results["cameras"].get(worst_cam, {}).get("skew_floor_ms")
        # "Rate-limited" means the skew sits AT the floor — a band around it,
        # not merely below it. Skew well below the floor is a good result (the
        # cameras happen to be in phase), not evidence of a rate limit.
        rate_limited = floor is not None and floor * 0.75 <= worst <= floor * 1.15

        if worst < 30:
            print("  Within the ~10-30ms target. Software alignment is sufficient;")
            print("  no synchronization hardware needed.")
            if rate_limited:
                print(f"  (Note: this is already at {worst_cam}'s frame-rate floor of "
                      f"{floor:.1f} ms — it cannot improve without a higher fps.)")
        elif rate_limited:
            print(f"  This is AT the frame-rate floor for {worst_cam} "
                  f"({floor:.1f} ms at its {results['cameras'][worst_cam]['fps']:.1f} fps).")
            print("  => RATE-LIMITED, not clock-limited. The cameras are as aligned")
            print("     as their frame rates permit. Synchronization hardware would")
            print("     NOT improve this; a higher frame rate would.")
            print("     For Lucid see CAPTURE_INSTRUCTIONS.md issue 4 (packet delay).")
        elif worst < 50:
            print("  Usable but marginal, and NOT explained by frame rate alone.")
            print("  Check the drift figures above before considering hardware.")
        else:
            print("  Above target and not explained by frame rate alone — check the")
            print("  drift figures above; a large ppm error is fixable in software.")
    else:
        print("VERDICT: only one camera in this run — nothing to compare.")
    print("=" * 78)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
