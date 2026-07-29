"""
Camera overlap analysis from existing footage — OVGU AMS calibration tool.

Answers a question that has been open for this whole project, using recordings
you already have and no lab access: **do these cameras see the same belt, and
if so, how are their views related?**

It matters because everything downstream depends on it. If cameras overlap, a
bag can be handed between them geometrically. If they do not, a bag is never in
two views at once and hand-off has to rely on belt travel instead.

How it works, with no calibration board and no human input:

1. Detect SIFT features in a frame from each camera.
2. Match them between every pair of cameras.
3. Fit a homography with MAGSAC and count geometric inliers.

The third step is what makes this trustworthy. Raw feature matches between two
photographs of a mostly-grey conveyor are full of nonsense — similar-looking
belt texture and repeated hardware match each other freely. A homography is a
strong geometric constraint: matches that survive it must be consistent with
*one* planar mapping between the views, which random pairings are not.

Why a homography specifically: the conveyor surface is a plane, and any two
views of a plane are related by exactly one. That also means the result is
directly useful — the fitted homography maps one camera's pixels onto the
other's, which is a conveyor map in relative terms even before any metric
calibration exists.

Two honest limitations:

- **The bags sit above the belt plane.** They are also the only real texture on
  it, so most features come from objects that break the planar assumption
  slightly. Expect a good-but-not-perfect fit; the inlier count is still a valid
  overlap signal.
- **The result has no scale.** It relates cameras to each other, not to
  millimetres. One known distance — the belt width — converts it to metric.

Usage:
    python3 analyse_overlap.py --dir ../real_data/raw_recordings --session 20260727_140907
    python3 analyse_overlap.py --images frames/            # a folder of stills
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
from pathlib import Path

import cv2
import numpy as np

MIN_INLIERS = 15          # below this, a "match" is not credible
GOOD_INLIERS = 40


def frames_from_session(folder: Path, session: str, t_frac: float = 0.5) -> dict:
    """One frame per camera, sampled at the same fraction of each clip.

    Sampled by *time*, not frame index: the cameras run at different real frame
    rates (measured 8.5 to 15.2 fps on this rig), so equal indices would be
    different moments.
    """
    out = {}
    for f in sorted(glob.glob(str(folder / f"*{session}*.avi"))):
        if "depth" in os.path.basename(f):
            continue
        cap = cv2.VideoCapture(f)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * t_frac))
        ok, img = cap.read()
        cap.release()
        if ok:
            out[os.path.basename(f).split("_1280")[0]] = img
    return out


def frames_from_folder(folder: Path) -> dict:
    out = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for f in sorted(glob.glob(str(folder / ext))):
            out[Path(f).stem] = cv2.imread(f)
    return out


def features(img, n=4000):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # CLAHE first: the belt is low-contrast grey and several cameras are dim.
    # Without it SIFT finds most of its keypoints on bright hardware at the
    # frame edges rather than on the belt, which is the region of interest.
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    sift = cv2.SIFT_create(nfeatures=n)
    return sift.detectAndCompute(gray, None)


def match_pair(a, b, ratio=0.75):
    kp1, d1 = a
    kp2, d2 = b
    if d1 is None or d2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(d1, d2, k=2)
    # Lowe's ratio test — discards matches whose best and second-best are
    # similar, which is exactly the repeated-texture case this scene is full of.
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < 4:
        return {"raw": len(good), "inliers": 0, "H": None, "pts": None}

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 4.0,
                                 maxIters=20000, confidence=0.999)
    inl = int(mask.sum()) if mask is not None else 0
    return {"raw": len(good), "inliers": inl, "H": H,
            "pts": (src[mask.ravel() > 0] if mask is not None else None,
                    dst[mask.ravel() > 0] if mask is not None else None)}


def verdict(inl: int) -> str:
    if inl >= GOOD_INLIERS:
        return "OVERLAP"
    if inl >= MIN_INLIERS:
        return "weak/uncertain"
    return "no overlap"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="../real_data/raw_recordings")
    ap.add_argument("--session", default=None,
                    help="timestamp substring, e.g. 20260727_140907")
    ap.add_argument("--images", default=None, help="folder of stills instead")
    ap.add_argument("--t", type=float, default=0.5, help="sample point in clip (0-1)")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    if args.images:
        frames = frames_from_folder(Path(args.images))
    else:
        if not args.session:
            print("ERROR: pass --session or --images")
            return 1
        frames = frames_from_session(Path(args.dir), args.session, args.t)

    if len(frames) < 2:
        print(f"ERROR: need >=2 cameras, found {len(frames)}")
        return 1

    print("=" * 78)
    print(f"CAMERA OVERLAP ANALYSIS — {len(frames)} cameras")
    print("=" * 78)
    feats = {}
    for name, img in frames.items():
        kp, des = features(img)
        feats[name] = (kp, des)
        print(f"  {name:16} {img.shape[1]}x{img.shape[0]}   {len(kp):5d} features")

    print()
    print(f"{'pair':<36}{'matches':>9}{'inliers':>9}  verdict")
    print("-" * 78)
    results = {}
    for a, b in itertools.combinations(sorted(frames), 2):
        r = match_pair(feats[a], feats[b])
        if r is None:
            continue
        results[(a, b)] = r
        print(f"{a + ' <-> ' + b:<36}{r['raw']:>9}{r['inliers']:>9}  {verdict(r['inliers'])}")

        if args.save_dir and r["inliers"] >= MIN_INLIERS:
            d = Path(args.save_dir)
            d.mkdir(parents=True, exist_ok=True)
            # Warp a onto b using the fitted homography and blend: if the two
            # views really do overlap, belt features line up in the blend.
            hb, wb = frames[b].shape[:2]
            warp = cv2.warpPerspective(frames[a], r["H"], (wb, hb))
            blend = cv2.addWeighted(frames[b], 0.5, warp, 0.5, 0)
            cv2.imwrite(str(d / f"blend_{a}__{b}.png"), blend)

    print()
    print("=" * 78)
    over = [p for p, r in results.items() if r["inliers"] >= GOOD_INLIERS]
    weak = [p for p, r in results.items()
            if MIN_INLIERS <= r["inliers"] < GOOD_INLIERS]
    if over:
        print("OVERLAPPING PAIRS (geometrically verified):")
        for a, b in over:
            print(f"   {a} <-> {b}   {results[(a,b)]['inliers']} inliers")
        print()
        print("  These can be tied into one frame directly from footage — the")
        print("  fitted homography already maps one view onto the other. Scale")
        print("  still needs one known distance, e.g. the belt width.")
    else:
        print("NO PAIR SHOWS CONVINCING OVERLAP.")
        print("  Either the cameras genuinely cover different stretches of belt,")
        print("  or the scene lacked shared texture in this frame. Try --t at a")
        print("  different point, or a session with more bags in view, before")
        print("  concluding they are disjoint.")
    if weak:
        print()
        print("  Uncertain pairs (some geometry, not enough to trust):")
        for a, b in weak:
            print(f"   {a} <-> {b}   {results[(a,b)]['inliers']} inliers")
    print("=" * 78)
    if args.save_dir:
        print(f"blends written to {args.save_dir} — check the belt lines up")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
