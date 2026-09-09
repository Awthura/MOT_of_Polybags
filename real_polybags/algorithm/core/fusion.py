"""
Fusion + global re-ID — OVGU AMS algorithm phase.

Turns per-camera, per-frame detections (already in belt millimetres) into a
single set of world objects with **persistent global IDs**:

1. **Spatial dedup.** Detections within a gate across overlapping cameras
   (basler_1 ∩ basler_2 near the entry) are merged into one object.

2. **Closed-world formation re-ID across the gap.** The rig has a blind stretch
   between the upstream set (baslers, y≈−170…220) and the downstream set (lucid,
   y≈1090…1480). **No object can appear between the sets** — the belt only feeds
   from the basler side — so every object at lucid is an upstream object that
   travelled down: we re-ID it, we do not mint a new ID (a fresh ID downstream
   would be a detector artifact).

   The belt translates the whole formation +Y (and lucid's homography-only
   calibration adds a global offset), but the **relative geometry between bags is
   preserved**. So matching is done on the *formation as a whole*: estimate the
   belt translation Δ from bags already matched downstream (their lucid − origin
   offset, which absorbs the lucid offset too), then assign the arriving
   constellation to the departed formation by minimising |arrival − (departed+Δ)|
   — a translation-invariant, approximate-formation match. Cheap for a handful of
   bags, so it runs live.

Deliberately simple (greedy min-cost assignment, no Kalman): the belt is slow
and one-directional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


@dataclass
class _Track:
    id: int
    x: float
    y: float
    conf: float
    cams: list
    metric: bool
    last_t: float
    reid: bool = False               # True once restored across the gap
    hits: int = 1                    # times seen (confirmation)
    origin: tuple | None = None      # (x,y) of the departed bag it was re-ID'd from


@dataclass
class FusionTracker:
    dedup_gate_mm: float = 130.0      # merge same-bag detections across cameras
    assoc_gate_mm: float = 300.0      # match a detection to an existing track
    set2_entry_y_mm: float = 900.0    # y>=this = downstream (lucid); below = upstream
    max_match_mm: float = 500.0       # formation match: max residual to accept a re-ID
    min_hits: int = 2                 # a track must be seen >= this to count as real
    active_ttl_s: float = 1.2         # retire (and, if upstream, depart) after this gap
    departed_ttl_s: float = 90.0      # keep a departed bag re-ID-able this long

    _active: dict = field(default_factory=dict)
    _departed: dict = field(default_factory=dict)   # id -> {x, y, exit_t}
    _next_id: int = 1

    # ── spatial dedup ─────────────────────────────────────────────────────────
    def _dedup(self, dets):
        pts = [(float(d["x_mm"]), float(d["y_mm"]), float(d["conf"]),
                bool(d.get("metric", False)), d["cam"]) for d in dets]
        order = sorted(range(len(pts)), key=lambda i: -pts[i][2])
        used = [False] * len(pts)
        obs = []
        g2 = self.dedup_gate_mm ** 2
        for i in order:
            if used[i]:
                continue
            members = [pts[i]]
            used[i] = True
            for j in order:
                if not used[j] and (pts[j][0] - pts[i][0]) ** 2 + \
                        (pts[j][1] - pts[i][1]) ** 2 <= g2:
                    members.append(pts[j])
                    used[j] = True
            w = sum(m[2] for m in members) or 1.0
            obs.append({"x": sum(m[0] * m[2] for m in members) / w,
                        "y": sum(m[1] * m[2] for m in members) / w,
                        "conf": max(m[2] for m in members),
                        "cams": sorted({m[4] for m in members}),
                        "metric": any(m[3] for m in members)})
        return obs

    # ── belt translation Δ from already-matched siblings ─────────────────────
    def _belt_delta(self, arrivals):
        sib = [(tr.x - tr.origin[0], tr.y - tr.origin[1])
               for tr in self._active.values() if tr.origin is not None]
        if sib:
            return median(d[0] for d in sib), median(d[1] for d in sib)
        # bootstrap: shift the departed formation onto the arrivals by region medians
        ay = median(o["y"] for o in arrivals)
        dy = median(dp["y"] for dp in self._departed.values())
        return 0.0, ay - dy

    # ── approximate-formation assignment (closed world) ──────────────────────
    def _formation_match(self, arrivals):
        """Assign each arriving object to a departed bag by |arrival-(departed+Δ)|.

        Greedy min-cost over the pairwise costs (optimal enough for a handful of
        bags), gated by `max_match_mm` so a stray detection can't steal an ID.
        Returns {arrival_index_in_list: departed_id}."""
        if not self._departed:
            return {}
        dx, dy = self._belt_delta(arrivals)
        dep = list(self._departed.items())               # [(id, {x,y,exit_t})]
        cands = []
        for ai, o in enumerate(arrivals):
            for dj, (_did, dp) in enumerate(dep):
                d = ((o["x"] - (dp["x"] + dx)) ** 2 +
                     (o["y"] - (dp["y"] + dy)) ** 2) ** 0.5
                if d <= self.max_match_mm:
                    cands.append((d, ai, dj))
        cands.sort(key=lambda z: z[0])
        used_a, used_d, out = set(), set(), {}
        for _d, ai, dj in cands:
            if ai in used_a or dj in used_d:
                continue
            used_a.add(ai)
            used_d.add(dj)
            out[ai] = dep[dj][0]
        return out

    # ── main update ───────────────────────────────────────────────────────────
    def update(self, dets, t):
        obs = self._dedup(dets)

        # 1. associate observations to existing active tracks (greedy nearest)
        pairs = []
        for oi, o in enumerate(obs):
            for tid, tr in self._active.items():
                d2 = (o["x"] - tr.x) ** 2 + (o["y"] - tr.y) ** 2
                if d2 <= self.assoc_gate_mm ** 2:
                    pairs.append((d2, oi, tid))
        pairs.sort(key=lambda z: z[0])
        assigned, used_tracks = {}, set()
        for _d2, oi, tid in pairs:
            if oi not in assigned and tid not in used_tracks:
                assigned[oi] = tid
                used_tracks.add(tid)

        out, updated, arrivals = [], set(), []

        def emit(tid, o, reid, hits, origin):
            self._active[tid] = _Track(tid, o["x"], o["y"], o["conf"], o["cams"],
                                       o["metric"], t, reid, hits, origin)
            updated.add(tid)
            out.append({"id": int(tid), "x_mm": round(o["x"], 1),
                        "y_mm": round(o["y"], 1), "conf": round(o["conf"], 2),
                        "metric": o["metric"], "n_cams": len(o["cams"]),
                        "reid": reid})

        for oi, o in enumerate(obs):
            tid = assigned.get(oi)
            if tid is not None:                      # continues an existing track
                prev = self._active[tid]
                emit(tid, o, prev.reid, prev.hits + 1, prev.origin)
            elif o["y"] >= self.set2_entry_y_mm:     # new downstream obs -> re-ID later
                arrivals.append((oi, o))
            else:                                    # new upstream bag
                emit(self._new_id(), o, False, 1, None)

        # 2. closed-world formation re-ID for the downstream arrivals
        if arrivals:
            match = self._formation_match([o for _oi, o in arrivals])
            for k, (_oi, o) in enumerate(arrivals):
                did = match.get(k)
                if did is not None:
                    origin = (self._departed[did]["x"], self._departed[did]["y"])
                    del self._departed[did]
                    emit(did, o, True, 1, origin)
                else:                                # no plausible match -> new
                    emit(self._new_id(), o, False, 1, None)

        # 3. retire tracks not seen this tick; confirmed upstream ones "depart"
        for tid in list(self._active):
            if tid in updated:
                continue
            tr = self._active[tid]
            if t - tr.last_t > self.active_ttl_s:
                if tr.hits >= self.min_hits and tr.y < self.set2_entry_y_mm:
                    self._departed[tid] = {"x": tr.x, "y": tr.y, "exit_t": tr.last_t}
                del self._active[tid]
        for tid in list(self._departed):
            if t - self._departed[tid]["exit_t"] > self.departed_ttl_s:
                del self._departed[tid]
        return out

    def _new_id(self):
        i = self._next_id
        self._next_id += 1
        return i
