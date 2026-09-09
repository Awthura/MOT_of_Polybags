"""
Fusion + global re-ID — OVGU AMS algorithm phase.

Turns per-camera, per-frame detections (already in belt millimetres) into a
single set of world objects with **persistent global IDs**:

1. **Spatial dedup.** Detections within a gate across overlapping cameras
   (basler_1 ∩ basler_2 near the entry) are merged into one object.

2. **FIFO re-ID across the blind gap.** The rig has a blind stretch between the
   upstream set (baslers, y≈−170…220) and the downstream set (lucid,
   y≈1090…1480). The belt is a single, one-directional conveyor, so it behaves
   as a **FIFO queue: bags cannot pass each other** — the order in which they
   leave the baslers is the order in which they reach lucid. So re-ID does NOT
   depend on lucid's (unreliable, homography-only) world-X: when a confirmed
   upstream bag stops being seen we push its ID onto a departure queue, and when
   a bag appears at lucid we hand it the ID at the head of that queue.

   A **belt-speed transit-time gate** guards the queue: belt speed is estimated
   live from how fast tracked bags move along +Y, giving an expected gap-transit
   time. An arrival too early for the head bag (it can't physically be here yet)
   is treated as an untracked bag → fresh ID; a head bag long overdue (missed at
   lucid) is dropped from the queue so it can't steal a later arrival's ID.

Deliberately simple (greedy dedup/association, ordered queue, no Kalman): the
belt is slow and one-directional.
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
    min_hits: int = 2                 # a track must be seen >= this to count as real
    active_ttl_s: float = 1.2         # retire (and, if upstream, depart) after this gap
    departed_ttl_s: float = 90.0      # keep a departed bag re-ID-able this long

    # belt-speed transit-time gate (see module docstring)
    default_belt_speed_mm_s: float = 0.0   # fallback until enough motion is measured
                                           # (0 = unknown -> pure FIFO order, no time gate)
    transit_tol: float = 0.5          # accept transit time within +-this fraction of expected
    transit_slack_s: float = 2.0      # extra absolute slack on the late side before "overdue"

    _active: dict = field(default_factory=dict)
    _departed: list = field(default_factory=list)   # FIFO queue, oldest first:
                                                     #   [{id, x, y, exit_t}, ...]
    _speed_samples: list = field(default_factory=list)
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

    # ── belt speed (mm/s along +Y) from within-camera motion ─────────────────
    def _belt_speed(self):
        if self._speed_samples:
            return median(self._speed_samples)
        return max(self.default_belt_speed_mm_s, 0.0)

    def _add_speed_sample(self, prev, o, t):
        dy = o["y"] - prev.y
        dt = t - prev.last_t
        if 0.02 <= dt <= 2.0 and 2.0 <= dy <= 600.0:   # forward, plausible step
            self._speed_samples.append(dy / dt)
            if len(self._speed_samples) > 200:
                self._speed_samples.pop(0)

    # ── FIFO re-ID across the gap (closed world, order-preserving) ────────────
    def _fifo_reid(self, arrivals, t):
        """Hand each downstream arrival the ID at the head of the departure queue.

        arrivals: list of (obs_index, obs). Returns {obs_index: global_id}.
        Arrivals are consumed in arrival order (furthest downstream = arrived
        first; across-belt x only breaks ties). A belt-speed transit-time gate
        skips arrivals that can't physically be the head bag yet, and drops head
        bags that are long overdue."""
        out = {}
        if not self._departed:
            return out
        speed = self._belt_speed()
        for oi, o in sorted(arrivals, key=lambda p: (-p[1]["y"], p[1]["x"])):
            while self._departed:
                dep = self._departed[0]
                if speed > 0:
                    expected = (o["y"] - dep["y"]) / speed
                    actual = t - dep["exit_t"]
                    if expected > 0:
                        if actual < expected * (1 - self.transit_tol):
                            break                       # head not here yet -> untracked
                        if actual > expected * (1 + self.transit_tol) + self.transit_slack_s:
                            self._departed.pop(0)        # head overdue -> drop, try next
                            continue
                out[oi] = dep["id"]
                self._departed.pop(0)
                break
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
                self._add_speed_sample(prev, o, t)   # learn belt speed from motion
                emit(tid, o, prev.reid, prev.hits + 1, prev.origin)
            elif o["y"] >= self.set2_entry_y_mm:     # new downstream obs -> re-ID
                arrivals.append((oi, o))
            else:                                    # new upstream bag
                emit(self._new_id(), o, False, 1, None)

        # 2. FIFO re-ID for the downstream arrivals
        if arrivals:
            match = self._fifo_reid(arrivals, t)
            for oi, o in arrivals:
                did = match.get(oi)
                if did is not None:
                    emit(did, o, True, 1, None)      # restore the global ID
                else:                                # no queued bag -> new
                    emit(self._new_id(), o, False, 1, None)

        # 3. retire tracks not seen this tick; confirmed upstream ones "depart"
        departing = []
        for tid in list(self._active):
            if tid in updated:
                continue
            tr = self._active[tid]
            if t - tr.last_t > self.active_ttl_s:
                if tr.hits >= self.min_hits and tr.y < self.set2_entry_y_mm:
                    departing.append(tr)
                del self._active[tid]
        # queue departures in belt order (further downstream = left first)
        for tr in sorted(departing, key=lambda r: -r.y):
            self._departed.append({"id": tr.id, "x": tr.x, "y": tr.y,
                                   "exit_t": tr.last_t})
        # drop stale departures (never re-appeared)
        self._departed = [d for d in self._departed
                          if t - d["exit_t"] <= self.departed_ttl_s]
        return out

    def _new_id(self):
        i = self._next_id
        self._next_id += 1
        return i
