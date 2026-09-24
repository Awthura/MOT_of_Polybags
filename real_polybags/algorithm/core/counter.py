"""
Line counter — OVGU AMS algorithm phase.

Counts objects that cross a fixed belt line in the travel direction (+Y), once
per track. Used on both the fused stream (one counter) and each raw camera
stream (one counter per camera); the raw counts' maximum is the reference the
fused count should match — fusion must merge the overlapping cameras into one
object per bag, so it should neither double-count (fused > max) nor drop bags
(fused < max).

Parameters are purely physical/geometric (a belt y-line and an x-segment in mm),
derived from the rig geometry — no belt-speed, frame-rate, or clip-specific
temporal constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LineCounter:
    y_mm: float                              # the line: belt Y (perpendicular to travel)
    x_mm: tuple = (-1e9, 1e9)                # only count crossings within this x-segment
    _prev_y: dict = field(default_factory=dict)   # track id -> last y seen
    _counted: set = field(default_factory=set)    # ids already counted (no double count)
    count: int = 0

    def update(self, objs) -> int:
        """objs: iterable of dicts with 'id', 'x_mm', 'y_mm'. Returns the count.

        A track is counted the first time it moves from below the line to on/above
        it (forward, +Y), within the x-segment. The counted-id set makes it
        idempotent to jitter and re-observation of the same track.

        NOTE: crossing-per-track counting fails on this rig's DENSE, fast flow
        (up to 12 to 21 bags per frame, ~88 mm of travel per 5 Hz tick vs ~105 mm
        bag spacing): tracks merge or fragment and most crossings are lost. Use
        FluxCounter below for the throughput count; this stays for reference.
        """
        for o in objs:
            i = o["id"]
            y = float(o["y_mm"])
            x = float(o["x_mm"])
            py = self._prev_y.get(i)
            if (py is not None and py < self.y_mm <= y
                    and i not in self._counted
                    and self.x_mm[0] <= x <= self.x_mm[1]):
                self.count += 1
                self._counted.add(i)
            self._prev_y[i] = y
        return self.count


@dataclass
class FluxCounter:
    """Occupancy-flux counter for dense flow (no per-bag tracking).

    Counts how many bags pass a line by measuring how much "bag-presence"
    accumulates in a thin band around it and dividing by how long one bag lingers
    there (Little's law): N = (integral over time of #detections in the band) *
    belt_speed / band_thickness. Robust to dense, fast flow where individual
    tracking is ambiguous; needs only raw detections and the belt speed.
    Sensitive to detector recall (misses lower it) and false positives (raise it),
    so report it as an approximate throughput estimate.
    """
    y_mm: float                          # line position (belt Y)
    band_mm: float = 200.0               # band thickness centred on the line
    x_mm: tuple = (-1e9, 1e9)            # only count within this across-belt segment
    integral: float = 0.0                # sum over ticks of (#in band) * dt seconds

    def update(self, objs, dt: float) -> None:
        """Accumulate band occupancy for one tick of real duration `dt` seconds."""
        lo, hi = self.y_mm - self.band_mm / 2.0, self.y_mm + self.band_mm / 2.0
        n = sum(1 for o in objs
                if lo <= float(o["y_mm"]) <= hi
                and self.x_mm[0] <= float(o["x_mm"]) <= self.x_mm[1])
        self.integral += n * max(dt, 0.0)

    def count(self, belt_speed_mm_s: float) -> float:
        """Throughput estimate given the current belt speed (mm/s)."""
        if self.band_mm <= 0 or belt_speed_mm_s <= 0:
            return 0.0
        return self.integral * belt_speed_mm_s / self.band_mm


@dataclass
class SpeedEstimator:
    """Online belt-speed estimate (mm/s along +Y) from tracked motion.

    Belt speed is a physical property of the rig, measured live rather than tuned;
    `default` is only a cold-start fallback until enough motion is seen.
    """
    default: float = 0.0
    _prev: dict = field(default_factory=dict)     # track id -> (y, t)
    _samples: list = field(default_factory=list)

    def update(self, objs, t: float) -> None:
        for o in objs:
            i = o["id"]
            y = float(o["y_mm"])
            if i in self._prev:
                py, pt = self._prev[i]
                dy, dt = y - py, t - pt
                if 0.02 <= dt <= 1.0 and 2.0 <= dy <= 600.0:   # forward, plausible
                    self._samples.append(dy / dt)
                    if len(self._samples) > 300:
                        self._samples.pop(0)
            self._prev[i] = (y, t)

    def speed(self) -> float:
        if self._samples:
            s = sorted(self._samples)
            return s[len(s) // 2]                 # median
        return self.default
