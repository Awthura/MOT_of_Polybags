"""
Per-camera track IDs — OVGU AMS algorithm phase.

Deliberately minimal. There is no cross-camera fusion in this phase; each camera
keeps its own set of IDs so a bag drawn on the map has a stable label while it
stays in one camera's view. This is a greedy nearest-neighbour associator on the
belt-plane positions (millimetres), with a gate distance and a time-to-live —
enough to hold an ID across the small frame-to-frame motion at ~5 fps, not a
Kalman/ByteTrack tracker (that arrives with the fusion phase).

One `Tracker` per camera. `update()` takes this frame's detections in world mm
and returns each one tagged with an ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Track:
    id: int
    xy: np.ndarray            # last belt-mm position
    last_t: float
    conf: float = 0.0


@dataclass
class Tracker:
    gate_mm: float = 300.0    # max frame-to-frame jump to keep the same ID
    ttl_s: float = 1.0        # drop a track unseen this long
    _tracks: dict[int, _Track] = field(default_factory=dict)
    _next_id: int = 1

    def update(self, dets_xy_conf, t: float) -> list[tuple[int, float, float, float]]:
        """Associate detections to IDs.

        `dets_xy_conf`: iterable of (x_mm, y_mm, conf). Returns a list of
        (id, x_mm, y_mm, conf), same order as the input.
        """
        dets = [(float(x), float(y), float(c)) for x, y, c in dets_xy_conf]
        self._prune(t)

        # Candidate (distance, det_idx, track_id) pairs within the gate.
        live = list(self._tracks.values())
        pairs = []
        for di, (x, y, _c) in enumerate(dets):
            p = np.array([x, y])
            for tr in live:
                d = float(np.linalg.norm(p - tr.xy))
                if d <= self.gate_mm:
                    pairs.append((d, di, tr.id))
        pairs.sort(key=lambda z: z[0])

        det_to_id: dict[int, int] = {}
        used_ids: set[int] = set()
        for d, di, tid in pairs:
            if di in det_to_id or tid in used_ids:
                continue
            det_to_id[di] = tid
            used_ids.add(tid)

        out = []
        for di, (x, y, c) in enumerate(dets):
            tid = det_to_id.get(di)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
            self._tracks[tid] = _Track(id=tid, xy=np.array([x, y]),
                                       last_t=t, conf=c)
            out.append((tid, x, y, c))
        return out

    def _prune(self, t: float) -> None:
        dead = [tid for tid, tr in self._tracks.items()
                if t - tr.last_t > self.ttl_s]
        for tid in dead:
            del self._tracks[tid]
