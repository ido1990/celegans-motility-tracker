"""Centroid tracker: frame-to-frame worm identity + lifespan + dead-state tracking."""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_DISTANCE = 55.0
MAX_DISAPPEARED = 30
DEAD_POSITION_DELTA = 6.0
DEAD_BEND_DELTA = 12.0
DEAD_WINDOW_FRAMES = 90

ACTIVE = "ACTIVE"
DEAD = "DEAD"


@dataclass
class WormTrack:
    id: int
    centroid: tuple
    contour: np.ndarray
    frame_entry: int
    last_seen_frame: int
    bend_angle_history: list = field(default_factory=list)
    centroid_history: list = field(default_factory=list)
    area_history: list = field(default_factory=list)
    disappeared_count: int = 0
    state: str = ACTIVE


class CentroidTracker:
    def __init__(self, max_distance=MAX_DISTANCE, max_disappeared=MAX_DISAPPEARED,
                 dead_position_delta=DEAD_POSITION_DELTA, dead_bend_delta=DEAD_BEND_DELTA,
                 dead_window_frames=DEAD_WINDOW_FRAMES):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.dead_position_delta = dead_position_delta
        self.dead_bend_delta = dead_bend_delta
        self.dead_window_frames = dead_window_frames
        self.tracks = {}
        self.finalized = []
        self._next_id = 1

    def _register(self, det, frame_idx):
        tid = self._next_id
        self._next_id += 1
        t = WormTrack(id=tid, centroid=det["centroid"], contour=det["contour"],
                      frame_entry=frame_idx, last_seen_frame=frame_idx)
        t.bend_angle_history.append(det.get("bend_angle"))
        t.centroid_history.append(det["centroid"])
        t.area_history.append(det.get("area", 0.0))
        self.tracks[tid] = t
        return tid

    def _deregister(self, tid):
        t = self.tracks.pop(tid)
        self.finalized.append(t)

    def _update_dead_state(self, t):
        window = t.centroid_history[-self.dead_window_frames:]
        if len(window) < self.dead_window_frames:
            return
        xs = [p[0] for p in window]
        ys = [p[1] for p in window]
        pos_delta = max(xs) - min(xs) + max(ys) - min(ys)

        bend_window = [a for a in t.bend_angle_history[-self.dead_window_frames:] if a is not None]
        bend_delta = (max(bend_window) - min(bend_window)) if bend_window else 0.0

        if pos_delta < self.dead_position_delta and bend_delta < self.dead_bend_delta:
            t.state = DEAD
        else:
            t.state = ACTIVE

    def update(self, detections, frame_idx):
        candidates = [d for d in detections if not d.get("is_merged")]
        merged = [d for d in detections if d.get("is_merged")]
        all_dets = candidates + merged

        track_ids = list(self.tracks.keys())
        matched_track_ids = set()
        matched_det_idxs = set()

        if track_ids and all_dets:
            cost = np.zeros((len(track_ids), len(all_dets)))
            for i, tid in enumerate(track_ids):
                tc = self.tracks[tid].centroid
                for j, d in enumerate(all_dets):
                    dc = d["centroid"]
                    cost[i, j] = ((tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2) ** 0.5

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] > self.max_distance:
                    continue
                tid = track_ids[r]
                det = all_dets[c]
                t = self.tracks[tid]
                if det.get("is_merged"):
                    # A merged blob's blended centroid is an average of (at least) two worms'
                    # positions — not this worm's true position. Keep the last real position
                    # instead of jumping to it (same treatment as a miss, below); only the
                    # contour follows the blob so the on-screen box tracks what's visible.
                    t.contour = det["contour"]
                else:
                    t.centroid = det["centroid"]
                    t.contour = det["contour"]
                    t.area_history.append(det.get("area", 0.0))
                t.last_seen_frame = frame_idx
                t.disappeared_count = 0
                angle = None if det.get("is_merged") else det.get("bend_angle")
                t.bend_angle_history.append(angle)
                t.centroid_history.append(t.centroid)
                self._update_dead_state(t)
                matched_track_ids.add(tid)
                matched_det_idxs.add(c)

        for tid in track_ids:
            if tid in matched_track_ids:
                continue
            t = self.tracks[tid]
            t.disappeared_count += 1
            t.bend_angle_history.append(None)
            t.centroid_history.append(t.centroid)
            if t.disappeared_count > self.max_disappeared:
                self._deregister(tid)

        for j, det in enumerate(candidates):
            if j in matched_det_idxs:
                continue
            self._register(det, frame_idx)

        return self.tracks

    def finalize_all(self):
        for tid in list(self.tracks.keys()):
            self._deregister(tid)
        return self.finalized


if __name__ == "__main__":
    tracker = CentroidTracker(max_distance=50, max_disappeared=5, dead_window_frames=1000)

    frame1 = [
        {"centroid": (10, 10), "contour": np.zeros((3, 1, 2), dtype=int), "area": 100},
        {"centroid": (100, 100), "contour": np.zeros((3, 1, 2), dtype=int), "area": 100},
    ]
    tracker.update(frame1, 0)
    ids_before = set(tracker.tracks.keys())
    assert len(ids_before) == 2, f"expected 2 tracks registered, got {len(ids_before)}"

    frame2 = [
        {"centroid": (55, 55), "contour": np.zeros((3, 1, 2), dtype=int), "area": 250,
         "is_merged": True},
    ]
    tracker.update(frame2, 1)
    assert set(tracker.tracks.keys()) == ids_before, "merge frame must not create a new ID"
    assert all(tracker.tracks[t].disappeared_count == 1 for t in ids_before), (
        "both original tracks should be marked missed (not matched) during a merge blob"
    )

    frame3 = [
        {"centroid": (12, 12), "contour": np.zeros((3, 1, 2), dtype=int), "area": 100},
        {"centroid": (102, 102), "contour": np.zeros((3, 1, 2), dtype=int), "area": 100},
    ]
    tracker.update(frame3, 2)
    assert set(tracker.tracks.keys()) == ids_before, "IDs must survive a merge/split with no churn"
    assert all(tracker.tracks[t].disappeared_count == 0 for t in ids_before)

    # A merged blob's blended centroid must not corrupt the matched track's position: if two
    # tracks at very different spots produce one merged detection near their midpoint, the
    # track that "wins" the Hungarian match should stay at its own last real position, not
    # snap to the blend.
    corrupt = CentroidTracker(max_distance=200, max_disappeared=5, dead_window_frames=1000)
    corrupt.update([
        {"centroid": (0, 0), "contour": np.zeros((3, 1, 2), dtype=int), "area": 100},
        {"centroid": (100, 100), "contour": np.zeros((3, 1, 2), dtype=int), "area": 100},
    ], 0)
    before = {tid: t.centroid for tid, t in corrupt.tracks.items()}
    corrupt.update([
        {"centroid": (50, 50), "contour": np.zeros((3, 1, 2), dtype=int), "area": 500,
         "is_merged": True},
    ], 1)
    matched = [tid for tid, t in corrupt.tracks.items() if t.disappeared_count == 0]
    assert len(matched) == 1, f"exactly one track should match the merged blob, got {matched}"
    assert corrupt.tracks[matched[0]].centroid == before[matched[0]], (
        "a merge match must not move the track's centroid to the blended blob's centroid"
    )

    print("tracker.py self-check: PASS")
