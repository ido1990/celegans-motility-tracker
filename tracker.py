"""Centroid tracker: frame-to-frame worm identity + lifespan + dead-state tracking."""
from dataclasses import dataclass, field
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_DISTANCE = 55.0
MAX_DISAPPEARED = 30
DEAD_POSITION_DELTA = 6.0
DEAD_BEND_DELTA = 12.0
DEAD_WINDOW_FRAMES = 90
# Holding a track through an overlap or a flickering outline (see CentroidTracker.update):
FUSED_AREA_RATIO = 1.5      # a blob this many times a worm's own usual area is two worms touching
OCCLUDED_PATIENCE = 150     # frames a worm may stay hidden inside a multi-worm blob
WEAK_PATIENCE = 90          # frames a worm may be seen only as an undersized/curled outline
WEAK_AREA_FRACTION = 0.25   # ...which must be at least this fraction of its usual area
OCCLUSION_MARGIN_PX = 8.0
# Experimental, off by default: on the validation videos it gives fewer but cleaner worms
# (reported worms -30%, per-video error 7.1 -> 6.5 thrashes/min) and renumbers worms, so it
# stays off until per-worm hand counts show it's worth it.
HOLD_THROUGH_OVERLAP = False

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
    midline_history: list = field(default_factory=list)
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
        t.midline_history.append(det.get("midline"))
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

    @staticmethod
    def _usual_area(t):
        areas = t.area_history[-50:]
        return float(np.median(areas)) if len(areas) >= 5 else None

    def _is_fused(self, t, det):
        """True when `det` is too big to be this worm alone: it's touching another worm."""
        usual = self._usual_area(t)
        return usual is not None and det.get("area", 0.0) > FUSED_AREA_RATIO * usual

    def _hidden_in(self, t, blobs):
        """True when the track's last real position lies in/next to a multi-worm blob."""
        pt = (float(t.centroid[0]), float(t.centroid[1]))
        usual = self._usual_area(t)
        for det in blobs:
            if not det.get("is_merged") and (usual is None or det.get("area", 0.0) <= FUSED_AREA_RATIO * usual):
                continue
            if cv2.pointPolygonTest(det["contour"], pt, True) >= -OCCLUSION_MARGIN_PX:
                return True
        return False

    def _weak_match(self, t, weak, taken):
        """Nearest undersized/curled outline that could be this worm seen badly."""
        usual = self._usual_area(t)
        if usual is None:
            return None
        best, best_d = None, self.max_distance / 2.0
        for k, det in enumerate(weak):
            if k in taken or det.get("area", 0.0) < WEAK_AREA_FRACTION * usual:
                continue
            d = ((t.centroid[0] - det["centroid"][0]) ** 2 + (t.centroid[1] - det["centroid"][1]) ** 2) ** 0.5
            if d < best_d:
                best, best_d = k, d
        return best

    def update(self, detections, frame_idx, weak=None):
        """`weak` (optional): outlines the classifier rejected as too small or too round.
        They never start a track or give a shape measurement, but an existing track with
        nothing better nearby follows one, so a worm whose outline flickers isn't lost."""
        weak = weak or []
        weak_taken = set()
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
                unmeasurable = det.get("is_merged") or (HOLD_THROUGH_OVERLAP and self._is_fused(t, det))
                if unmeasurable:
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
                t.bend_angle_history.append(None if unmeasurable else det.get("bend_angle"))
                t.midline_history.append(None if unmeasurable else det.get("midline"))
                t.centroid_history.append(t.centroid)
                self._update_dead_state(t)
                matched_track_ids.add(tid)
                matched_det_idxs.add(c)

        for tid in track_ids:
            if tid in matched_track_ids:
                continue
            t = self.tracks[tid]
            t.disappeared_count += 1
            patience = self.max_disappeared
            if HOLD_THROUGH_OVERLAP:
                if self._hidden_in(t, all_dets):
                    patience = max(patience, OCCLUDED_PATIENCE)
                else:
                    k = self._weak_match(t, weak, weak_taken)
                    if k is not None:
                        weak_taken.add(k)
                        t.centroid = weak[k]["centroid"]
                        t.contour = weak[k]["contour"]
                        patience = max(patience, WEAK_PATIENCE)
            t.bend_angle_history.append(None)
            t.midline_history.append(None)
            t.centroid_history.append(t.centroid)
            if t.disappeared_count > patience:
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

    # Experimental hold-through-overlap: a worm hidden inside a multi-worm blob outlives the
    # normal patience, and a blob far bigger than the worm itself gives no shape measurement.
    def _sq(cx, cy, half):
        return np.array([[[cx - half, cy - half]], [[cx + half, cy - half]],
                         [[cx + half, cy + half]], [[cx - half, cy + half]]], dtype=np.int32)
    HOLD_THROUGH_OVERLAP = True
    hold = CentroidTracker(max_distance=50, max_disappeared=3, dead_window_frames=1000)
    for f in range(6):
        hold.update([{"centroid": (20, 20), "contour": _sq(20, 20, 5), "area": 100, "bend_angle": 170.0},
                     {"centroid": (40, 20), "contour": _sq(40, 20, 5), "area": 100, "bend_angle": 170.0}], f)
    ids = set(hold.tracks)
    for f in range(6, 16):  # the two touch: one blob of twice the area, for longer than max_disappeared
        hold.update([{"centroid": (30, 20), "contour": _sq(30, 20, 16), "area": 200, "bend_angle": 90.0}], f)
    assert set(hold.tracks) == ids, "both worms must survive an overlap longer than the normal patience"
    assert all(t.bend_angle_history[-1] is None for t in hold.tracks.values()), (
        "a fused blob must not be measured as one worm's shape")
    HOLD_THROUGH_OVERLAP = False

    print("tracker.py self-check: PASS")
