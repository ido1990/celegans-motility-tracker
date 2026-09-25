"""Per-worm comparison: each hand count names a worm ID from the numbered videos
(validation/annotate.py) and, when tap-counted, the exact video window it covers. The
tracker's thrash rate for that same worm over that same window is computed with every
counting method, so methods are scored on identical worms and time spans.

Usage:
    python validation/compare_worms.py --videos <folder> --counts <per-worm counts CSV>

The counts CSV has columns video, worm_id, thrashes, time_s, t_start, t_end (t_start/t_end
empty for counts entered by hand over the worm's whole visible time).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import analyzer  # noqa: E402
import gui  # noqa: E402


def collect_tracks(path):
    """Runs the pipeline once and returns every reported track object by ID, plus fps."""
    tracks = {}

    def on_frame(_, live):
        tracks.update(live)

    writer = csv.DictWriter(open(os.devnull, "w"), fieldnames=gui.CSV_COLUMNS)
    _, rows = gui.process_video(path, True, writer, on_frame=on_frame)
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return {r["worm_id"]: tracks[r["worm_id"]] for r in rows if r["worm_id"] in tracks}, fps


def window(track, fps, t_start, t_end):
    if t_start is None:
        return slice(None)
    a = max(0, int(round(t_start * fps)) - track.frame_entry)
    b = max(a, int(round(t_end * fps)) - track.frame_entry + 1)
    return slice(a, b)


def method_rates(track, fps, win):
    out = {}
    thr, meas = analyzer.count_thrashes_measured(
        track.bend_angle_history[win], gui.DEFAULT_SENSITIVITY, min_distance_frames=max(1, int(fps // 10)))
    out["measured"] = (thr / (meas / fps) * 60.0 if meas else None, meas / fps)
    thr, meas = analyzer.count_thrashes_centerline(track.midline_history[win])
    out["centerline"] = (thr / (meas / fps) * 60.0 if meas else None, meas / fps)
    return out


def load_counts(path):
    by_video = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by_video[r["video"]].append({
                "worm_id": int(r["worm_id"]), "thrashes": float(r["thrashes"]), "time_s": float(r["time_s"]),
                "t_start": float(r["t_start"]) if r.get("t_start") not in (None, "") else None,
                "t_end": float(r["t_end"]) if r.get("t_end") not in (None, "") else None,
            })
    return by_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--min-overlap-s", type=float, default=5.0,
                    help="skip a worm if the tracker measured it for less than this within the window")
    args = ap.parse_args()

    counts = load_counts(args.counts)
    methods = ("measured", "centerline")
    errs = {m: [] for m in methods}
    print(f"{'video':<14}{'worm':>5}{'hand':>7}" + "".join(f"{m:>12}" for m in methods))
    for video, rows in sorted(counts.items()):
        path = os.path.join(args.videos, video if video.endswith(".mp4") else video + ".mp4")
        if not os.path.exists(path):
            print(f"{video}: video not found in {args.videos}, skipped")
            continue
        tracks, fps = collect_tracks(path)
        for c in sorted(rows, key=lambda r: r["worm_id"]):
            hand = c["thrashes"] / c["time_s"] * 60.0
            t = tracks.get(c["worm_id"])
            cells = []
            for m in methods:
                if t is None:
                    cells.append("no track")
                    continue
                rate, secs = method_rates(t, fps, window(t, fps, c["t_start"], c["t_end"]))[m]
                if rate is None or secs < args.min_overlap_s:
                    cells.append("too short")
                    continue
                errs[m].append(rate - hand)
                cells.append(f"{rate:6.1f} {rate - hand:+5.1f}")
            short = video.replace("bandicam ", "")[-12:]
            print(f"{short:<14}{c['worm_id']:>5}{hand:>7.1f}" + "".join(f"{x:>12}" for x in cells))

    print()
    for m in methods:
        e = np.array(errs[m])
        if len(e):
            print(f"{m:<11} worms {len(e):3d}  mean abs error {np.mean(np.abs(e)):5.1f}/min  "
                  f"bias {np.mean(e):+5.1f}  within 10/min: {np.mean(np.abs(e) <= 10):4.0%}")


if __name__ == "__main__":
    main()
