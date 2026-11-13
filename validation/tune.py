"""Tunes the thrash-counting parameters against per-worm hand counts and saves the result.

    python validation/tune.py --videos <folder with the counted videos>
                              [--counts validation/worm_counts_2026-09-10.csv] [--dry-run]

What it does:
1. Runs the tracker once on every counted video (cached under <videos>/.tune_cache, so
   later runs take seconds; pass --refresh after changing detection or tracking code).
2. Matches every hand count to its worm and searches the counting parameters
   (analyzer.TUNABLE) for the lowest mean error against the hand counts.
3. Checks the gain honestly: the same search is repeated with each video held out in turn
   and scored only on the held-out video. Only if that held-out error beats the current
   settings by --min-gain does it write tuned_params.json, which analyzer.py loads on
   import (and build.py bundles into the executable).

It tunes how thrashes are counted from the tracked worms, not how worms are detected or
tracked: those change which worms exist, so every trial would need a full re-run of every
video. More hand counts make the result more trustworthy; with few, expect "no change".
"""
import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import analyzer  # noqa: E402
import compare_worms  # noqa: E402

# name -> candidate values, searched one parameter at a time (coordinate descent)
SEARCH = {
    "CL_PROMINENCE_DEG": [20, 25, 30, 35, 40, 45, 50, 60, 70],
    "CL_PEAKS_PER_S": [4, 5, 6, 7.5, 10],
    "CL_MIN_SEGMENT": [12, 18, 25, 35, 50],
    "CL_LENGTH_TOL": [0.2, 0.25, 0.3, 0.4, 0.5],
    "CL_SIDE_MIN": [0.5, 0.6, 0.7, 0.8],
    "CL_MAX_JUMP_DEG": [60, 90, 120, 180],
}


def code_fingerprint():
    h = hashlib.sha1()
    for name in ("analyzer.py", "tracker.py", "gui.py"):
        with open(os.path.join(HERE, "..", name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def load_samples(videos_dir, counts_path, tracks_dir, refresh, min_hand_s):
    """One sample per usable hand count: its rate and the worm's centerlines in that window."""
    cache_dir = os.path.join(videos_dir, ".tune_cache")
    os.makedirs(cache_dir, exist_ok=True)
    samples, skipped = [], 0
    for video, rows in sorted(compare_worms.load_counts(counts_path).items()):
        path = os.path.join(videos_dir, video if video.endswith(".mp4") else video + ".mp4")
        if not os.path.exists(path):
            print(f"  {video}: video not found, its {len(rows)} count(s) are skipped")
            skipped += len(rows)
            continue
        cache = os.path.join(cache_dir, os.path.basename(path) + ".pkl")
        data = None
        if os.path.exists(cache) and not refresh:
            with open(cache, "rb") as f:
                data = pickle.load(f)
            if data.get("code") != code_fingerprint():
                data = None
        if data is None:
            print(f"  tracking {video} ...", flush=True)
            tracks, fps = compare_worms.collect_tracks(path, all_tracks=True)
            data = {"code": code_fingerprint(), "fps": fps, "tracks": tracks}
            with open(cache, "wb") as f:
                pickle.dump(data, f)
        seen = set()
        for c in rows:
            key = (c["worm_id"], c["t_start"], c["t_end"])
            if key in seen or c["time_s"] < min_hand_s:
                skipped += 1
                continue
            seen.add(key)
            t = compare_worms.find_track(data["tracks"], data["fps"], c, path, tracks_dir)
            if t is None:
                skipped += 1
                continue
            win = compare_worms.window(t, data["fps"], c["t_start"], c["t_end"])
            samples.append({"video": video, "fps": data["fps"], "hand": c["thrashes"] / c["time_s"] * 60.0,
                            "midlines": t.midline_history[win]})
    return samples, skipped


def set_params(params):
    for k, v in params.items():
        setattr(analyzer, k, type(getattr(analyzer, k))(v))


def current_params():
    return {k: getattr(analyzer, k) for k in analyzer.TUNABLE}


def evaluate(samples, params, min_measured_s=4.0, missing_penalty=40.0):
    """Mean abs error (thrashes/min). A worm the settings can't measure counts as a
    fixed penalty, so the search can't win by measuring only the easy worms."""
    saved = current_params()
    set_params(params)
    try:
        errs = []
        for s in samples:
            thr, meas = analyzer.count_thrashes_centerline(
                s["midlines"], min_distance_frames=analyzer.centerline_min_distance(s["fps"]))
            if meas < min_measured_s * s["fps"]:
                errs.append(missing_penalty)
            else:
                errs.append(abs(thr / (meas / s["fps"]) * 60.0 - s["hand"]))
        return float(np.mean(errs)) if errs else float("nan")
    finally:
        set_params(saved)


def search(samples, start, passes=2):
    best, best_err = dict(start), evaluate(samples, start)
    for _ in range(passes):
        for name, values in SEARCH.items():
            for v in values:
                trial = dict(best, **{name: v})
                err = evaluate(samples, trial)
                if err < best_err - 1e-9:
                    best, best_err = trial, err
    return best, best_err


def held_out_error(samples, start):
    """Tune on all videos but one, score on that one; repeat for every video."""
    videos = sorted({s["video"] for s in samples})
    base, tuned, weights = [], [], []
    for v in videos:
        train = [s for s in samples if s["video"] != v]
        test = [s for s in samples if s["video"] == v]
        params, _ = search(train, start)
        base.append(evaluate(test, start))
        tuned.append(evaluate(test, params))
        weights.append(len(test))
    return float(np.average(base, weights=weights)), float(np.average(tuned, weights=weights))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--counts", default=os.path.join(HERE, "worm_counts_2026-09-10.csv"))
    ap.add_argument("--tracks-dir", default=os.path.join(HERE, "tracks"))
    ap.add_argument("--min-hand-s", type=float, default=5.0, help="ignore hand counts shorter than this")
    ap.add_argument("--min-gain", type=float, default=1.0,
                    help="held-out improvement (thrashes/min) required before saving new settings")
    ap.add_argument("--refresh", action="store_true", help="re-run tracking even if cached")
    ap.add_argument("--dry-run", action="store_true", help="report only, never write tuned_params.json")
    args = ap.parse_args()

    samples, skipped = load_samples(args.videos, args.counts, args.tracks_dir, args.refresh, args.min_hand_s)
    n_videos = len({s["video"] for s in samples})
    print(f"\n{len(samples)} usable hand counts from {n_videos} videos ({skipped} skipped: too short, "
          f"duplicate, video missing or worm not tracked)")
    if len(samples) < 10 or n_videos < 3:
        print("Too few to tune on (need at least 10 counts across 3 videos). Nothing changed.")
        return

    start = current_params()
    before = evaluate(samples, start)
    best, after = search(samples, start)
    cv_before, cv_after = held_out_error(samples, start)
    print(f"\ncurrent settings : {before:5.1f} thrashes/min mean error")
    print(f"best found       : {after:5.1f}   (on the same counts it was tuned on, so optimistic)")
    print(f"held-out check   : {cv_before:5.1f} -> {cv_after:5.1f}   (tuned without each video, scored on it)")
    changed = {k: (start[k], best[k]) for k in best if best[k] != start[k]}
    for k, (a, b) in changed.items():
        print(f"  {k}: {a} -> {b}")

    if not changed:
        print("\nThe current settings are already the best found. Nothing changed.")
    elif cv_before - cv_after < args.min_gain:
        print(f"\nThe held-out gain is under {args.min_gain}/min, so this is likely fitting noise. "
              "Nothing changed. More hand counts would help.")
    elif args.dry_run:
        print("\n--dry-run: not saved.")
    else:
        out = analyzer.tuned_params_path()
        with open(out, "w") as f:
            json.dump({"params": best, "tuned_on": {"counts": len(samples), "videos": n_videos,
                                                    "error_before": round(before, 2), "error_after": round(after, 2),
                                                    "held_out_before": round(cv_before, 2),
                                                    "held_out_after": round(cv_after, 2)}}, f, indent=2)
        print(f"\nSaved to {out}. The tracker uses these from the next run; commit the file to keep them.")


if __name__ == "__main__":
    main()
