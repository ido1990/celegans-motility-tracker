"""Compares the tracker's thrash rates against hand counts, per video.

Hand counts cover a subset of worms per video with no position info, so worms can't be
paired one-to-one — the comparison is per video: mean/median thrash rate of the tracker's
worms vs. the counter's.

Usage:
    python validation/compare.py --videos <folder> [--manual validation/manual_counts_2026-09-03.csv]
                                 [--min-measured-s 5] [--set sensitivity=20 ...]
"""
import argparse
import csv
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gui  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_manual(path):
    by_video = defaultdict(list)
    meta = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by_video[r["video"]].append(float(r["thrashes_per_min"]))
            meta[r["video"]] = (r["strain"], r["treatment"])
    return by_video, meta


def parse_params(pairs):
    params = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        params[k] = float(v)
    return params


def tracker_rates(rows, fps, min_measured_s):
    by_video = defaultdict(list)
    for r in rows:
        if r["final_state"] == "DEAD":
            continue
        if int(r["measured_frames"]) / fps < min_measured_s:
            continue
        by_video[r["source_video"]].append(float(r["thrash_rate_per_min"]))
    return by_video


def summarize(manual, meta, auto):
    print(f"{'video':<38}{'group':<22}{'n_man':>6}{'man_mean':>9}{'n_auto':>7}"
          f"{'auto_mean':>10}{'auto_med':>9}{'err':>8}")
    errs = []
    for video in sorted(manual):
        m = manual[video]
        a = auto.get(video, [])
        am = float(np.mean(a)) if a else float("nan")
        amed = float(np.median(a)) if a else float("nan")
        err = am - float(np.mean(m))
        errs.append(err)
        group = " ".join(meta[video])
        print(f"{video[9:-4]:<38}{group:<22}{len(m):>6}{np.mean(m):>9.1f}{len(a):>7}"
              f"{am:>10.1f}{amed:>9.1f}{err:>+8.1f}")
    errs = np.array(errs)
    print(f"\nmean abs error: {np.nanmean(np.abs(errs)):.1f} thrashes/min, "
          f"bias: {np.nanmean(errs):+.1f}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--manual", default=os.path.join(HERE, "manual_counts_2026-09-03.csv"))
    ap.add_argument("--min-measured-s", type=float, default=5.0,
                    help="ignore worms whose body shape was measured for less than this "
                         "(a hand counter wouldn't count them either)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--set", nargs="*", help="parameter overrides, e.g. sensitivity=20")
    args = ap.parse_args()

    manual, meta = load_manual(args.manual)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        out = tmp.name
    rows = gui.run_batch(args.videos, out, dry_run=True, params=parse_params(args.set))
    summarize(manual, meta, tracker_rates(rows, args.fps, args.min_measured_s))


if __name__ == "__main__":
    main()
