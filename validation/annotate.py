"""Renders a copy of each video with the tracker's worm IDs drawn on it, for per-worm hand
counting: the counter picks a numbered worm, counts its thrashes, and records the number,
so each hand count can be matched to exactly one tracker row.

Only worms the pipeline actually reports (see gui.DEFAULT_MIN_MEASURED_S) get a number.
Also writes <video>.tracks.json.gz: every reported ID's centroid every 5th frame, so the
counts can still be matched by position if tracking (and therefore numbering) changes.

Usage:
    python validation/annotate.py --videos <folder> --out <folder> [--crf 26]
"""
import argparse
import colorsys
import csv
import gzip
import json
import os
import subprocess
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gui  # noqa: E402

CENTROID_EVERY = 5


def _color(tid):
    r, g, b = colorsys.hsv_to_rgb((tid * 0.61803) % 1.0, 0.9, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def track_video(path):
    """Runs the real pipeline once, recording where every track was on every frame."""
    seen = {}  # frame_idx -> [(tid, contour, centroid)]

    def on_frame(frame_idx, tracks):
        seen[frame_idx] = [(tid, t.contour.copy(), t.centroid) for tid, t in tracks.items()
                           if t.last_seen_frame == frame_idx]

    writer = csv.DictWriter(open(os.devnull, "w"), fieldnames=gui.CSV_COLUMNS)
    _, rows = gui.process_video(path, True, writer, on_frame=on_frame)
    return seen, {r["worm_id"] for r in rows}


def render(path, out_path, seen, reported, crf):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc = subprocess.Popen(
        [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "slow",
         "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, contour, _ in seen.get(frame_idx, []):
            if tid not in reported:
                continue
            color = _color(tid)
            cv2.drawContours(frame, [contour], -1, color, 1)
            x, y, _, _ = cv2.boundingRect(contour)
            label = str(tid)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(y - 3, th + 2)
            cv2.rectangle(frame, (x, ty - th - 2), (x + tw + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (x + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        cv2.LINE_AA)
        stamp = f"{frame_idx / fps:5.1f}s"
        cv2.rectangle(frame, (w - 70, h - 22), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, stamp, (w - 66, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        proc.stdin.write(frame.tobytes())
        frame_idx += 1
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def save_tracks(json_path, seen, reported):
    tracks = {}
    for frame_idx in sorted(seen):
        if frame_idx % CENTROID_EVERY:
            continue
        for tid, _, (cx, cy) in seen[frame_idx]:
            if tid in reported:
                tracks.setdefault(str(tid), []).append([frame_idx, round(cx), round(cy)])
    with gzip.open(json_path, "wt") as f:
        json.dump({"every_n_frames": CENTROID_EVERY, "tracks": tracks}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crf", type=int, default=26, help="x264 quality; higher = smaller file")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for path in gui.find_videos(args.videos):
        name = os.path.splitext(os.path.basename(path))[0]
        seen, reported = track_video(path)
        render(path, os.path.join(args.out, f"{name} numbered.mp4"), seen, reported, args.crf)
        save_tracks(os.path.join(args.out, f"{name}.tracks.json.gz"), seen, reported)
        print(f"{name}: {len(reported)} numbered worms")


if __name__ == "__main__":
    main()
