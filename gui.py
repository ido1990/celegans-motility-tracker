"""Main execution loop: batch folder processing, OpenCV window, trackbars, CSV export."""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

import analyzer
import tracker as tracker_mod
from tracker import CentroidTracker, ACTIVE, DEAD

WINDOW_NAME = "Motility Tracker"
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")
CSV_COLUMNS = [
    "source_video", "worm_id", "final_state", "frame_entry", "frame_exit",
    "visible_frames", "total_thrashes", "thrash_rate_per_min", "mean_area",
    "mean_bend_amplitude_deg",
]

DEFAULT_MIN_AREA = 150
DEFAULT_SENSITIVITY = 15
DEFAULT_DELAY_MS = 33
DEFAULT_HEALTHY_THRESHOLD = 20  # thrashes/min; below this an ACTIVE worm reads as DISEASED


def _nop(_):
    pass


def setup_window(overrides=None):
    o = overrides or {}
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Min Area", WINDOW_NAME, int(o.get("min_area", DEFAULT_MIN_AREA)), 3000, _nop)
    cv2.createTrackbar("Sensitivity", WINDOW_NAME, int(o.get("sensitivity", DEFAULT_SENSITIVITY)), 60, _nop)
    cv2.createTrackbar("Playback Speed (ms)", WINDOW_NAME, int(o.get("delay_ms", DEFAULT_DELAY_MS)), 200, _nop)
    cv2.createTrackbar("Max Track Dist", WINDOW_NAME, int(o.get("max_distance", tracker_mod.MAX_DISTANCE)), 150, _nop)
    cv2.createTrackbar("Track Patience (fr)", WINDOW_NAME,
                        int(o.get("max_disappeared", tracker_mod.MAX_DISAPPEARED)), 120, _nop)
    cv2.createTrackbar("Healthy Rate", WINDOW_NAME, int(o.get("healthy_threshold", DEFAULT_HEALTHY_THRESHOLD)),
                        100, _nop)
    cv2.createTrackbar("Dead Pos Delta", WINDOW_NAME, int(o.get("dead_pos_delta", tracker_mod.DEAD_POSITION_DELTA)),
                        60, _nop)
    cv2.createTrackbar("Dead Bend Delta", WINDOW_NAME, int(o.get("dead_bend_delta", tracker_mod.DEAD_BEND_DELTA)),
                        90, _nop)
    cv2.createTrackbar("Dead Window (fr)", WINDOW_NAME, int(o.get("dead_window", tracker_mod.DEAD_WINDOW_FRAMES)),
                        300, _nop)


def read_controls(dry_run, overrides=None):
    """Segmentation/playback controls, read fresh every frame.

    In live mode these come from the trackbars (draggable mid-run); in dry-run mode (no
    window/trackbars exist) they come straight from `overrides`, which is also what seeds
    the trackbars' initial position in live mode — see setup_window().
    """
    o = overrides or {}
    if dry_run:
        return o.get("min_area", DEFAULT_MIN_AREA), o.get("sensitivity", DEFAULT_SENSITIVITY), \
            o.get("delay_ms", DEFAULT_DELAY_MS)
    min_area = cv2.getTrackbarPos("Min Area", WINDOW_NAME)
    sensitivity = max(1, cv2.getTrackbarPos("Sensitivity", WINDOW_NAME))
    delay_ms = max(1, cv2.getTrackbarPos("Playback Speed (ms)", WINDOW_NAME))
    return min_area, sensitivity, delay_ms


def read_tracker_controls(dry_run, overrides=None):
    """Tracker/classification tuning knobs, read fresh every frame. See read_controls()."""
    o = overrides or {}
    if dry_run:
        return (o.get("max_distance", tracker_mod.MAX_DISTANCE), o.get("max_disappeared", tracker_mod.MAX_DISAPPEARED),
                o.get("healthy_threshold", DEFAULT_HEALTHY_THRESHOLD), o.get("dead_pos_delta", tracker_mod.DEAD_POSITION_DELTA),
                o.get("dead_bend_delta", tracker_mod.DEAD_BEND_DELTA), o.get("dead_window", tracker_mod.DEAD_WINDOW_FRAMES))
    max_distance = max(1, cv2.getTrackbarPos("Max Track Dist", WINDOW_NAME))
    max_disappeared = max(1, cv2.getTrackbarPos("Track Patience (fr)", WINDOW_NAME))
    healthy_threshold = cv2.getTrackbarPos("Healthy Rate", WINDOW_NAME)
    dead_pos_delta = cv2.getTrackbarPos("Dead Pos Delta", WINDOW_NAME)
    dead_bend_delta = cv2.getTrackbarPos("Dead Bend Delta", WINDOW_NAME)
    dead_window = max(2, cv2.getTrackbarPos("Dead Window (fr)", WINDOW_NAME))
    return max_distance, max_disappeared, healthy_threshold, dead_pos_delta, dead_bend_delta, dead_window


def apply_tracker_controls(trk, controls):
    max_distance, max_disappeared, _healthy_threshold, dead_pos_delta, dead_bend_delta, dead_window = controls
    trk.max_distance = max_distance
    trk.max_disappeared = max_disappeared
    trk.dead_position_delta = dead_pos_delta
    trk.dead_bend_delta = dead_bend_delta
    trk.dead_window_frames = dead_window


def live_thrash_rate(track, fps, prominence):
    visible = track.last_seen_frame - track.frame_entry + 1
    if visible <= 0:
        return 0.0
    sig = analyzer.bend_signal(track.bend_angle_history)
    thrashes = analyzer.count_thrashes(sig, prominence, min_distance_frames=max(1, int(fps // 10)))
    return thrashes / (visible / fps) * 60.0


def status_label(track, fps, prominence, healthy_threshold):
    if track.state == DEAD:
        return "DEAD"
    rate = live_thrash_rate(track, fps, prominence)
    return "HEALTHY" if rate >= healthy_threshold else "DISEASED"


def draw_track(vis, track, fps, prominence, healthy_threshold):
    x, y, w, h = cv2.boundingRect(track.contour)
    color = (0, 255, 0) if track.state == ACTIVE else (0, 0, 255)
    cv2.drawContours(vis, [track.contour], -1, color, 2)
    label = status_label(track, fps, prominence, healthy_threshold)
    cv2.putText(vis, f"ID:{track.id} {label}", (x, max(y - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    if track.state == ACTIVE:
        rate = live_thrash_rate(track, fps, prominence)
        cv2.putText(vis, f"{rate:.1f}/min", (x, y + h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_juveniles(vis, contours):
    for c in contours:
        cv2.drawContours(vis, [c], -1, (160, 160, 160), 1)


def draw_summary(vis, active_count, avg_rate):
    lines = [f"Active Adults: {active_count}", f"Avg Thrashes/Min: {avg_rate:.1f}"]
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1, cv2.LINE_AA)


def finalize_row(track, source_video, fps, prominence, healthy_threshold):
    visible = track.last_seen_frame - track.frame_entry + 1
    sig = analyzer.bend_signal(track.bend_angle_history)
    thrashes = analyzer.count_thrashes(sig, prominence, min_distance_frames=max(1, int(fps // 10)))
    rate = thrashes / (visible / fps) * 60.0 if visible > 0 else 0.0
    if track.state == DEAD:
        final_state = "DEAD"
    else:
        final_state = "HEALTHY" if rate >= healthy_threshold else "DISEASED"
    return {
        "source_video": source_video,
        "worm_id": track.id,
        "final_state": final_state,
        "frame_entry": track.frame_entry,
        "frame_exit": track.last_seen_frame,
        "visible_frames": visible,
        "total_thrashes": thrashes,
        "thrash_rate_per_min": round(rate, 2),
        "mean_area": round(float(np.mean(track.area_history)), 1) if track.area_history else 0.0,
        "mean_bend_amplitude_deg": round(float(np.mean(sig)), 2) if len(sig) else 0.0,
    }


def process_video(path, dry_run, writer, overrides=None, stop_event=None):
    """Runs the pipeline on one video, writes its rows to `writer`, and returns
    (aborted, rows) where rows is the same per-worm dicts that were written.

    `overrides` (optional dict) lets a caller (the CLI's trackbars are the other source)
    pin parameters up front, e.g. {"sensitivity": 20, "min_area": 200}. Any key left out
    keeps its usual default/auto behavior. Min Area is special: if not overridden, it's
    auto-calibrated per video from that video's own footage (see analyzer.estimate_min_area).

    `stop_event` (optional threading.Event) is checked once per frame so a caller running
    this on a background thread (the launcher's Stop button) can interrupt a run early —
    same as pressing q/Esc in the live preview window, partial results are still written.
    """
    overrides = overrides or {}
    if stop_event is not None and stop_event.is_set():
        return True, []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  could not open {path}, skipping")
        return False, []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Calibration (background model + auto Min Area) alone can take tens of seconds on slow-
    # seeking video files, so it checks stop_event too — otherwise Stop wouldn't respond until
    # the whole calibration phase finished, even before the per-frame loop below.
    bg_model = analyzer.build_background_model(cap, stop_event=stop_event)
    if stop_event is not None and stop_event.is_set():
        return True, []
    if overrides.get("min_area") is not None:
        calibrated_min_area = int(overrides["min_area"])
    else:
        calibrated_min_area = analyzer.estimate_min_area(
            cap, bg_model, fallback=DEFAULT_MIN_AREA, stop_event=stop_event)
        print(f"  auto-calibrated Min Area for {os.path.basename(path)}: {calibrated_min_area}px")
    if stop_event is not None and stop_event.is_set():
        return True, []
    if not dry_run:
        cv2.setTrackbarPos("Min Area", WINDOW_NAME, min(calibrated_min_area, 3000))
    video_overrides = {**overrides, "min_area": calibrated_min_area}
    trk = CentroidTracker()
    adult_areas = []
    frame_idx = 0
    aborted = False
    source_video = os.path.basename(path)
    sensitivity = DEFAULT_SENSITIVITY
    healthy_threshold = DEFAULT_HEALTHY_THRESHOLD

    while True:
        if stop_event is not None and stop_event.is_set():
            aborted = True
            break

        ok, frame = cap.read()
        if not ok:
            break

        min_area, sensitivity, delay_ms = read_controls(dry_run, video_overrides)
        tracker_controls = read_tracker_controls(dry_run, video_overrides)
        healthy_threshold = tracker_controls[2]
        apply_tracker_controls(trk, tracker_controls)
        max_single_area = np.median(adult_areas[-500:]) * 2.75 if adult_areas else 1500.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = analyzer.preprocess_frame(gray, bg_model)
        contours = analyzer.extract_contours(mask)

        detections = []
        juvenile_contours = []
        for c in contours:
            label, metrics = analyzer.classify_contour(c, min_area, max_single_area)
            if label == analyzer.JUVENILE:
                juvenile_contours.append(c)
                continue
            if label == analyzer.DEBRIS:
                continue
            is_merged = label == analyzer.MERGED
            angle = None if is_merged else analyzer.bend_angle(c)
            detections.append({
                "centroid": metrics["centroid"], "contour": c, "area": metrics["area"],
                "bend_angle": angle, "is_merged": is_merged,
            })
            if not is_merged:
                adult_areas.append(metrics["area"])

        tracks = trk.update(detections, frame_idx)

        if not dry_run:
            vis = frame.copy()
            draw_juveniles(vis, juvenile_contours)
            active_rates = []
            for t in tracks.values():
                draw_track(vis, t, fps, sensitivity, healthy_threshold)
                if t.state == ACTIVE:
                    active_rates.append(live_thrash_rate(t, fps, sensitivity))
            avg_rate = float(np.mean(active_rates)) if active_rates else 0.0
            draw_summary(vis, len(active_rates), avg_rate)
            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                aborted = True
                break

        frame_idx += 1

    cap.release()

    rows = [finalize_row(t, source_video, fps, sensitivity, healthy_threshold)
            for t in trk.finalize_all()]
    for row in rows:
        writer.writerow(row)

    return aborted, rows


def find_videos(folder):
    return sorted(
        f for f in glob.glob(os.path.join(folder, "*"))
        if f.lower().endswith(VIDEO_EXTS)
    )


def run_batch(folder, output_path, dry_run, progress=None, params=None, stop_event=None):
    """Processes every video in `folder`, writes `output_path` CSV, and returns the
    combined list of per-worm row dicts (across all videos) for a caller (CLI print,
    or a GUI results view) to use without having to re-read the CSV.

    `params` (optional dict) sets initial/fixed parameter values up front (e.g. from a
    launcher UI) instead of relying purely on live trackbar dragging — see process_video().

    `stop_event` (optional threading.Event) is checked between videos and (inside
    process_video) once per frame, so a caller can interrupt mid-batch or mid-video and
    still get back the rows collected so far.
    """
    videos = find_videos(folder)
    if not videos:
        if progress:
            progress(f"No video files found in {folder}")
        return []

    if not dry_run:
        setup_window(params)

    all_rows = []
    try:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for i, path in enumerate(videos, 1):
                if stop_event is not None and stop_event.is_set():
                    if progress:
                        progress("Aborted by user.")
                    break
                if progress:
                    progress(f"Processing {i}/{len(videos)}: {os.path.basename(path)}")
                aborted, rows = process_video(path, dry_run, writer, params, stop_event)
                all_rows.extend(rows)
                f.flush()
                if aborted:
                    if progress:
                        progress("Aborted by user.")
                    break
    finally:
        if not dry_run:
            cv2.destroyAllWindows()

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Batch C. elegans motility/thrash analysis")
    parser.add_argument("--folder", default="assets", help="folder of videos to process")
    parser.add_argument("--output", default="worm_motility_results.csv")
    parser.add_argument("--dry-run", action="store_true", help="headless, no cv2.imshow window")
    args = parser.parse_args()

    rows = run_batch(args.folder, args.output, args.dry_run, progress=print)
    if rows:
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
