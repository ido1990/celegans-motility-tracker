"""Pure image-analysis functions: segmentation, contour classification, bend/thrash signal.

No OpenCV window/GUI code and no tracker state lives here.
"""
import numpy as np
import cv2
from scipy.signal import find_peaks

JUVENILE = "JUVENILE"
DEBRIS = "DEBRIS"
MERGED = "MERGED"
CANDIDATE_ADULT = "CANDIDATE_ADULT"

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def build_background_model(cap, n_samples=40, stop_event=None):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = max(total, 1)
    idxs = np.linspace(0, total - 1, min(n_samples, total)).astype(int)
    frames = []
    for i in idxs:
        if stop_event is not None and stop_event.is_set():
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not frames:
        raise ValueError("could not sample any frames to build background model")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def preprocess_frame(gray, bg_model, watermark_rows=32):
    signed = np.clip(bg_model.astype(np.int16) - gray.astype(np.int16), 0, 255).astype(np.uint8)
    signed[:watermark_rows, :] = 0
    _, mask = cv2.threshold(signed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    return mask


def extract_contours(mask, min_pixel_floor=15):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_pixel_floor]


def _otsu_area_cutoff(areas, fallback):
    """Splits a list of contour areas into a small (juvenile/debris) and large (adult)
    population via Otsu on the log-scaled areas — the same trick preprocess_frame uses on
    pixel intensities, applied to area instead so the cutoff isn't tied to one zoom level."""
    if len(areas) < 10:
        return fallback
    log_areas = np.log(np.array(areas, dtype=np.float64) + 1.0)
    spread = np.ptp(log_areas)
    if spread < 1e-6:
        return fallback
    scaled = ((log_areas - log_areas.min()) / spread * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(scaled.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cutoff_log = log_areas.min() + (otsu_val / 255.0) * spread
    return max(15, int(np.exp(cutoff_log) - 1.0))


def estimate_min_area(cap, bg_model, watermark_rows=32, n_samples=25, fallback=150, stop_event=None):
    """Auto-calibrates the adult/juvenile area cutoff from this video's own footage, so the
    same physical worm size doesn't need a different Min Area typed in by hand at every
    zoom/magnification level."""
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    idxs = np.linspace(0, total - 1, min(n_samples, total)).astype(int)
    areas = []
    for i in idxs:
        if stop_event is not None and stop_event.is_set():
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = preprocess_frame(gray, bg_model, watermark_rows)
        areas.extend(cv2.contourArea(c) for c in extract_contours(mask))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return _otsu_area_cutoff(areas, fallback)


def classify_contour(c, min_area, max_single_area):
    area = cv2.contourArea(c)
    (cx, cy), (w, h), _ = cv2.minAreaRect(c)
    elongation = max(w, h) / max(min(w, h), 1e-6)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1e-6)
    M = cv2.moments(c)
    if M["m00"] != 0:
        centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
    else:
        centroid = (cx, cy)

    metrics = {
        "area": area,
        "elongation": elongation,
        "solidity": solidity,
        "centroid": centroid,
    }

    if area < min_area:
        return JUVENILE, metrics
    if solidity > 0.75 and elongation < 2.0:
        return DEBRIS, metrics
    if area > max_single_area:
        return MERGED, metrics
    return CANDIDATE_ADULT, metrics


def bend_angle(contour):
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
    if len(hull) < 3:
        return None

    best = (-1.0, None, None)
    for i in range(len(hull)):
        d = np.sum((hull[i + 1:] - hull[i]) ** 2, axis=1)
        if d.size == 0:
            continue
        j_local = int(np.argmax(d))
        if d[j_local] > best[0]:
            best = (d[j_local], hull[i], hull[i + 1 + j_local])
    _, p1, p2 = best
    if p1 is None:
        return None

    mid_target = (p1 + p2) / 2.0
    pts = contour.reshape(-1, 2).astype(np.float64)
    dists = np.sum((pts - mid_target) ** 2, axis=1)
    mid = pts[int(np.argmin(dists))]

    v1 = p1 - mid
    v2 = p2 - mid
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def bend_signal(angle_history):
    filled = []
    last = 180.0
    for a in angle_history:
        if a is not None:
            last = a
        filled.append(last)
    return 180.0 - np.array(filled, dtype=np.float64)


def count_thrashes(signal, prominence, min_distance_frames=1):
    if len(signal) < 3:
        return 0
    peaks, _ = find_peaks(signal, prominence=prominence, distance=max(1, min_distance_frames))
    return len(peaks) // 2


MAX_GAP_FRAMES = 3
MIN_SEGMENT_FRAMES = 12


def measured_segments(angle_history, max_gap=MAX_GAP_FRAMES):
    """Splits a bend-angle history into runs of frames where the worm's shape was actually
    measured. Gaps (None: worm merged with another or briefly lost) of up to `max_gap`
    frames are bridged by linear interpolation; longer gaps end the run."""
    segments, cur, gap = [], [], 0
    for a in angle_history:
        if a is None:
            gap += 1
            if gap > max_gap and cur:
                segments.append(cur)
                cur = []
            continue
        if cur and gap:
            cur.extend(np.linspace(cur[-1], a, gap + 2)[1:-1])
        gap = 0
        cur.append(a)
    if cur:
        segments.append(cur)
    return segments


def count_thrashes_measured(angle_history, prominence, min_distance_frames=1,
                            max_gap=MAX_GAP_FRAMES, min_segment=MIN_SEGMENT_FRAMES):
    """Counts thrashes only over the frames where the body shape was measured, and returns
    (thrashes, measured_frames) so the rate can be normalized by that same time.

    Holding the last angle across unmeasured stretches (as bend_signal does) flattens the
    signal there, so those stretches contribute time but no thrashes — which drags the rate
    down for any worm that spends a lot of time touching others. Validated against hand
    counts in validation/ (per-video mean error 26 -> 7 thrashes/min)."""
    peaks = 0
    measured = 0
    for seg in measured_segments(angle_history, max_gap):
        if len(seg) < min_segment:
            continue
        sig = 180.0 - np.asarray(seg, dtype=np.float64)
        p, _ = find_peaks(sig, prominence=prominence, distance=max(1, min_distance_frames))
        peaks += len(p)
        measured += len(seg)
    return peaks // 2, measured


def _resample_closed(pts, m):
    loop = np.vstack([pts, pts[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])
    if s[-1] <= 0:
        return None
    t = np.linspace(0, s[-1], m, endpoint=False)
    return np.column_stack([np.interp(t, s, loop[:, 0]), np.interp(t, s, loop[:, 1])])


def _resample_open(pts, k):
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if s[-1] <= 0:
        return None
    t = np.linspace(0, s[-1], k)
    return np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])


def midline(contour, n_outline=80, n_points=21, tip_span=4):
    """Centerline of a thin worm outline, as (points, length, side_ratio) or None.

    The head and tail are the two sharpest tips of the outline (at least a quarter of
    the outline apart); the two sides between them are resampled and averaged pointwise.
    side_ratio (shorter side / longer side) is near 1 for a clean single-worm outline and
    drops when the outline is really two touching worms or a partial shape."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return None
    c = _resample_closed(pts, n_outline)
    if c is None:
        return None
    a = np.roll(c, tip_span, axis=0) - c
    b = np.roll(c, -tip_span, axis=0) - c
    sharp = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    i1 = int(np.argmax(sharp))
    circ = np.abs((np.arange(n_outline) - i1 + n_outline // 2) % n_outline - n_outline // 2)
    i2 = int(np.argmax(np.where(circ >= n_outline // 4, sharp, -np.inf)))
    i1, i2 = min(i1, i2), max(i1, i2)
    side1 = c[i1:i2 + 1]
    side2 = np.vstack([c[i2:], c[:i1 + 1]])[::-1]
    len1 = np.linalg.norm(np.diff(side1, axis=0), axis=1).sum()
    len2 = np.linalg.norm(np.diff(side2, axis=0), axis=1).sum()
    r1, r2 = _resample_open(side1, n_points), _resample_open(side2, n_points)
    if r1 is None or r2 is None:
        return None
    ml = (r1 + r2) / 2.0
    length = float(np.linalg.norm(np.diff(ml, axis=0), axis=1).sum())
    return ml, length, float(min(len1, len2) / max(len1, len2))


def signed_bend(ml):
    """Signed bend (deg) between the head half and tail half of a centerline: positive
    one way, negative the other, so a full thrash swings it through + and -."""
    v = np.diff(ml, axis=0)
    ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    n = len(ang)
    return float(np.degrees(ang[n // 2:].mean() - ang[:n // 2].mean()))


def _orient(ml, prev):
    """Keeps head/tail order consistent with the previous frame's centerline."""
    if prev is None:
        return ml
    same = np.linalg.norm(ml[0] - prev[0]) + np.linalg.norm(ml[-1] - prev[-1])
    flip = np.linalg.norm(ml[0] - prev[-1]) + np.linalg.norm(ml[-1] - prev[0])
    return ml[::-1] if flip < same else ml


CL_LENGTH_TOL = 0.30     # centerline length must be within +-30% of the worm's median
CL_SIDE_MIN = 0.60       # both sides of the outline at least 60% as long as each other
CL_MAX_JUMP_DEG = 90.0   # a real worm can't change its bend more than this in one frame
CL_PROMINENCE_DEG = 20.0
CL_MIN_SEGMENT = 25


def centerline_bend_segments(midline_history, max_gap=MAX_GAP_FRAMES):
    """Signed-bend signal over the frames whose centerline passes quality checks, split
    into runs like measured_segments(). Frames that fail count as gaps."""
    lengths = [m[1] for m in midline_history if m is not None]
    if not lengths:
        return []
    median_len = float(np.median(lengths))
    segments, cur, prev, gap = [], [], None, 0
    for m in midline_history:
        ok = (m is not None and median_len > 0
              and abs(m[1] / median_len - 1.0) <= CL_LENGTH_TOL and m[2] >= CL_SIDE_MIN)
        if ok:
            ml = _orient(m[0], prev)
            bend = signed_bend(ml)
            if cur and abs(bend - cur[-1]) > CL_MAX_JUMP_DEG * (gap + 1):
                ok = False
        if not ok:
            gap += 1
            if gap > max_gap and cur:
                segments.append(cur)
                cur, prev = [], None
            continue
        if cur and gap:
            cur.extend(np.linspace(cur[-1], bend, gap + 2)[1:-1])
        gap = 0
        prev = ml
        cur.append(bend)
    if cur:
        segments.append(cur)
    return segments


def count_thrashes_centerline(midline_history, prominence=CL_PROMINENCE_DEG, min_distance_frames=3,
                              min_segment=CL_MIN_SEGMENT):
    """(thrashes, measured_frames) from the signed centerline bend: one thrash per full
    swing, i.e. the average of the number of + peaks and - peaks. Experimental: reported
    alongside count_thrashes_measured until per-worm hand counts show which is better."""
    peaks = 0.0
    measured = 0
    for seg in centerline_bend_segments(midline_history):
        if len(seg) < min_segment:
            continue
        x = np.asarray(seg, dtype=np.float64)
        hi, _ = find_peaks(x, prominence=prominence, distance=min_distance_frames)
        lo, _ = find_peaks(-x, prominence=prominence, distance=min_distance_frames)
        peaks += (len(hi) + len(lo)) / 2.0
        measured += len(seg)
    return int(round(peaks)), measured


if __name__ == "__main__":
    canvas = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(canvas, (100, 100), (60, 15), 0, 0, 360, 255, -1)
    straight_c = max(cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                      key=cv2.contourArea)
    straight_angle = bend_angle(straight_c)
    assert straight_angle is not None and straight_angle > 120, f"expected near-straight angle, got {straight_angle}"

    canvas2 = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(canvas2, (100, 100), (60, 60), 0, 30, 300, 255, 12)
    curled_c = max(cv2.findContours(canvas2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                   key=cv2.contourArea)
    curled_angle = bend_angle(curled_c)
    assert curled_angle is not None and curled_angle < straight_angle, (
        f"expected curled ({curled_angle}) < straight ({straight_angle})"
    )

    round_canvas = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(round_canvas, (50, 50), 30, 255, -1)
    round_c = max(cv2.findContours(round_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                  key=cv2.contourArea)
    label, _ = classify_contour(round_c, min_area=50, max_single_area=1e9)
    assert label == DEBRIS, f"expected round blob to classify as DEBRIS, got {label}"

    elongated_canvas = np.zeros((100, 300), dtype=np.uint8)
    cv2.ellipse(elongated_canvas, (150, 50), (120, 10), 0, 0, 360, 255, -1)
    elong_c = max(cv2.findContours(elongated_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                  key=cv2.contourArea)
    label2, _ = classify_contour(elong_c, min_area=50, max_single_area=1e9)
    assert label2 == CANDIDATE_ADULT, f"expected elongated shape to classify as CANDIDATE_ADULT, got {label2}"

    rng = np.random.default_rng(0)
    small_cluster = list(rng.normal(30, 3, 30))  # juvenile/debris-sized, e.g. one zoom level
    large_cluster = list(rng.normal(800, 60, 30))  # adult-sized
    cutoff = _otsu_area_cutoff(small_cluster + large_cluster, fallback=999)
    assert cutoff < min(large_cluster), f"cutoff {cutoff} must stay below the adult cluster"
    assert cutoff > max(small_cluster) * 0.5, f"cutoff {cutoff} should be in the juvenile cluster's ballpark, not near zero"
    assert all(a >= cutoff for a in large_cluster), "every adult-sized sample should clear the cutoff"
    assert _otsu_area_cutoff([50, 51, 52], fallback=999) == 999, "expected fallback on too-few samples"
    assert _otsu_area_cutoff([100] * 20, fallback=999) == 999, "expected fallback on zero-spread data"

    segs = measured_segments([10, None, 30, None, None, None, None, 50, 60])
    assert segs == [[10, 20.0, 30], [50, 60]], f"unexpected segments {segs}"

    # A worm bending at 1 Hz for 10 s at 25 fps, unmeasured for the middle 10 s: the rate
    # must come from the 20 measured seconds, not the 30 s it was on screen.
    t = np.arange(250) / 25.0
    wave = list(180 - 60 * np.abs(np.sin(2 * np.pi * t)))  # left C and right C each dip the angle
    hist = wave + [None] * 250 + wave
    thrashes, measured = count_thrashes_measured(hist, prominence=12, min_distance_frames=2)
    assert measured == 500, f"expected 500 measured frames, got {measured}"
    rate = thrashes / (measured / 25.0) * 60
    assert 55 <= rate <= 62, f"expected ~60 thrashes/min from measured frames only, got {rate}"

    # Centerline: a synthetic worm bending left and right at 1 Hz for 10 s (25 fps) should
    # read as ~60 thrashes/min, and a straight worm's centerline as ~0 deg of bend.
    def _worm(bend_deg, n=60, length=80, width=4):
        k = np.radians(bend_deg) / length  # constant curvature
        s = np.linspace(-length / 2, length / 2, n)
        theta = k * s
        x = np.cumsum(np.cos(theta)); y = np.cumsum(np.sin(theta))
        pts = np.column_stack([x, y]) + 100
        canvas = np.zeros((300, 300), dtype=np.uint8)
        cv2.polylines(canvas, [pts.astype(np.int32).reshape(-1, 1, 2)], False, 255, width)
        return max(cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
                   key=cv2.contourArea)

    straight = midline(_worm(0.1))
    assert straight is not None and abs(signed_bend(straight[0])) < 15, "straight worm should read ~0 bend"
    curved = midline(_worm(120))
    assert curved is not None and abs(signed_bend(curved[0])) > 40, "C-shaped worm should read a large bend"

    t = np.arange(250) / 25.0
    hist = [midline(_worm(100 * np.sin(2 * np.pi * ti))) for ti in t]
    thr, meas = count_thrashes_centerline(hist)
    rate = thr / (meas / 25.0) * 60
    assert 50 <= rate <= 70, f"expected ~60 thrashes/min from centerline, got {rate}"

    print("analyzer.py self-check: PASS")
