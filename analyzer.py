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


def build_background_model(cap, n_samples=40):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = max(total, 1)
    idxs = np.linspace(0, total - 1, min(n_samples, total)).astype(int)
    frames = []
    for i in idxs:
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

    print("analyzer.py self-check: PASS")
