"""
club_estimation.py — Estimate golf club position from video.

Primary path: YOLO-World open-vocabulary object detector finds the club's
2D bounding box per frame; the box endpoint farthest from the wrists is
treated as the club head, and the head's 3D position is recovered by
scaling the 2D grip→head displacement with the wrist-span ratio between
image pixels and MediaPipe world-landmark meters.

Fallback path: if no video is supplied or YOLO/MediaPipe are unavailable,
a forearm-projection biomechanical heuristic is used.
"""

from pathlib import Path

import numpy as np


# ── MediaPipe landmark indices ──────────────────────────────────────────────
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW,    RIGHT_ELBOW    = 13, 14
LEFT_WRIST,    RIGHT_WRIST    = 15, 16
LEFT_HIP,      RIGHT_HIP      = 23, 24


# ── Optional deps ──────────────────────────────────────────────────────────
try:
    import cv2  # noqa: F401
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from ultralytics import YOLOWorld
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


# ── Model paths ────────────────────────────────────────────────────────────
_PROJ_ROOT     = Path(__file__).parent.parent
_YOLO_MODEL_FP = _PROJ_ROOT / "models" / "yolov8s-world.pt"
_POSE_MODEL_FP = _PROJ_ROOT / "pose_landmarker.task"

_YOLO_PROMPTS  = ["golf club", "club", "stick"]
_YOLO_CONF     = 0.03   # YOLOWorld scores skew low; keep liberal and rely on geometry filter
_YOLO_IOU      = 0.45
_YOLO_IMGSZ    = 640

# Lazy global model cache so a Streamlit re-run doesn't reload weights
_yolo_model = None


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLOWorld(str(_YOLO_MODEL_FP))
        _yolo_model.set_classes(_YOLO_PROMPTS)
    return _yolo_model


# ── Small helpers ──────────────────────────────────────────────────────────
def _get_landmark_pos(landmarks, idx):
    """Extract (x, y, z) from a world-landmarks list, or None if missing."""
    for lm in landmarks:
        if lm["idx"] == idx and lm["x"] is not None:
            return np.array([lm["x"], lm["y"], lm["z"]])
    return None


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.array([0.0, 1.0, 0.0])


def _round_point(p):
    return {"x": round(float(p[0]), 5),
            "y": round(float(p[1]), 5),
            "z": round(float(p[2]), 5)}


def _missing_entry():
    return {"grip": None, "shaft_end": None, "has_club": False}


# ── Public API ─────────────────────────────────────────────────────────────
def estimate_club_positions(
    frames_3d: list,
    phase_ranges: dict = None,
    video_path: str = None,
    use_ai_detection: bool = True,
) -> list:
    """
    Estimate golf club grip and shaft-end positions per frame.

    Args:
        frames_3d:        Output of extract_full_3d_landmarks() — 3D world
                          landmarks in meters, hip-centered.
        phase_ranges:     Optional phase dict (unused; kept for API stability).
        video_path:       Path to source video. Required for AI detection.
        use_ai_detection: When True and YOLO+MediaPipe+video are all available,
                          uses YOLO-World to localize the club head per frame
                          and back-projects into 3D. Falls back to the
                          biomechanical heuristic otherwise.

    Returns:
        List of dicts per frame:
        [{grip: {x,y,z}, shaft_end: {x,y,z}, has_club: bool}, ...]
    """
    if (use_ai_detection
            and video_path
            and Path(video_path).exists()
            and _CV2_AVAILABLE and _YOLO_AVAILABLE and _MP_AVAILABLE
            and _YOLO_MODEL_FP.exists()
            and _POSE_MODEL_FP.exists()):
        try:
            return _estimate_with_ai(frames_3d, video_path)
        except Exception as exc:  # noqa: BLE001
            # Detection should never break analysis — fall back loudly to logs.
            import sys
            print(f"[club_estimation] AI detection failed: {exc!r}. "
                  "Falling back to heuristic.", file=sys.stderr)

    return _estimate_with_heuristic(frames_3d)


# ── AI detection path ──────────────────────────────────────────────────────
def _extract_image_pose(video_path: str):
    """
    Run MediaPipe Pose once over the video and return per-frame 2D pixel
    coords for wrists & shoulders, plus the frame size.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_POSE_MODEL_FP)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose_px = []  # list of dicts per frame
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            ts = int(frame_idx * 1000 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_image, ts)

            entry = {"lw": None, "rw": None, "ls": None, "rs": None}
            if res.pose_landmarks:
                lms = res.pose_landmarks[0]
                def _xy(i):
                    lm = lms[i]
                    return np.array([lm.x * width, lm.y * height])
                entry["lw"] = _xy(LEFT_WRIST)
                entry["rw"] = _xy(RIGHT_WRIST)
                entry["ls"] = _xy(LEFT_SHOULDER)
                entry["rs"] = _xy(RIGHT_SHOULDER)
            pose_px.append(entry)
            frame_idx += 1
    cap.release()
    return pose_px, (width, height)


def _detect_club_boxes(video_path: str):
    """Run YOLO-World per frame, return [(bbox, conf) | None] per frame."""
    import cv2

    model = _get_yolo_model()
    cap = cv2.VideoCapture(video_path)
    boxes = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        res = model.predict(
            frame,
            conf=_YOLO_CONF,
            iou=_YOLO_IOU,
            imgsz=_YOLO_IMGSZ,
            verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            boxes.append(None)
            continue
        confs = res.boxes.conf.cpu().numpy()
        xyxy  = res.boxes.xyxy.cpu().numpy()
        boxes.append((xyxy, confs))
    cap.release()
    return boxes


def _pick_best_box(detections, wrist_mid_px):
    """
    Choose the detection most likely to be the club:
    score = confidence * proximity_to_wrist (so we ignore unrelated objects).
    """
    if detections is None or wrist_mid_px is None:
        return None
    xyxy, confs = detections
    best, best_score = None, -1.0
    for box, c in zip(xyxy, confs):
        cx = 0.5 * (box[0] + box[2])
        cy = 0.5 * (box[1] + box[3])
        center = np.array([cx, cy])
        dist = np.linalg.norm(center - wrist_mid_px)
        # Soft proximity prior: clubs are usually within ~3× wrist span of grip
        proximity = 1.0 / (1.0 + dist / 200.0)
        score = float(c) * proximity
        if score > best_score:
            best_score = score
            best = box
    return best


def _bbox_head_endpoint(bbox, wrist_mid_px):
    """
    Decide which end of the bbox is the club head: the corner farthest
    from the wrist midpoint (clubs are elongated, so the diagonal that
    runs along the shaft has the head at its far end).
    """
    x1, y1, x2, y2 = bbox
    corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    dists = np.linalg.norm(corners - wrist_mid_px, axis=1)
    return corners[int(np.argmax(dists))]


def _interp_missing(series):
    """
    Linearly interpolate None entries in a sequence of np.ndarray (or None).
    Pads endpoints by repeating the nearest valid value.
    """
    n = len(series)
    valid = [i for i, v in enumerate(series) if v is not None]
    if not valid:
        return series

    out = list(series)
    # Pad left
    for i in range(valid[0]):
        out[i] = series[valid[0]].copy()
    # Pad right
    for i in range(valid[-1] + 1, n):
        out[i] = series[valid[-1]].copy()
    # Interpolate gaps
    for a, b in zip(valid, valid[1:]):
        if b - a <= 1:
            continue
        va, vb = series[a], series[b]
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            out[i] = (1 - t) * va + t * vb
    return out


def _estimate_with_ai(frames_3d: list, video_path: str) -> list:
    """AI path: YOLO-World + MediaPipe + 2D→3D back-projection."""

    pose_px, _frame_size = _extract_image_pose(video_path)
    yolo_boxes           = _detect_club_boxes(video_path)
    n = min(len(frames_3d), len(pose_px), len(yolo_boxes))

    # First pass: per frame, derive grip_3d (wrist midpoint, world) and
    # raw 2D head endpoint (pixels).
    grip_3d_series = []
    head_px_series = []
    px_per_meter_series = []
    forearm_dir_series = []

    for i in range(n):
        world_lms = frames_3d[i].get("landmarks", [])
        lw_w = _get_landmark_pos(world_lms, LEFT_WRIST)
        rw_w = _get_landmark_pos(world_lms, RIGHT_WRIST)
        le_w = _get_landmark_pos(world_lms, LEFT_ELBOW)
        re_w = _get_landmark_pos(world_lms, RIGHT_ELBOW)

        if lw_w is None or rw_w is None:
            grip_3d_series.append(None)
            head_px_series.append(None)
            px_per_meter_series.append(None)
            forearm_dir_series.append(None)
            continue

        grip_3d = (lw_w + rw_w) / 2.0

        # Forearm direction (used to infer Z and as fallback shaft dir)
        if le_w is not None and re_w is not None:
            forearm = _normalize(_normalize(lw_w - le_w) + _normalize(rw_w - re_w))
        else:
            forearm = np.array([0.0, 1.0, 0.0])

        # Pixel scale from wrist span
        lw_px = pose_px[i]["lw"]
        rw_px = pose_px[i]["rw"]
        scale = None
        wrist_mid_px = None
        if lw_px is not None and rw_px is not None:
            wrist_mid_px = (lw_px + rw_px) / 2.0
            wrist_span_px    = np.linalg.norm(lw_px - rw_px)
            wrist_span_meter = np.linalg.norm(lw_w - rw_w)
            # Fallback to shoulder span if wrists nearly coincide (address pose)
            if wrist_span_px < 8.0 or wrist_span_meter < 0.02:
                ls_px = pose_px[i]["ls"]; rs_px = pose_px[i]["rs"]
                ls_w  = _get_landmark_pos(world_lms, LEFT_SHOULDER)
                rs_w  = _get_landmark_pos(world_lms, RIGHT_SHOULDER)
                if (ls_px is not None and rs_px is not None
                        and ls_w is not None and rs_w is not None):
                    shoulder_span_px    = np.linalg.norm(ls_px - rs_px)
                    shoulder_span_meter = np.linalg.norm(ls_w - rs_w)
                    if shoulder_span_px > 1.0 and shoulder_span_meter > 1e-3:
                        scale = shoulder_span_px / shoulder_span_meter
            else:
                scale = wrist_span_px / wrist_span_meter

        # AI detection → club head pixel
        head_px = None
        if wrist_mid_px is not None and yolo_boxes[i] is not None:
            bbox = _pick_best_box(yolo_boxes[i], wrist_mid_px)
            if bbox is not None:
                head_px = _bbox_head_endpoint(bbox, wrist_mid_px)

        grip_3d_series.append(grip_3d)
        head_px_series.append(head_px)
        px_per_meter_series.append(scale)
        forearm_dir_series.append(forearm)

    # Interpolate missing pixel/scale entries so smoothing has continuity.
    head_px_series      = _interp_missing(head_px_series)
    px_per_meter_series = _interp_missing(
        [None if s is None else np.array([s]) for s in px_per_meter_series]
    )

    # Second pass: smooth head pixels across frames (3-tap moving average),
    # then back-project to 3D and emit.
    smoothed_head = _smooth_points(head_px_series, window=3)

    # Estimate a stable club length (median of valid frames) for fallback use.
    body_scale = _estimate_body_scale(frames_3d)
    fallback_len = body_scale * 1.8 if body_scale > 0 else 1.1

    club_lengths = []
    for i in range(n):
        if (grip_3d_series[i] is None or smoothed_head[i] is None
                or px_per_meter_series[i] is None
                or pose_px[i]["lw"] is None or pose_px[i]["rw"] is None):
            continue
        wrist_mid_px = (pose_px[i]["lw"] + pose_px[i]["rw"]) / 2.0
        scale_px_per_m = float(px_per_meter_series[i][0])
        head_meters_xy = (smoothed_head[i] - wrist_mid_px) / scale_px_per_m
        # MediaPipe image-Y and world-Y both point down → no flip.
        club_lengths.append(np.linalg.norm(head_meters_xy))

    if club_lengths:
        # Clamp to a plausible golf-club range (0.8 m – 1.4 m for arms+shaft)
        median_len = float(np.median(club_lengths))
        median_len = float(np.clip(median_len, 0.8, 1.6))
    else:
        median_len = fallback_len

    # Third pass: produce output entries.
    out = []
    for i in range(n):
        grip_3d = grip_3d_series[i]
        if grip_3d is None:
            out.append(_missing_entry())
            continue

        # Try AI back-projection.
        if (smoothed_head[i] is not None
                and px_per_meter_series[i] is not None
                and pose_px[i]["lw"] is not None
                and pose_px[i]["rw"] is not None):
            wrist_mid_px = (pose_px[i]["lw"] + pose_px[i]["rw"]) / 2.0
            scale_px_per_m = float(px_per_meter_series[i][0])

            dxy_meters = (smoothed_head[i] - wrist_mid_px) / scale_px_per_m
            # Constrain the planar magnitude to the median club length so the
            # shaft stays a consistent length even when the head box jitters.
            planar_mag = np.linalg.norm(dxy_meters)
            if planar_mag > 1e-6:
                dxy_meters = dxy_meters * (median_len / planar_mag)

            # Borrow Z from the forearm so the club extends naturally in depth
            # but keep its sign consistent and amplitude bounded.
            forearm = forearm_dir_series[i]
            dz = float(forearm[2]) * median_len * 0.35 if forearm is not None else 0.0

            offset = np.array([float(dxy_meters[0]),
                               float(dxy_meters[1]),
                               dz])
            # Renormalize to keep total length = median_len
            mag = np.linalg.norm(offset)
            if mag > 1e-6:
                offset = offset * (median_len / mag)
            shaft_end = grip_3d + offset

            out.append({
                "grip":      _round_point(grip_3d),
                "shaft_end": _round_point(shaft_end),
                "has_club":  True,
            })
            continue

        # Fallback for this frame: forearm projection.
        forearm = forearm_dir_series[i]
        if forearm is None:
            forearm = np.array([0.0, 1.0, 0.0])
        down = np.array([0.0, 1.0, 0.0])
        shaft_dir = _normalize(forearm * 0.6 + down * 0.4)
        shaft_end = grip_3d + shaft_dir * median_len
        out.append({
            "grip":      _round_point(grip_3d),
            "shaft_end": _round_point(shaft_end),
            "has_club":  True,
        })

    # Pad output to match original frames_3d length (in case n < len(frames_3d))
    while len(out) < len(frames_3d):
        out.append(_missing_entry())
    return out


def _smooth_points(series, window=3):
    """Centered moving-average smoothing over a sequence of np.ndarray | None."""
    n = len(series)
    out = [None] * n
    half = window // 2
    for i in range(n):
        if series[i] is None:
            continue
        accum = []
        for j in range(max(0, i - half), min(n, i + half + 1)):
            if series[j] is not None:
                accum.append(series[j])
        if accum:
            out[i] = np.mean(accum, axis=0)
    return out


# ── Heuristic fallback path (original logic) ───────────────────────────────
def _estimate_with_heuristic(frames_3d: list) -> list:
    body_scale = _estimate_body_scale(frames_3d)
    club_length = body_scale * 1.8 if body_scale > 0 else 1.1

    club_data = []
    for frame in frames_3d:
        landmarks = frame.get("landmarks", [])
        if not landmarks:
            club_data.append(_missing_entry())
            continue

        lw = _get_landmark_pos(landmarks, LEFT_WRIST)
        rw = _get_landmark_pos(landmarks, RIGHT_WRIST)
        le = _get_landmark_pos(landmarks, LEFT_ELBOW)
        re = _get_landmark_pos(landmarks, RIGHT_ELBOW)

        if lw is None or rw is None:
            club_data.append(_missing_entry())
            continue

        grip = (lw + rw) / 2.0

        shaft_dir = np.array([0.0, 1.0, 0.0])
        if le is not None and re is not None:
            left_forearm  = _normalize(lw - le)
            right_forearm = _normalize(rw - re)
            forearm_avg   = _normalize(left_forearm + right_forearm)
            down = np.array([0.0, 1.0, 0.0])
            shaft_dir = _normalize(forearm_avg * 0.6 + down * 0.4)

        shaft_end = grip + shaft_dir * club_length

        club_data.append({
            "grip":      _round_point(grip),
            "shaft_end": _round_point(shaft_end),
            "has_club":  True,
        })

    return club_data


def _estimate_body_scale(frames_3d: list) -> float:
    """Body scale = shoulder-to-hip distance from the first valid frame."""
    for frame in frames_3d[:30]:
        landmarks = frame.get("landmarks", [])
        ls = _get_landmark_pos(landmarks, LEFT_SHOULDER)
        lh = _get_landmark_pos(landmarks, LEFT_HIP)
        if ls is not None and lh is not None:
            return float(np.linalg.norm(ls - lh))
    return 0.0
