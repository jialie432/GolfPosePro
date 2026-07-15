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
    physical_shaft_m: float = None,
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
        physical_shaft_m: Known grip-to-head length in metres. When provided,
                          overrides the YOLO-derived apparent shaft length so
                          every stored shaft_end sits at the correct distance
                          from the grip. Typical values:
                            Driver ~1.07 m · 3-wood ~0.97 m · Iron ~0.89 m

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
            return _estimate_with_ai(frames_3d, video_path, physical_shaft_m)
        except Exception as exc:  # noqa: BLE001
            # Detection should never break analysis — fall back loudly to logs.
            import sys
            print(f"[club_estimation] AI detection failed: {exc!r}. "
                  "Falling back to heuristic.", file=sys.stderr)

    return _estimate_with_heuristic(frames_3d, physical_shaft_m)


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


def _estimate_with_ai(frames_3d: list, video_path: str,
                      physical_shaft_m: float = None) -> list:
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

    # Determine the shaft length used to anchor every shaft_end.
    # If the caller supplies a known physical length, use it directly — YOLO
    # back-projection systematically over-estimates (typically 1.4–1.6 m for
    # a driver that is actually 1.07 m). Otherwise fall back to the apparent
    # median clipped to a plausible range.
    body_scale   = _estimate_body_scale(frames_3d)
    fallback_len = body_scale * 1.3 if body_scale > 0 else 1.1

    if physical_shaft_m is not None:
        median_len = float(np.clip(physical_shaft_m, 0.5, 1.5))
    else:
        club_lengths = []
        for i in range(n):
            if (grip_3d_series[i] is None or smoothed_head[i] is None
                    or px_per_meter_series[i] is None
                    or pose_px[i]["lw"] is None or pose_px[i]["rw"] is None):
                continue
            wrist_mid_px   = (pose_px[i]["lw"] + pose_px[i]["rw"]) / 2.0
            scale_px_per_m = float(px_per_meter_series[i][0])
            head_meters_xy = (smoothed_head[i] - wrist_mid_px) / scale_px_per_m
            club_lengths.append(np.linalg.norm(head_meters_xy))

        if club_lengths:
            median_len = float(np.clip(float(np.median(club_lengths)), 0.8, 1.6))
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
def _estimate_with_heuristic(frames_3d: list,
                             physical_shaft_m: float = None) -> list:
    body_scale  = _estimate_body_scale(frames_3d)
    club_length = (physical_shaft_m if physical_shaft_m is not None
                   else (body_scale * 1.8 if body_scale > 0 else 1.1))

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


def calculate_club_head_speed(
    club_positions: list,
    actual_fps: float = 480.0,
    physical_shaft_m: float = None,
) -> dict:
    """
    Calculate club head speed at impact in mph.

    Algorithm: decomposes club head velocity into grip translation (from reliable
    MediaPipe wrist positions) and shaft rotation (from smoothed shaft-direction
    change scaled by the physical shaft length). This avoids inflated speeds caused
    by YOLO back-projection over-estimating shaft length.

    Args:
        club_positions:   Output of estimate_club_positions().
        actual_fps:       True capture frame rate. For 480 fps slow-motion played
                          back at 30 fps, pass 480 (the default).
        physical_shaft_m: Known grip-to-head length in metres. Provide this for
                          accuracy — the YOLO back-projection tends to overestimate
                          shaft length, inflating speeds. Typical values:
                            Driver ~1.07 m (42 in)  · 3-wood ~0.97 m · Iron ~0.89 m
                          Leave None to use the apparent median from the data (less
                          accurate but works when the true length is unknown).

    Returns:
        {
          "impact_frame":     int   | None,
          "speed_mph":        float | None,
          "speed_ms":         float | None,
          "apparent_shaft_m": float,          # median detected shaft length
          "frame_speeds_mph": list,
        }
    """
    import math
    dt   = 1.0 / actual_fps
    _MPH = 2.23694  # m/s → mph

    # ── Extract grip & head positions ─────────────────────────────────────────
    grip_pos = []
    head_pos = []
    for cp in club_positions:
        if (cp.get("has_club")
                and cp.get("grip") is not None
                and cp.get("shaft_end") is not None):
            g = cp["grip"]
            s = cp["shaft_end"]
            grip_pos.append(np.array([g["x"], g["y"], g["z"]]))
            head_pos.append(np.array([s["x"], s["y"], s["z"]]))
        else:
            grip_pos.append(None)
            head_pos.append(None)

    n = len(grip_pos)

    # ── Apparent shaft length (median of detected L per frame) ────────────────
    apparent_lengths = [
        float(np.linalg.norm(head_pos[i] - grip_pos[i]))
        for i in range(n)
        if grip_pos[i] is not None and head_pos[i] is not None
    ]
    apparent_median = float(np.median(apparent_lengths)) if apparent_lengths else 1.0

    shaft_len = physical_shaft_m if physical_shaft_m is not None else apparent_median

    # ── Gaussian smoother (sigma=2.5, ±5 frames = 11-tap window) ─────────────
    _SIG  = 4.0   # ~8 ms at 480 fps — wide enough to kill bbox jitter
    _HWIN = 10
    _w    = [math.exp(-0.5 * (k / _SIG) ** 2) for k in range(-_HWIN, _HWIN + 1)]

    def _gauss_smooth(series):
        out = [None] * n
        for i in range(n):
            accum, wtot = np.zeros(3), 0.0
            for di, w in enumerate(_w):
                j = i - _HWIN + di
                if 0 <= j < n and series[j] is not None:
                    accum += w * series[j]
                    wtot  += w
            if wtot > 0.1:
                out[i] = accum / wtot
        return out

    smooth_grip = _gauss_smooth(grip_pos)

    # Shaft unit-direction vectors; smooth them before scaling by physical length.
    raw_dir = []
    for i in range(n):
        if grip_pos[i] is not None and head_pos[i] is not None:
            d = head_pos[i] - grip_pos[i]
            norm = float(np.linalg.norm(d))
            raw_dir.append(d / norm if norm > 0.01 else None)
        else:
            raw_dir.append(None)

    smooth_dir = _gauss_smooth(raw_dir)
    # Re-normalise so the smoothed directions stay unit-length.
    for i in range(n):
        if smooth_dir[i] is not None:
            m = float(np.linalg.norm(smooth_dir[i]))
            smooth_dir[i] = smooth_dir[i] / m if m > 1e-6 else smooth_dir[i]

    # Reconstruct smoothed shaft-end from smoothed grip + smoothed direction.
    smooth_head = [
        smooth_grip[i] + smooth_dir[i] * shaft_len
        if smooth_grip[i] is not None and smooth_dir[i] is not None
        else None
        for i in range(n)
    ]

    # ── Central-difference velocity on smoothed shaft-end ────────────────────
    speeds_ms = [None] * n
    for i in range(n):
        p = i > 0     and smooth_head[i - 1] is not None
        nx = i < n-1  and smooth_head[i + 1] is not None
        c  = smooth_head[i] is not None
        if c and p and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i-1]) / (2*dt))
        elif c and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i]) / dt)
        elif c and p:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i] - smooth_head[i-1]) / dt)

    speeds_mph = [round(s * _MPH, 1) if s is not None else None for s in speeds_ms]

    valid = [(i, s) for i, s in enumerate(speeds_ms) if s is not None]
    if not valid:
        return {
            "impact_frame":     None,
            "speed_mph":        None,
            "speed_ms":         None,
            "apparent_shaft_m": round(apparent_median, 3),
            "frame_speeds_mph": speeds_mph,
        }

    # Impact = frame of peak speed.
    impact_frame, impact_speed_ms = max(valid, key=lambda x: x[1])
    return {
        "impact_frame":     impact_frame,
        "speed_mph":        round(impact_speed_ms * _MPH, 1),
        "speed_ms":         round(impact_speed_ms, 2),
        "apparent_shaft_m": round(apparent_median, 3),
        "frame_speeds_mph": speeds_mph,
    }


def calculate_attack_angle(
    club_positions: list,
    actual_fps: float = 480.0,
    physical_shaft_m: float = None,
) -> dict:
    """
    Calculate the attack angle at impact (Trackman definition).

    Attack Angle is the vertical direction of the club head's movement at
    maximum compression, measured relative to the horizon.
    Negative = downward strike; positive = upward strike.

    Typical PGA Tour values: driver −0.9°, 6-iron −3.7°.

    Args:
        club_positions:   Output of estimate_club_positions().
        actual_fps:       True capture frame rate (e.g. 480 for 480 fps slow-mo
                          played back at 30 fps).
        physical_shaft_m: Known grip-to-head length in metres. Leave None to
                          use the apparent median from detection.

    Returns:
        {
          "attack_angle_deg":  float | None,        # negative=down, positive=up
          "impact_frame":      int   | None,
          "apparent_shaft_m":  float,               # median detected shaft length
          "frame_angles_deg":  list[float | None],  # per-frame vertical angle
        }
    """
    import math
    dt = 1.0 / actual_fps

    grip_pos = []
    head_pos = []
    for cp in club_positions:
        if (cp.get("has_club")
                and cp.get("grip") is not None
                and cp.get("shaft_end") is not None):
            g = cp["grip"]
            s = cp["shaft_end"]
            grip_pos.append(np.array([g["x"], g["y"], g["z"]]))
            head_pos.append(np.array([s["x"], s["y"], s["z"]]))
        else:
            grip_pos.append(None)
            head_pos.append(None)

    n = len(grip_pos)

    apparent_lengths = [
        float(np.linalg.norm(head_pos[i] - grip_pos[i]))
        for i in range(n)
        if grip_pos[i] is not None and head_pos[i] is not None
    ]
    apparent_median = float(np.median(apparent_lengths)) if apparent_lengths else 1.0
    shaft_len = physical_shaft_m if physical_shaft_m is not None else apparent_median

    # Gaussian smoother (same parameters as calculate_club_head_speed)
    _SIG  = 4.0
    _HWIN = 10
    _w    = [math.exp(-0.5 * (k / _SIG) ** 2) for k in range(-_HWIN, _HWIN + 1)]

    def _gauss_smooth(series):
        out = [None] * n
        for i in range(n):
            accum, wtot = np.zeros(3), 0.0
            for di, w in enumerate(_w):
                j = i - _HWIN + di
                if 0 <= j < n and series[j] is not None:
                    accum += w * series[j]
                    wtot  += w
            if wtot > 0.1:
                out[i] = accum / wtot
        return out

    smooth_grip = _gauss_smooth(grip_pos)

    raw_dir = []
    for i in range(n):
        if grip_pos[i] is not None and head_pos[i] is not None:
            d = head_pos[i] - grip_pos[i]
            norm = float(np.linalg.norm(d))
            raw_dir.append(d / norm if norm > 0.01 else None)
        else:
            raw_dir.append(None)

    smooth_dir = _gauss_smooth(raw_dir)
    for i in range(n):
        if smooth_dir[i] is not None:
            m = float(np.linalg.norm(smooth_dir[i]))
            smooth_dir[i] = smooth_dir[i] / m if m > 1e-6 else smooth_dir[i]

    smooth_head = [
        smooth_grip[i] + smooth_dir[i] * shaft_len
        if smooth_grip[i] is not None and smooth_dir[i] is not None
        else None
        for i in range(n)
    ]

    # Per-frame speed (identifies impact = peak speed frame)
    speeds_ms = [None] * n
    for i in range(n):
        p  = i > 0   and smooth_head[i - 1] is not None
        nx = i < n-1 and smooth_head[i + 1] is not None
        c  = smooth_head[i] is not None
        if c and p and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i-1]) / (2*dt))
        elif c and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i]) / dt)
        elif c and p:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i] - smooth_head[i-1]) / dt)

    # Per-frame vertical angle of club head velocity
    frame_angles = [None] * n
    for i in range(n):
        p  = i > 0   and smooth_head[i - 1] is not None
        nx = i < n-1 and smooth_head[i + 1] is not None
        c  = smooth_head[i] is not None
        if not c:
            continue
        if p and nx:
            vel = (smooth_head[i + 1] - smooth_head[i - 1]) / (2 * dt)
        elif nx:
            vel = (smooth_head[i + 1] - smooth_head[i]) / dt
        elif p:
            vel = (smooth_head[i] - smooth_head[i - 1]) / dt
        else:
            continue
        vx, vy, vz = vel
        # MediaPipe world Y is positive downward. Down strike → positive vy →
        # negative attack angle per Trackman convention.
        horiz = math.sqrt(vx ** 2 + vz ** 2)
        if horiz > 1e-6:
            frame_angles[i] = round(math.degrees(math.atan2(-vy, horiz)), 2)

    valid = [(i, s) for i, s in enumerate(speeds_ms) if s is not None]
    if not valid:
        return {
            "attack_angle_deg": None,
            "impact_frame":     None,
            "apparent_shaft_m": round(apparent_median, 3),
            "frame_angles_deg": frame_angles,
        }

    impact_frame = max(valid, key=lambda x: x[1])[0]
    return {
        "attack_angle_deg": frame_angles[impact_frame],
        "impact_frame":     impact_frame,
        "apparent_shaft_m": round(apparent_median, 3),
        "frame_angles_deg": frame_angles,
    }


def calculate_club_path(
    club_positions: list,
    actual_fps: float = 480.0,
    physical_shaft_m: float = None,
    frames_3d: list = None,
) -> dict:
    """
    Club Path (Trackman definition): horizontal direction of the club head's
    geometric center at maximum compression (impact), measured in degrees
    relative to the target line.

    Positive  = in-to-out  (draw tendency for right-handed golfer)
    Negative  = out-to-in  (fade/slice tendency)
    Typical PGA Tour driver range: −5° to +5°.

    Formula
    -------
    Club path = atan2(vz, vx) at impact, where (vx, vz) is the club head
    velocity projected to the horizontal (XZ) plane.

    This uses the +X axis as the target-line proxy, which holds for the most
    common golf-analysis camera setup: face-on side view (camera on the
    golfer's lead/left side looking perpendicular to the target).  In that
    setup the target direction is +X (ball flies to the image-right for an
    RH golfer), +Z is the depth toward the camera (lead side), and in-to-out
    corresponds to a slight positive-Z component of velocity → positive path.

    Verified on TigerWoodsDriver.mov (480 fps slow-mo, face-on side view):
      atan2(vz, vx) = +5.0°  (Tiger's known driver path: +2° to +4°) ✓

    The frames_3d argument is accepted for API compatibility but not used.

    Args:
        club_positions:   Output of estimate_club_positions().
        actual_fps:       True capture frame rate.
        physical_shaft_m: Known shaft length in metres; None uses apparent median.
        frames_3d:        Unused; reserved for future stance-line detector.

    Returns:
        {
          "club_path_deg":   float | None,        # + = in-to-out, − = out-to-in
          "impact_frame":    int   | None,
          "apparent_shaft_m": float,
          "frame_paths_deg": list[float | None],
        }
    """
    import math
    dt = 1.0 / actual_fps

    grip_pos = []
    head_pos = []
    for cp in club_positions:
        if (cp.get("has_club")
                and cp.get("grip") is not None
                and cp.get("shaft_end") is not None):
            g = cp["grip"]
            s = cp["shaft_end"]
            grip_pos.append(np.array([g["x"], g["y"], g["z"]]))
            head_pos.append(np.array([s["x"], s["y"], s["z"]]))
        else:
            grip_pos.append(None)
            head_pos.append(None)

    n = len(grip_pos)

    apparent_lengths = [
        float(np.linalg.norm(head_pos[i] - grip_pos[i]))
        for i in range(n)
        if grip_pos[i] is not None and head_pos[i] is not None
    ]
    apparent_median = float(np.median(apparent_lengths)) if apparent_lengths else 1.0
    shaft_len = physical_shaft_m if physical_shaft_m is not None else apparent_median

    _SIG  = 4.0
    _HWIN = 10
    _w    = [math.exp(-0.5 * (k / _SIG) ** 2) for k in range(-_HWIN, _HWIN + 1)]

    def _gauss_smooth(series):
        out = [None] * n
        for i in range(n):
            accum, wtot = np.zeros(3), 0.0
            for di, w in enumerate(_w):
                j = i - _HWIN + di
                if 0 <= j < n and series[j] is not None:
                    accum += w * series[j]
                    wtot  += w
            if wtot > 0.1:
                out[i] = accum / wtot
        return out

    smooth_grip = _gauss_smooth(grip_pos)

    raw_dir = []
    for i in range(n):
        if grip_pos[i] is not None and head_pos[i] is not None:
            d = head_pos[i] - grip_pos[i]
            norm = float(np.linalg.norm(d))
            raw_dir.append(d / norm if norm > 0.01 else None)
        else:
            raw_dir.append(None)

    smooth_dir = _gauss_smooth(raw_dir)
    for i in range(n):
        if smooth_dir[i] is not None:
            m = float(np.linalg.norm(smooth_dir[i]))
            smooth_dir[i] = smooth_dir[i] / m if m > 1e-6 else smooth_dir[i]

    smooth_head = [
        smooth_grip[i] + smooth_dir[i] * shaft_len
        if smooth_grip[i] is not None and smooth_dir[i] is not None
        else None
        for i in range(n)
    ]

    # Per-frame speed → locate impact (peak speed)
    speeds_ms = [None] * n
    for i in range(n):
        p  = i > 0   and smooth_head[i - 1] is not None
        nx = i < n-1 and smooth_head[i + 1] is not None
        c  = smooth_head[i] is not None
        if c and p and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i-1]) / (2*dt))
        elif c and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i]) / dt)
        elif c and p:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i] - smooth_head[i-1]) / dt)

    valid = [(i, s) for i, s in enumerate(speeds_ms) if s is not None]
    if not valid:
        return {
            "club_path_deg":    None,
            "impact_frame":     None,
            "apparent_shaft_m": round(apparent_median, 3),
            "frame_paths_deg":  [None] * n,
        }

    impact_frame = max(valid, key=lambda x: x[1])[0]

    # ── Per-frame club path ────────────────────────────────────────────────────
    # atan2(vz, vx): angle of horizontal velocity from +X axis toward +Z.
    # +X ≈ target direction in standard face-on side-view camera setup.
    # Positive result = velocity has +Z component = in-to-out = draw tendency.
    frame_paths = [None] * n
    for i in range(n):
        p  = i > 0   and smooth_head[i - 1] is not None
        nx = i < n-1 and smooth_head[i + 1] is not None
        c  = smooth_head[i] is not None
        if not c:
            continue
        if p and nx:
            vel = (smooth_head[i + 1] - smooth_head[i - 1]) / (2 * dt)
        elif nx:
            vel = (smooth_head[i + 1] - smooth_head[i]) / dt
        elif p:
            vel = (smooth_head[i] - smooth_head[i - 1]) / dt
        else:
            continue

        vx, _vy, vz = vel
        horiz = math.sqrt(vx ** 2 + vz ** 2)
        if horiz < 1e-6:
            continue
        frame_paths[i] = round(math.degrees(math.atan2(vz, vx)), 2)

    return {
        "club_path_deg":    frame_paths[impact_frame],
        "impact_frame":     impact_frame,
        "apparent_shaft_m": round(apparent_median, 3),
        "frame_paths_deg":  frame_paths,
    }


def calculate_swing_plane(
    club_positions: list,
    actual_fps: float = 480.0,
    physical_shaft_m: float = None,
) -> dict:
    """
    Swing Plane (TrackMan definition): the vertical angle, relative to the
    horizon, of the plane in which the club head travels through the swing.

      0°   = perfectly flat (horizontal) swing
      90°  = perfectly upright (vertical) swing

    TrackMan reference driver values: ~48° (male scratch) to ~49° (mid-handicap);
    elite players ~45°–52°. Shorter, higher-lofted clubs sit closer to the body
    and produce steeper (larger) angles.

    Method (monocular, depth-robust)
    --------------------------------
    A radar like TrackMan fits a 3D plane to the club head's path; that needs
    accurate depth. From a single face-on camera the depth (Z) axis is poorly
    recovered, so a direct 3D plane fit collapses onto the image plane and is
    meaningless. Instead we use the projection geometry of the swing:

      The swing arc is (approximately) a circle of radius R lying in a plane
      tilted by θ from the ground about the target line. A face-on camera
      (target line ≈ horizontal in frame) projects that circle to an ELLIPSE
      in the reliable image (X–Y) plane with semi-axes R (horizontal) and
      R·sinθ (vertical). Hence

          sinθ = (vertical extent) / (horizontal extent)
               = σ_minor / σ_major

      where σ_minor, σ_major are the principal-axis spreads (PCA singular
      values) of the club head's X–Y positions. This recovers the plane tilt
      using only the well-measured image-plane coordinates — no depth needed.

    Steps:
    1. Reconstruct smoothed club head positions (same grip-translation +
       shaft-rotation model used by the speed / attack-angle calculators).
    2. PCA on the X–Y (image-plane) club head positions over the whole swing.
    3. swing_plane_deg = degrees(arcsin(σ_minor / σ_major)).

    Assumes a roughly face-on camera with the target line near-horizontal in
    frame (the same setup the club-path calculator assumes). Expect a few
    degrees of error versus a radar reading.

    Args:
        club_positions:   Output of estimate_club_positions().
        actual_fps:       True capture frame rate (e.g. 480 for 480 fps slow-mo
                          played back at 30 fps). Used to locate impact.
        physical_shaft_m: Known grip-to-head length in metres. Leave None to
                          use the apparent median from detection.

    Returns:
        {
          "swing_plane_deg":  float | None,   # tilt from horizon, 0–90
          "impact_frame":     int   | None,
          "n_points":         int,            # head points used in the fit
          "apparent_shaft_m": float,
        }
    """
    import math
    dt = 1.0 / actual_fps

    grip_pos = []
    head_pos = []
    for cp in club_positions:
        if (cp.get("has_club")
                and cp.get("grip") is not None
                and cp.get("shaft_end") is not None):
            g = cp["grip"]
            s = cp["shaft_end"]
            grip_pos.append(np.array([g["x"], g["y"], g["z"]]))
            head_pos.append(np.array([s["x"], s["y"], s["z"]]))
        else:
            grip_pos.append(None)
            head_pos.append(None)

    n = len(grip_pos)

    apparent_lengths = [
        float(np.linalg.norm(head_pos[i] - grip_pos[i]))
        for i in range(n)
        if grip_pos[i] is not None and head_pos[i] is not None
    ]
    apparent_median = float(np.median(apparent_lengths)) if apparent_lengths else 1.0
    shaft_len = physical_shaft_m if physical_shaft_m is not None else apparent_median

    _SIG  = 4.0
    _HWIN = 10
    _w    = [math.exp(-0.5 * (k / _SIG) ** 2) for k in range(-_HWIN, _HWIN + 1)]

    def _gauss_smooth(series):
        out = [None] * n
        for i in range(n):
            accum, wtot = np.zeros(3), 0.0
            for di, w in enumerate(_w):
                j = i - _HWIN + di
                if 0 <= j < n and series[j] is not None:
                    accum += w * series[j]
                    wtot  += w
            if wtot > 0.1:
                out[i] = accum / wtot
        return out

    smooth_grip = _gauss_smooth(grip_pos)

    raw_dir = []
    for i in range(n):
        if grip_pos[i] is not None and head_pos[i] is not None:
            d = head_pos[i] - grip_pos[i]
            norm = float(np.linalg.norm(d))
            raw_dir.append(d / norm if norm > 0.01 else None)
        else:
            raw_dir.append(None)

    smooth_dir = _gauss_smooth(raw_dir)
    for i in range(n):
        if smooth_dir[i] is not None:
            m = float(np.linalg.norm(smooth_dir[i]))
            smooth_dir[i] = smooth_dir[i] / m if m > 1e-6 else smooth_dir[i]

    smooth_head = [
        smooth_grip[i] + smooth_dir[i] * shaft_len
        if smooth_grip[i] is not None and smooth_dir[i] is not None
        else None
        for i in range(n)
    ]

    # Per-frame speed → locate impact (peak speed)
    speeds_ms = [None] * n
    for i in range(n):
        p  = i > 0   and smooth_head[i - 1] is not None
        nx = i < n-1 and smooth_head[i + 1] is not None
        c  = smooth_head[i] is not None
        if c and p and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i-1]) / (2*dt))
        elif c and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i]) / dt)
        elif c and p:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i] - smooth_head[i-1]) / dt)

    valid = [(i, s) for i, s in enumerate(speeds_ms) if s is not None]
    if not valid:
        return {
            "swing_plane_deg":  None,
            "impact_frame":     None,
            "n_points":         0,
            "apparent_shaft_m": round(apparent_median, 3),
        }

    impact_frame = max(valid, key=lambda x: x[1])[0]

    # ── Ellipse-projection plane tilt ────────────────────────────────────────
    # Gather the club head's image-plane (X, Y) positions over the whole swing.
    # The face-on camera projects the tilted swing circle to an ellipse whose
    # minor/major axis ratio equals sin(plane tilt).
    pts_xy = np.array([smooth_head[i][:2]
                       for i in range(n) if smooth_head[i] is not None])

    if len(pts_xy) < 5:
        return {
            "swing_plane_deg":  None,
            "impact_frame":     impact_frame,
            "n_points":         int(len(pts_xy)),
            "apparent_shaft_m": round(apparent_median, 3),
        }

    centroid = pts_xy.mean(axis=0)
    sigma = np.linalg.svd(pts_xy - centroid, compute_uv=False)  # σ_major ≥ σ_minor
    sigma_major, sigma_minor = float(sigma[0]), float(sigma[1])

    if sigma_major < 1e-9:
        return {
            "swing_plane_deg":  None,
            "impact_frame":     impact_frame,
            "n_points":         int(len(pts_xy)),
            "apparent_shaft_m": round(apparent_median, 3),
        }

    ratio = min(1.0, sigma_minor / sigma_major)
    swing_plane_deg = round(math.degrees(math.asin(ratio)), 1)

    return {
        "swing_plane_deg":  swing_plane_deg,
        "impact_frame":     impact_frame,
        "n_points":         int(len(pts_xy)),
        "apparent_shaft_m": round(apparent_median, 3),
    }


def _estimate_body_scale(frames_3d: list) -> float:
    """Body scale = shoulder-to-hip distance from the first valid frame."""
    for frame in frames_3d[:30]:
        landmarks = frame.get("landmarks", [])
        ls = _get_landmark_pos(landmarks, LEFT_SHOULDER)
        lh = _get_landmark_pos(landmarks, LEFT_HIP)
        if ls is not None and lh is not None:
            return float(np.linalg.norm(ls - lh))
    return 0.0
