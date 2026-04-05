# Walkthrough: Velocity & Angle Extraction with Smoothing

## Summary

Added velocity computation, joint angle tracking, and adaptive One-Euro smoothing to GolfPosePro. The system now extracts per-frame kinematics data from MediaPipe landmarks and displays it in a new **📐 Kinematics** tab.

## New Files

### [smoothing.py](file:///Users/jialielu/highschoolproj/GolfPosePro/golf_pose_pro/smoothing.py)

Implements the **One-Euro Filter** (Casiez et al., CHI 2012) — an adaptive low-pass filter designed for noisy human motion signals:
- **Slow motion** → heavy smoothing (eliminates jitter)
- **Fast motion** → light smoothing (preserves responsiveness)

Key functions:
- `OneEuroFilter` class — core adaptive filter
- `smooth_series()` — apply filter to a 1D time-series
- `smooth_landmarks()` — apply filter to all landmark coordinates in frame_data
- `smooth_series_uniform()` — backward-compatible moving average fallback

---

### [kinematics.py](file:///Users/jialielu/highschoolproj/GolfPosePro/golf_pose_pro/kinematics.py)

Core computation module for biomechanical metrics:

**Velocities:**
- `compute_landmark_velocity()` — 2D/3D speed from coordinate time-series (px/s or m/s)
- `compute_velocities_from_frame_data()` — batch compute for all tracked landmarks
- `compute_angular_velocity()` — angular velocity (°/s) from angle series

**Joint Angles:**
- `compute_joint_angle()` — 3-point angle at a vertex (A-B-C)
- `compute_all_joint_angles()` — compute all defined angles per frame from 3D data:
  - Left/Right **elbow** (shoulder→elbow→wrist)
  - Left/Right **knee** (hip→knee→ankle)
  - Left/Right **shoulder** (hip→shoulder→elbow)
  - **Spine tilt** (vertical vs. mid-shoulder→mid-hip)

**Peak Detection:**
- `find_peak_velocities()` — peak speed per swing phase
- `find_peak_angular_velocities()` — peak angular velocity per joint per phase

## Modified Files

### [pose_extraction.py](file:///Users/jialielu/highschoolproj/GolfPosePro/golf_pose_pro/pose_extraction.py)

```diff:pose_extraction.py
"""
pose_extraction.py — MediaPipe Pose Tasks API (mediapipe >= 0.10.14)
Supports wrist, hip, and shoulder tracking per frame.
"""

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
        PoseLandmark,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# Default model path — sits next to the package in the project root
_DEFAULT_MODEL = Path(__file__).parent.parent / "pose_landmarker.task"

# ─── Landmark groups (by Tasks API enum int values) ──────────────────────────

LANDMARK_GROUPS = {
    "wrist":    [PoseLandmark.LEFT_WRIST.value,    PoseLandmark.RIGHT_WRIST.value]    if MP_AVAILABLE else [],
    "hip":      [PoseLandmark.LEFT_HIP.value,      PoseLandmark.RIGHT_HIP.value]      if MP_AVAILABLE else [],
    "shoulder": [PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.RIGHT_SHOULDER.value] if MP_AVAILABLE else [],
}


# ─── Video utilities ─────────────────────────────────────────────────────────

def add_silent_audio(input_path: str, output_path: str) -> str:
    """Re-encode video with a silent audio track via FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest",
        "-c:v", "libx264", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return output_path


def get_video_info(video_path: str) -> dict:
    """Return basic metadata about a video file."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


# ─── Core extraction ─────────────────────────────────────────────────────────

def extract_pose_features(
    video_path: str,
    track_landmarks: list = None,
    visibility_threshold: float = 0.4,
    model_path: str = None,
    progress_callback=None,
) -> list:
    """
    Extract pose landmarks frame-by-frame using the MediaPipe Tasks API.

    Args:
        video_path:           Path to input video.
        track_landmarks:      Groups to track ('wrist', 'hip', 'shoulder').
        visibility_threshold: Landmark visibility threshold (0–1).
        model_path:           Path to .task model file. Defaults to project-root model.
        progress_callback:    Optional callable(frame_idx, total_frames).

    Returns:
        List of dicts [{frame_idx, wrist_y, hip_y?, shoulder_y?}, …]
    """
    if not MP_AVAILABLE:
        raise RuntimeError("mediapipe is not installed. Run: pip install mediapipe")

    if track_landmarks is None:
        track_landmarks = ["wrist"]

    if model_path is None:
        model_path = str(_DEFAULT_MODEL)

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Pose model not found at: {model_path}\n"
            "Download it with:\n"
            "  curl -L -o pose_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        )

    # Build per-group index lists
    lm_indices = {group: LANDMARK_GROUPS[group] for group in track_landmarks
                  if group in LANDMARK_GROUPS}

    cap = cv2.VideoCapture(video_path)
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_data = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Tasks API requires timestamps in milliseconds
            timestamp_ms = int(frame_idx * 1000 / fps)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            entry = {"frame_idx": frame_idx}

            poses = result.pose_landmarks  # list of lists of NormalizedLandmark
            if poses:
                landmarks = poses[0]  # first detected pose
                for group, indices in lm_indices.items():
                    vals = []
                    for idx in indices:
                        lm = landmarks[idx]
                        if lm.visibility is None or lm.visibility > visibility_threshold:
                            vals.append(lm.y * height)
                    entry[f"{group}_y"] = float(np.mean(vals)) if vals else None
            else:
                for group in track_landmarks:
                    entry[f"{group}_y"] = None

            frame_data.append(entry)
            frame_idx += 1

            if progress_callback:
                progress_callback(frame_idx, total_frames)

    cap.release()
    return frame_data


def extract_full_3d_landmarks(
    video_path: str,
    visibility_threshold: float = 0.4,
    model_path: str = None,
    progress_callback=None,
) -> list:
    """
    Extract all 33 world landmarks (x, y, z in meters) per frame.

    Uses pose_world_landmarks which provides 3D coordinates relative
    to the hip center — suitable for Three.js rendering.

    Returns:
        List of dicts: [{frame_idx, timestamp_ms, landmarks: [{idx, x, y, z, visibility}, ...]}, ...]
    """
    if not MP_AVAILABLE:
        raise RuntimeError("mediapipe is not installed. Run: pip install mediapipe")

    if model_path is None:
        model_path = str(_DEFAULT_MODEL)

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Pose model not found at: {model_path}\n"
            "Download it with:\n"
            "  curl -L -o pose_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    frames_3d = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp_ms = int(frame_idx * 1000 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            entry = {"frame_idx": frame_idx, "timestamp_ms": timestamp_ms, "landmarks": []}

            world_landmarks = result.pose_world_landmarks
            if world_landmarks:
                wl = world_landmarks[0]
                for i, lm in enumerate(wl):
                    vis = lm.visibility if lm.visibility is not None else 0.0
                    if vis >= visibility_threshold:
                        entry["landmarks"].append({
                            "idx": i,
                            "x": round(float(lm.x), 5),
                            "y": round(float(lm.y), 5),
                            "z": round(float(lm.z), 5),
                            "visibility": round(float(vis), 3),
                        })
                    else:
                        entry["landmarks"].append({
                            "idx": i, "x": None, "y": None, "z": None,
                            "visibility": round(float(vis), 3),
                        })

            frames_3d.append(entry)
            frame_idx += 1

            if progress_callback:
                progress_callback(frame_idx, total_frames)

    cap.release()
    return frames_3d


def extract_y_series(frame_data: list, key: str = "wrist_y") -> np.ndarray:
    """Pull a Y-coordinate series from frame_data, forward-filling NaNs."""
    values = [f.get(key) for f in frame_data]
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)

    nans = np.isnan(arr)
    if nans.any():
        idx = np.where(~nans, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        arr = arr[idx]

    return arr
===
"""
pose_extraction.py — MediaPipe Pose Tasks API (mediapipe >= 0.10.14)
Supports wrist, hip, and shoulder tracking per frame.
Extracts both x and y coordinates for velocity computation.
"""

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
        PoseLandmark,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# Default model path — sits next to the package in the project root
_DEFAULT_MODEL = Path(__file__).parent.parent / "pose_landmarker.task"

# ─── Landmark groups (by Tasks API enum int values) ──────────────────────────

LANDMARK_GROUPS = {
    "wrist":    [PoseLandmark.LEFT_WRIST.value,    PoseLandmark.RIGHT_WRIST.value]    if MP_AVAILABLE else [],
    "hip":      [PoseLandmark.LEFT_HIP.value,      PoseLandmark.RIGHT_HIP.value]      if MP_AVAILABLE else [],
    "shoulder": [PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.RIGHT_SHOULDER.value] if MP_AVAILABLE else [],
}


# ─── Video utilities ─────────────────────────────────────────────────────────

def add_silent_audio(input_path: str, output_path: str) -> str:
    """Re-encode video with a silent audio track via FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest",
        "-c:v", "libx264", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return output_path


def get_video_info(video_path: str) -> dict:
    """Return basic metadata about a video file."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


# ─── Core extraction ─────────────────────────────────────────────────────────

def extract_pose_features(
    video_path: str,
    track_landmarks: list = None,
    visibility_threshold: float = 0.4,
    model_path: str = None,
    progress_callback=None,
    apply_smoothing: bool = False,
    smoothing_min_cutoff: float = 1.0,
    smoothing_beta: float = 0.007,
) -> list:
    """
    Extract pose landmarks frame-by-frame using the MediaPipe Tasks API.

    Args:
        video_path:           Path to input video.
        track_landmarks:      Groups to track ('wrist', 'hip', 'shoulder').
        visibility_threshold: Landmark visibility threshold (0–1).
        model_path:           Path to .task model file. Defaults to project-root model.
        progress_callback:    Optional callable(frame_idx, total_frames).
        apply_smoothing:      If True, apply One-Euro filter to landmark coords.
        smoothing_min_cutoff: One-Euro min_cutoff param (lower = more smoothing).
        smoothing_beta:       One-Euro beta param (higher = faster adaptation).

    Returns:
        List of dicts [{frame_idx, wrist_x, wrist_y, hip_x?, hip_y?, …}, …]
    """
    if not MP_AVAILABLE:
        raise RuntimeError("mediapipe is not installed. Run: pip install mediapipe")

    if track_landmarks is None:
        track_landmarks = ["wrist"]

    if model_path is None:
        model_path = str(_DEFAULT_MODEL)

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Pose model not found at: {model_path}\n"
            "Download it with:\n"
            "  curl -L -o pose_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        )

    # Build per-group index lists
    lm_indices = {group: LANDMARK_GROUPS[group] for group in track_landmarks
                  if group in LANDMARK_GROUPS}

    cap = cv2.VideoCapture(video_path)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_data = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Tasks API requires timestamps in milliseconds
            timestamp_ms = int(frame_idx * 1000 / fps)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            entry = {"frame_idx": frame_idx}

            poses = result.pose_landmarks  # list of lists of NormalizedLandmark
            if poses:
                landmarks = poses[0]  # first detected pose
                for group, indices in lm_indices.items():
                    x_vals = []
                    y_vals = []
                    for idx in indices:
                        lm = landmarks[idx]
                        if lm.visibility is None or lm.visibility > visibility_threshold:
                            x_vals.append(lm.x * width)
                            y_vals.append(lm.y * height)
                    entry[f"{group}_x"] = float(np.mean(x_vals)) if x_vals else None
                    entry[f"{group}_y"] = float(np.mean(y_vals)) if y_vals else None
            else:
                for group in track_landmarks:
                    entry[f"{group}_x"] = None
                    entry[f"{group}_y"] = None

            frame_data.append(entry)
            frame_idx += 1

            if progress_callback:
                progress_callback(frame_idx, total_frames)

    cap.release()

    # Apply One-Euro smoothing if requested
    if apply_smoothing and frame_data:
        from .smoothing import smooth_landmarks
        frame_data = smooth_landmarks(
            frame_data, fps=fps,
            min_cutoff=smoothing_min_cutoff, beta=smoothing_beta,
        )

    return frame_data


def extract_full_3d_landmarks(
    video_path: str,
    visibility_threshold: float = 0.4,
    model_path: str = None,
    progress_callback=None,
) -> list:
    """
    Extract all 33 world landmarks (x, y, z in meters) per frame.

    Uses pose_world_landmarks which provides 3D coordinates relative
    to the hip center — suitable for Three.js rendering.

    Returns:
        List of dicts: [{frame_idx, timestamp_ms, landmarks: [{idx, x, y, z, visibility}, ...]}, ...]
    """
    if not MP_AVAILABLE:
        raise RuntimeError("mediapipe is not installed. Run: pip install mediapipe")

    if model_path is None:
        model_path = str(_DEFAULT_MODEL)

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Pose model not found at: {model_path}\n"
            "Download it with:\n"
            "  curl -L -o pose_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    frames_3d = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp_ms = int(frame_idx * 1000 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            entry = {"frame_idx": frame_idx, "timestamp_ms": timestamp_ms, "landmarks": []}

            world_landmarks = result.pose_world_landmarks
            if world_landmarks:
                wl = world_landmarks[0]
                for i, lm in enumerate(wl):
                    vis = lm.visibility if lm.visibility is not None else 0.0
                    if vis >= visibility_threshold:
                        entry["landmarks"].append({
                            "idx": i,
                            "x": round(float(lm.x), 5),
                            "y": round(float(lm.y), 5),
                            "z": round(float(lm.z), 5),
                            "visibility": round(float(vis), 3),
                        })
                    else:
                        entry["landmarks"].append({
                            "idx": i, "x": None, "y": None, "z": None,
                            "visibility": round(float(vis), 3),
                        })

            frames_3d.append(entry)
            frame_idx += 1

            if progress_callback:
                progress_callback(frame_idx, total_frames)

    cap.release()
    return frames_3d


def extract_y_series(frame_data: list, key: str = "wrist_y") -> np.ndarray:
    """Pull a Y-coordinate series from frame_data, forward-filling NaNs."""
    values = [f.get(key) for f in frame_data]
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)

    nans = np.isnan(arr)
    if nans.any():
        idx = np.where(~nans, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        arr = arr[idx]

    return arr
```

- Now extracts **x-coordinates** alongside y (needed for 2D velocity = √(Δx² + Δy²))
- Added `apply_smoothing`, `smoothing_min_cutoff`, `smoothing_beta` parameters
- When smoothing is enabled, applies One-Euro filter to all landmark coordinates post-extraction

---

### [phase_detection.py](file:///Users/jialielu/highschoolproj/GolfPosePro/golf_pose_pro/phase_detection.py)

```diff:phase_detection.py
"""
phase_detection.py — Golf swing phase segmentation.

Detects 6 phases from a wrist Y-trajectory:
  Address → Backswing → Top → Downswing → Impact → Follow Through
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PHASE_COLORS = {
    "Address":       "#6c757d",
    "Backswing":     "#3a86ff",
    "Top":           "#8338ec",
    "Downswing":     "#ff006e",
    "Impact":        "#fb5607",
    "Follow Through":"#06d6a0",
}


def detect_swing_phases(
    frame_data: list,
    signal_key: str = "wrist_y",
    smoothing_window: int = 5,
    precheck_window: int = 30,
    threshold_percentile: float = 90,
) -> tuple:
    """
    Detect golf swing phases from frame data.

    Args:
        frame_data:            Output of extract_pose_features().
        signal_key:            Which Y-series to use ('wrist_y', 'hip_y', etc.).
        smoothing_window:      Moving-average kernel size.
        precheck_window:       Frames at clip start to skip when finding motion onset.
        threshold_percentile:  Velocity percentile that marks swing start.

    Returns:
        (phase_ranges, swing_start, swing_end, wrist_y, smoothed)
        - phase_ranges: dict  {phase_name: (start_frame, end_frame)}
        - swing_start:  int
        - swing_end:    int
        - wrist_y:      np.ndarray  (raw)
        - smoothed:     np.ndarray  (smoothed)
    """
    # ── 1. Build raw + smoothed arrays ──────────────────────────────────────
    raw = np.array(
        [f.get(signal_key) if f.get(signal_key) is not None else np.nan
         for f in frame_data],
        dtype=float,
    )

    # forward-fill NaNs
    nans = np.isnan(raw)
    if nans.any():
        idx = np.where(~nans, np.arange(len(raw)), 0)
        np.maximum.accumulate(idx, out=idx)
        raw = raw[idx]

    smoothed = uniform_filter1d(raw, size=smoothing_window, mode="nearest")
    velocity_mag = np.abs(np.gradient(smoothed))

    # ── 2. Swing start (motion onset) ───────────────────────────────────────
    buffer_start = min(precheck_window, len(velocity_mag) // 4)
    threshold = np.nanpercentile(velocity_mag[buffer_start:], threshold_percentile)
    motion_indices = np.where(velocity_mag > threshold)[0]
    motion_indices = motion_indices[motion_indices >= buffer_start]

    if len(motion_indices) == 0:
        # Fallback: use whole clip
        swing_start = buffer_start
        swing_end   = len(raw) - 1
    else:
        swing_start = int(motion_indices[0])

        # ── 3. Swing end (wrist returns to address height) ───────────────────
        peak_idx   = motion_indices[np.argmax(velocity_mag[motion_indices])]
        target_y   = smoothed[swing_start]
        post_range = smoothed[peak_idx + 1:]
        if len(post_range) == 0:
            swing_end = len(raw) - 1
        else:
            offset    = np.nanargmin(np.abs(post_range - target_y))
            swing_end = peak_idx + 1 + int(offset)

    # Clamp
    swing_start = max(0, min(swing_start, len(raw) - 1))
    swing_end   = max(swing_start + 1, min(swing_end, len(raw) - 1))

    # ── 4. Address (pre-swing stillness) ────────────────────────────────────
    flat_std_thresh = 1.0
    min_flat_frames = 5
    address_start   = max(0, swing_start - min_flat_frames)

    for i in range(swing_start - min_flat_frames, 0, -1):
        window = smoothed[i:swing_start]
        if np.nanstd(window) < flat_std_thresh:
            address_start = i
        else:
            break

    address_end = swing_start

    # ── 5. Top of backswing ──────────────────────────────────────────────────
    swing_segment = smoothed[swing_start:swing_end + 1]
    if len(swing_segment) < 3:
        top_start = swing_start
        top_end   = swing_start + 1
    else:
        peak_local = int(np.argmin(swing_segment))  # min Y = highest wrist
        top_start  = max(swing_start, swing_start + peak_local - 5)
        top_end    = min(swing_end,   swing_start + peak_local + 5)

    # ── 6. Impact zone ───────────────────────────────────────────────────────
    post_top_segment = smoothed[top_end:swing_end + 1]
    if len(post_top_segment) < 2:
        impact_start = top_end
        impact_end   = swing_end
    else:
        y_range      = np.nanmax(smoothed) - np.nanmin(smoothed)
        tol          = y_range * 0.05
        impact_local = int(np.argmax(post_top_segment))  # max Y = lowest wrist (impact)
        impact_center = top_end + impact_local
        impact_start  = max(top_end,   impact_center - max(2, int(tol)))
        impact_end    = min(swing_end, impact_center + max(2, int(tol)))

    # ── 7. Assemble phase dict ───────────────────────────────────────────────
    phase_ranges = {
        "Address":       (address_start, address_end),
        "Backswing":     (swing_start,   top_start),
        "Top":           (top_start,     top_end),
        "Downswing":     (top_end,       impact_start),
        "Impact":        (impact_start,  impact_end),
        "Follow Through":(impact_end,    swing_end),
    }

    return phase_ranges, swing_start, swing_end, raw, smoothed


def get_phase_label(frame_idx: int, phase_ranges: dict) -> str | None:
    """Return phase name for a given frame index (or None if outside all phases)."""
    for label, (start, end) in phase_ranges.items():
        if start <= frame_idx <= end:
            return label
    return None


def plot_trajectory(
    wrist_y: np.ndarray,
    smoothed: np.ndarray,
    phase_ranges: dict,
    swing_start: int,
    swing_end: int,
    title: str = "Wrist Y-Trajectory",
    extra_series: dict = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Render a high-quality trajectory plot with phase spans.

    Args:
        extra_series: Optional dict of {label: np.ndarray} for hip/shoulder.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    x = np.arange(len(wrist_y))

    # Phase shading
    for phase, (s, e) in phase_ranges.items():
        color = PHASE_COLORS.get(phase, "#888888")
        ax.axvspan(s, e, alpha=0.18, color=color)
        mid = (s + e) / 2
        ax.text(
            mid, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else np.nanmax(wrist_y) + 5,
            phase, ha="center", fontsize=7, color=color, alpha=0.9,
        )

    # Raw wrist
    ax.plot(x, wrist_y, color="#94a3b8", alpha=0.4, linewidth=1, label="Raw Wrist Y")
    # Smoothed wrist
    ax.plot(x, smoothed, color="#4ade80", linewidth=2.2, label="Smoothed Wrist Y")

    # Extra landmark series
    if extra_series:
        palette = ["#fb923c", "#38bdf8", "#f472b6"]
        for i, (lbl, series) in enumerate(extra_series.items()):
            ax.plot(x[: len(series)], series, color=palette[i % len(palette)],
                    linewidth=1.6, linestyle="--", label=lbl, alpha=0.85)

    # Swing window
    ax.axvline(swing_start, color="#facc15", linestyle="--", linewidth=1.2, label="Swing Start")
    ax.axvline(swing_end,   color="#f87171", linestyle="--", linewidth=1.2, label="Swing End")

    # Phase legend patches
    phase_patches = [
        Patch(facecolor=PHASE_COLORS[p], alpha=0.5, label=p) for p in phase_ranges
    ]

    legend1 = ax.legend(loc="upper left", fontsize=8, facecolor="#1e293b",
                        labelcolor="white", framealpha=0.8)
    ax.add_artist(legend1)
    ax.legend(handles=phase_patches, loc="upper right", fontsize=7,
              facecolor="#1e293b", labelcolor="white", framealpha=0.8, title="Phases",
              title_fontsize=7)

    ax.invert_yaxis()  # MediaPipe Y is flipped — invert so "up" = up
    ax.set_title(title, color="white", fontsize=13, pad=12)
    ax.set_xlabel("Frame", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Wrist Y (px, inverted)", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    fig.tight_layout()
    return fig
===
"""
phase_detection.py — Golf swing phase segmentation.

Detects 6 phases from a wrist Y-trajectory:
  Address → Backswing → Top → Downswing → Impact → Follow Through
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .smoothing import smooth_series as _one_euro_smooth


PHASE_COLORS = {
    "Address":       "#6c757d",
    "Backswing":     "#3a86ff",
    "Top":           "#8338ec",
    "Downswing":     "#ff006e",
    "Impact":        "#fb5607",
    "Follow Through":"#06d6a0",
}


def detect_swing_phases(
    frame_data: list,
    signal_key: str = "wrist_y",
    smoothing_window: int = 5,
    precheck_window: int = 30,
    threshold_percentile: float = 90,
    use_one_euro: bool = False,
    fps: float = 30.0,
    one_euro_min_cutoff: float = 1.0,
    one_euro_beta: float = 0.007,
) -> tuple:
    """
    Detect golf swing phases from frame data.

    Args:
        frame_data:            Output of extract_pose_features().
        signal_key:            Which Y-series to use ('wrist_y', 'hip_y', etc.).
        smoothing_window:      Moving-average kernel size (used when use_one_euro=False).
        precheck_window:       Frames at clip start to skip when finding motion onset.
        threshold_percentile:  Velocity percentile that marks swing start.
        use_one_euro:          Use One-Euro adaptive filter instead of uniform average.
        fps:                   Video FPS (needed for One-Euro filter).
        one_euro_min_cutoff:   One-Euro min_cutoff (lower = more smoothing).
        one_euro_beta:         One-Euro beta (higher = faster adaptation).

    Returns:
        (phase_ranges, swing_start, swing_end, wrist_y, smoothed)
        - phase_ranges: dict  {phase_name: (start_frame, end_frame)}
        - swing_start:  int
        - swing_end:    int
        - wrist_y:      np.ndarray  (raw)
        - smoothed:     np.ndarray  (smoothed)
    """
    # ── 1. Build raw + smoothed arrays ──────────────────────────────────────
    raw = np.array(
        [f.get(signal_key) if f.get(signal_key) is not None else np.nan
         for f in frame_data],
        dtype=float,
    )

    # forward-fill NaNs
    nans = np.isnan(raw)
    if nans.any():
        idx = np.where(~nans, np.arange(len(raw)), 0)
        np.maximum.accumulate(idx, out=idx)
        raw = raw[idx]

    smoothed = (
        _one_euro_smooth(raw, fps=fps, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
        if use_one_euro
        else uniform_filter1d(raw, size=smoothing_window, mode="nearest")
    )
    velocity_mag = np.abs(np.gradient(smoothed))

    # ── 2. Swing start (motion onset) ───────────────────────────────────────
    buffer_start = min(precheck_window, len(velocity_mag) // 4)
    threshold = np.nanpercentile(velocity_mag[buffer_start:], threshold_percentile)
    motion_indices = np.where(velocity_mag > threshold)[0]
    motion_indices = motion_indices[motion_indices >= buffer_start]

    if len(motion_indices) == 0:
        # Fallback: use whole clip
        swing_start = buffer_start
        swing_end   = len(raw) - 1
    else:
        swing_start = int(motion_indices[0])

        # ── 3. Swing end (wrist returns to address height) ───────────────────
        peak_idx   = motion_indices[np.argmax(velocity_mag[motion_indices])]
        target_y   = smoothed[swing_start]
        post_range = smoothed[peak_idx + 1:]
        if len(post_range) == 0:
            swing_end = len(raw) - 1
        else:
            offset    = np.nanargmin(np.abs(post_range - target_y))
            swing_end = peak_idx + 1 + int(offset)

    # Clamp
    swing_start = max(0, min(swing_start, len(raw) - 1))
    swing_end   = max(swing_start + 1, min(swing_end, len(raw) - 1))

    # ── 4. Address (pre-swing stillness) ────────────────────────────────────
    flat_std_thresh = 1.0
    min_flat_frames = 5
    address_start   = max(0, swing_start - min_flat_frames)

    for i in range(swing_start - min_flat_frames, 0, -1):
        window = smoothed[i:swing_start]
        if np.nanstd(window) < flat_std_thresh:
            address_start = i
        else:
            break

    address_end = swing_start

    # ── 5. Top of backswing ──────────────────────────────────────────────────
    swing_segment = smoothed[swing_start:swing_end + 1]
    if len(swing_segment) < 3:
        top_start = swing_start
        top_end   = swing_start + 1
    else:
        peak_local = int(np.argmin(swing_segment))  # min Y = highest wrist
        top_start  = max(swing_start, swing_start + peak_local - 5)
        top_end    = min(swing_end,   swing_start + peak_local + 5)

    # ── 6. Impact zone ───────────────────────────────────────────────────────
    post_top_segment = smoothed[top_end:swing_end + 1]
    if len(post_top_segment) < 2:
        impact_start = top_end
        impact_end   = swing_end
    else:
        y_range      = np.nanmax(smoothed) - np.nanmin(smoothed)
        tol          = y_range * 0.05
        impact_local = int(np.argmax(post_top_segment))  # max Y = lowest wrist (impact)
        impact_center = top_end + impact_local
        impact_start  = max(top_end,   impact_center - max(2, int(tol)))
        impact_end    = min(swing_end, impact_center + max(2, int(tol)))

    # ── 7. Assemble phase dict ───────────────────────────────────────────────
    phase_ranges = {
        "Address":       (address_start, address_end),
        "Backswing":     (swing_start,   top_start),
        "Top":           (top_start,     top_end),
        "Downswing":     (top_end,       impact_start),
        "Impact":        (impact_start,  impact_end),
        "Follow Through":(impact_end,    swing_end),
    }

    return phase_ranges, swing_start, swing_end, raw, smoothed


def get_phase_label(frame_idx: int, phase_ranges: dict) -> str | None:
    """Return phase name for a given frame index (or None if outside all phases)."""
    for label, (start, end) in phase_ranges.items():
        if start <= frame_idx <= end:
            return label
    return None


def plot_trajectory(
    wrist_y: np.ndarray,
    smoothed: np.ndarray,
    phase_ranges: dict,
    swing_start: int,
    swing_end: int,
    title: str = "Wrist Y-Trajectory",
    extra_series: dict = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Render a high-quality trajectory plot with phase spans.

    Args:
        extra_series: Optional dict of {label: np.ndarray} for hip/shoulder.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    x = np.arange(len(wrist_y))

    # Phase shading
    for phase, (s, e) in phase_ranges.items():
        color = PHASE_COLORS.get(phase, "#888888")
        ax.axvspan(s, e, alpha=0.18, color=color)
        mid = (s + e) / 2
        ax.text(
            mid, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else np.nanmax(wrist_y) + 5,
            phase, ha="center", fontsize=7, color=color, alpha=0.9,
        )

    # Raw wrist
    ax.plot(x, wrist_y, color="#94a3b8", alpha=0.4, linewidth=1, label="Raw Wrist Y")
    # Smoothed wrist
    ax.plot(x, smoothed, color="#4ade80", linewidth=2.2, label="Smoothed Wrist Y")

    # Extra landmark series
    if extra_series:
        palette = ["#fb923c", "#38bdf8", "#f472b6"]
        for i, (lbl, series) in enumerate(extra_series.items()):
            ax.plot(x[: len(series)], series, color=palette[i % len(palette)],
                    linewidth=1.6, linestyle="--", label=lbl, alpha=0.85)

    # Swing window
    ax.axvline(swing_start, color="#facc15", linestyle="--", linewidth=1.2, label="Swing Start")
    ax.axvline(swing_end,   color="#f87171", linestyle="--", linewidth=1.2, label="Swing End")

    # Phase legend patches
    phase_patches = [
        Patch(facecolor=PHASE_COLORS[p], alpha=0.5, label=p) for p in phase_ranges
    ]

    legend1 = ax.legend(loc="upper left", fontsize=8, facecolor="#1e293b",
                        labelcolor="white", framealpha=0.8)
    ax.add_artist(legend1)
    ax.legend(handles=phase_patches, loc="upper right", fontsize=7,
              facecolor="#1e293b", labelcolor="white", framealpha=0.8, title="Phases",
              title_fontsize=7)

    ax.invert_yaxis()  # MediaPipe Y is flipped — invert so "up" = up
    ax.set_title(title, color="white", fontsize=13, pad=12)
    ax.set_xlabel("Frame", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Wrist Y (px, inverted)", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    fig.tight_layout()
    return fig
```

- Added `use_one_euro`, `fps`, `one_euro_min_cutoff`, `one_euro_beta` parameters
- When enabled, uses One-Euro filter instead of `uniform_filter1d` for the smoothed trajectory

---

### [export_utils.py](file:///Users/jialielu/highschoolproj/GolfPosePro/golf_pose_pro/export_utils.py)

```diff:export_utils.py
"""
export_utils.py — Export pose analysis results to CSV and JSON.
"""

import csv
import json
import io
from pathlib import Path


def export_to_csv(
    frame_data: list,
    phase_ranges: dict,
    output_path: str = None,
) -> bytes:
    """
    Export per-frame pose data with phase labels to CSV.

    Columns: frame_idx, phase, wrist_y, hip_y (opt), shoulder_y (opt)

    Returns bytes (UTF-8 CSV). Also writes to output_path if provided.
    """
    if not frame_data:
        return b""

    # Discover all keys in frame data
    sample = frame_data[0]
    landmark_keys = [k for k in sample if k != "frame_idx"]

    fieldnames = ["frame_idx", "phase"] + landmark_keys

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    for row in frame_data:
        frame_idx = row["frame_idx"]
        phase = _get_phase(frame_idx, phase_ranges)
        out_row = {"frame_idx": frame_idx, "phase": phase}
        for k in landmark_keys:
            v = row.get(k)
            out_row[k] = round(v, 3) if v is not None else ""
        writer.writerow(out_row)

    csv_bytes = buf.getvalue().encode("utf-8")

    if output_path:
        Path(output_path).write_bytes(csv_bytes)

    return csv_bytes


def export_to_json(
    frame_data: list,
    phase_ranges: dict,
    swing_start: int,
    swing_end: int,
    similarity_score: float = None,
    output_path: str = None,
) -> bytes:
    """
    Export full analysis as structured JSON.

    Schema:
    {
      "metadata": {...},
      "swing_window": {"start": int, "end": int},
      "similarity_score": float | null,
      "phases": {phase_name: {"start": int, "end": int, "duration_frames": int}},
      "frames": [{frame_idx, phase, wrist_y, ...}, ...]
    }
    """
    phases_summary = {}
    for name, (s, e) in phase_ranges.items():
        phases_summary[name] = {
            "start": int(s),
            "end":   int(e),
            "duration_frames": int(e - s),
        }

    frames_out = []
    for row in frame_data:
        frame_idx = row["frame_idx"]
        out = {
            "frame_idx": frame_idx,
            "phase": _get_phase(frame_idx, phase_ranges),
        }
        for k, v in row.items():
            if k != "frame_idx":
                out[k] = round(v, 3) if v is not None else None
        frames_out.append(out)

    payload = {
        "metadata": {
            "generator": "GolfPosePro",
            "total_frames": len(frame_data),
        },
        "swing_window": {
            "start": int(swing_start),
            "end":   int(swing_end),
            "duration_frames": int(swing_end - swing_start),
        },
        "similarity_score": similarity_score,
        "phases": phases_summary,
        "frames": frames_out,
    }

    json_bytes = json.dumps(payload, indent=2).encode("utf-8")

    if output_path:
        Path(output_path).write_bytes(json_bytes)

    return json_bytes


def export_3d_json(
    frames_3d: list,
    club_data: list,
    phase_ranges: dict,
    output_path: str = None,
) -> bytes:
    """
    Export full 3D landmark + club data as JSON for Three.js or external tools.

    Schema:
    {
      "metadata": {...},
      "phases": {phase_name: {"start": int, "end": int}},
      "frames": [{frame_idx, timestamp_ms, landmarks: [...], club: {...}}, ...]
    }
    """
    phases_summary = {}
    for name, (s, e) in phase_ranges.items():
        phases_summary[name] = {"start": int(s), "end": int(e)}

    frames_out = []
    for i, frame in enumerate(frames_3d):
        entry = {
            "frame_idx": frame["frame_idx"],
            "timestamp_ms": frame.get("timestamp_ms", 0),
            "landmarks": frame.get("landmarks", []),
        }
        if i < len(club_data):
            entry["club"] = club_data[i]
        frames_out.append(entry)

    payload = {
        "metadata": {
            "generator": "GolfPosePro",
            "format": "3d_landmarks",
            "total_frames": len(frames_3d),
        },
        "phases": phases_summary,
        "frames": frames_out,
    }

    json_bytes = json.dumps(payload).encode("utf-8")

    if output_path:
        Path(output_path).write_bytes(json_bytes)

    return json_bytes


# ─── helpers ─────────────────────────────────────────────────────────────────

def _get_phase(frame_idx: int, phase_ranges: dict) -> str:
    for name, (s, e) in phase_ranges.items():
        if s <= frame_idx <= e:
            return name
    return ""
===
"""
export_utils.py — Export pose analysis results to CSV and JSON.
"""

import csv
import json
import io
from pathlib import Path

import numpy as np


def export_to_csv(
    frame_data: list,
    phase_ranges: dict,
    output_path: str = None,
    velocity_data: dict = None,
    angle_data: dict = None,
    angular_velocity_data: dict = None,
) -> bytes:
    """
    Export per-frame pose data with phase labels to CSV.

    Columns: frame_idx, phase, wrist_x, wrist_y, hip_x (opt), hip_y (opt),
             shoulder_x (opt), shoulder_y (opt),
             wrist_velocity (opt), elbow_angle_L (opt), ...

    Returns bytes (UTF-8 CSV). Also writes to output_path if provided.
    """
    if not frame_data:
        return b""

    # Discover all keys in frame data
    sample = frame_data[0]
    landmark_keys = [k for k in sample if k != "frame_idx"]

    # Add velocity and angle column names
    extra_keys = []
    if velocity_data:
        extra_keys += sorted(velocity_data.keys())
    if angle_data:
        extra_keys += sorted(angle_data.keys())
    if angular_velocity_data:
        extra_keys += [f"{k}_angular_vel" for k in sorted(angular_velocity_data.keys())]

    fieldnames = ["frame_idx", "phase"] + landmark_keys + extra_keys

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    for i, row in enumerate(frame_data):
        frame_idx = row["frame_idx"]
        phase = _get_phase(frame_idx, phase_ranges)
        out_row = {"frame_idx": frame_idx, "phase": phase}
        for k in landmark_keys:
            v = row.get(k)
            out_row[k] = round(v, 3) if v is not None else ""
        # Velocity columns
        if velocity_data:
            for k, arr in velocity_data.items():
                out_row[k] = round(float(arr[i]), 2) if i < len(arr) else ""
        # Angle columns
        if angle_data:
            for k, arr in angle_data.items():
                val = arr[i] if i < len(arr) else np.nan
                out_row[k] = round(float(val), 2) if not np.isnan(val) else ""
        # Angular velocity columns
        if angular_velocity_data:
            for k, arr in angular_velocity_data.items():
                col = f"{k}_angular_vel"
                val = arr[i] if i < len(arr) else np.nan
                out_row[col] = round(float(val), 2) if not np.isnan(val) else ""
        writer.writerow(out_row)

    csv_bytes = buf.getvalue().encode("utf-8")

    if output_path:
        Path(output_path).write_bytes(csv_bytes)

    return csv_bytes


def export_to_json(
    frame_data: list,
    phase_ranges: dict,
    swing_start: int,
    swing_end: int,
    similarity_score: float = None,
    output_path: str = None,
    velocity_data: dict = None,
    angle_data: dict = None,
    angular_velocity_data: dict = None,
    peak_velocities: dict = None,
) -> bytes:
    """
    Export full analysis as structured JSON.

    Schema:
    {
      "metadata": {...},
      "swing_window": {"start": int, "end": int},
      "similarity_score": float | null,
      "phases": {phase_name: {"start": int, "end": int, "duration_frames": int}},
      "frames": [{frame_idx, phase, wrist_y, ...}, ...]
    }
    """
    phases_summary = {}
    for name, (s, e) in phase_ranges.items():
        phases_summary[name] = {
            "start": int(s),
            "end":   int(e),
            "duration_frames": int(e - s),
        }

    frames_out = []
    for i, row in enumerate(frame_data):
        frame_idx = row["frame_idx"]
        out = {
            "frame_idx": frame_idx,
            "phase": _get_phase(frame_idx, phase_ranges),
        }
        for k, v in row.items():
            if k != "frame_idx":
                out[k] = round(v, 3) if v is not None else None
        # Inline velocity and angle
        if velocity_data:
            for k, arr in velocity_data.items():
                out[k] = round(float(arr[i]), 2) if i < len(arr) else None
        if angle_data:
            for k, arr in angle_data.items():
                val = arr[i] if i < len(arr) else np.nan
                out[k] = round(float(val), 2) if not np.isnan(val) else None
        if angular_velocity_data:
            for k, arr in angular_velocity_data.items():
                val = arr[i] if i < len(arr) else np.nan
                out[f"{k}_angular_vel"] = round(float(val), 2) if not np.isnan(val) else None
        frames_out.append(out)

    payload = {
        "metadata": {
            "generator": "GolfPosePro",
            "total_frames": len(frame_data),
        },
        "swing_window": {
            "start": int(swing_start),
            "end":   int(swing_end),
            "duration_frames": int(swing_end - swing_start),
        },
        "similarity_score": similarity_score,
        "phases": phases_summary,
        "frames": frames_out,
    }

    json_bytes = json.dumps(payload, indent=2).encode("utf-8")

    # Add kinematics section if peak velocities are provided
    if peak_velocities:
        payload["kinematics"] = {
            "peak_velocities": {
                phase: {"peak_speed_px_per_s": round(v["peak_speed"], 2),
                        "peak_frame": v["peak_frame"]}
                for phase, v in peak_velocities.items()
            }
        }
        json_bytes = json.dumps(payload, indent=2).encode("utf-8")

    if output_path:
        Path(output_path).write_bytes(json_bytes)

    return json_bytes


def export_3d_json(
    frames_3d: list,
    club_data: list,
    phase_ranges: dict,
    output_path: str = None,
) -> bytes:
    """
    Export full 3D landmark + club data as JSON for Three.js or external tools.

    Schema:
    {
      "metadata": {...},
      "phases": {phase_name: {"start": int, "end": int}},
      "frames": [{frame_idx, timestamp_ms, landmarks: [...], club: {...}}, ...]
    }
    """
    phases_summary = {}
    for name, (s, e) in phase_ranges.items():
        phases_summary[name] = {"start": int(s), "end": int(e)}

    frames_out = []
    for i, frame in enumerate(frames_3d):
        entry = {
            "frame_idx": frame["frame_idx"],
            "timestamp_ms": frame.get("timestamp_ms", 0),
            "landmarks": frame.get("landmarks", []),
        }
        if i < len(club_data):
            entry["club"] = club_data[i]
        frames_out.append(entry)

    payload = {
        "metadata": {
            "generator": "GolfPosePro",
            "format": "3d_landmarks",
            "total_frames": len(frames_3d),
        },
        "phases": phases_summary,
        "frames": frames_out,
    }

    json_bytes = json.dumps(payload).encode("utf-8")

    if output_path:
        Path(output_path).write_bytes(json_bytes)

    return json_bytes


# ─── helpers ─────────────────────────────────────────────────────────────────

def _get_phase(frame_idx: int, phase_ranges: dict) -> str:
    for name, (s, e) in phase_ranges.items():
        if s <= frame_idx <= e:
            return name
    return ""
```

- `export_to_csv()` — new columns for velocity, angle, and angular velocity data
- `export_to_json()` — inline velocity/angle per frame + `kinematics.peak_velocities` section

---

### [app.py](file:///Users/jialielu/highschoolproj/GolfPosePro/app.py)

```diff:app.py
"""
GolfPosePro — Streamlit Web Application
A premium, locally-runnable golf swing analyzer.

Run with:
    streamlit run app.py
"""

import io
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ── Internal package imports ────────────────────────────────────────────────
from golf_pose_pro.pose_extraction import (
    add_silent_audio,
    extract_pose_features,
    extract_full_3d_landmarks,
    extract_y_series,
    get_video_info,
)
from golf_pose_pro.phase_detection import detect_swing_phases, plot_trajectory, PHASE_COLORS
from golf_pose_pro.comparison import compare_swing_phases
from golf_pose_pro.dtw_utils import align_phase_frames, compute_similarity_score
from golf_pose_pro.video_gen import generate_debug_video
from golf_pose_pro.export_utils import export_to_csv, export_to_json, export_3d_json
from golf_pose_pro.club_estimation import estimate_club_positions
from golf_pose_pro.threejs_component import build_3d_viewer_html, build_live_tracking_html


# ────────────────────────────────────────────────────────────────────────────
# Page config & global CSS
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GolfPosePro",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* ---- Background ---- */
.stApp {
    background: linear-gradient(135deg, #030712 0%, #0a1628 50%, #061a10 100%);
    min-height: 100vh;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: rgba(10, 22, 40, 0.95) !important;
    border-right: 1px solid rgba(74, 222, 128, 0.15);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ---- Hero header ---- */
.hero-header {
    background: linear-gradient(135deg, rgba(6,26,16,0.9), rgba(10,22,40,0.9));
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 60px rgba(74,222,128,0.08), 0 20px 60px rgba(0,0,0,0.4);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4ade80, #22d3ee, #4ade80);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s ease infinite;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.4rem;
    font-weight: 300;
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ---- Metric cards ---- */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(74,222,128,0.18);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(74,222,128,0.15);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #4ade80;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.metric-sub {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}

/* ---- Phase timeline ---- */
.phase-row {
    display: flex;
    border-radius: 10px;
    overflow: hidden;
    height: 28px;
    margin: 0.6rem 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.phase-seg {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: rgba(255,255,255,0.9);
    overflow: hidden;
    white-space: nowrap;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    transition: flex 0.5s ease;
}

/* ---- Section cards ---- */
.section-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(22,163,74,0.3) !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(22,163,74,0.45) !important;
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
}
.stDownloadButton > button {
    background: rgba(71,85,105,0.5) !important;
    border: 1px solid rgba(148,163,184,0.25) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
}

/* ---- Tabs ---- */
.stTabs [role="tablist"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [role="tab"] {
    border-radius: 9px;
    color: #64748b !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stTabs [role="tab"][aria-selected="true"] {
    background: rgba(74,222,128,0.12) !important;
    color: #4ade80 !important;
    font-weight: 600 !important;
}

/* ---- Progress ---- */
.stProgress > div > div {
    background: linear-gradient(90deg, #16a34a, #4ade80) !important;
    border-radius: 4px !important;
}

/* ---- Info / Alerts ---- */
.stAlert {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
}

/* ---- Inputs ---- */
.stSlider > div > div > div { background: #16a34a !important; }
.stSelectbox div[data-baseweb], .stMultiSelect div[data-baseweb] {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(74,222,128,0.2) !important;
}

/* ---- Divider ---- */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(74,222,128,0.3); border-radius: 3px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

PHASE_COLORS = {
    "Address":       "#6c757d",
    "Backswing":     "#3a86ff",
    "Top":           "#8338ec",
    "Downswing":     "#ff006e",
    "Impact":        "#fb5607",
    "Follow Through":"#06d6a0",
}


def _save_upload(uploaded_file, suffix: str = ".mp4") -> str:
    """Write a Streamlit UploadedFile to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def _fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def _phase_timeline_html(phase_ranges: dict, total_frames: int) -> str:
    """Build an HTML phase-timeline bar."""
    segs = ""
    for phase, (s, e) in phase_ranges.items():
        pct   = max(1.0, (e - s) / total_frames * 100)
        color = PHASE_COLORS.get(phase, "#888")
        segs += (
            f'<div class="phase-seg" style="flex:{pct:.2f};background:{color};">'
            f'{phase}</div>'
        )
    return f'<div class="phase-row">{segs}</div>'


def _metric_card(value, label, sub="") -> str:
    return (
        f'<div class="metric-card">'
        f'  <div class="metric-value">{value}</div>'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="metric-sub">{sub}</div>'
        f'</div>'
    )


def _section_title(icon: str, title: str) -> str:
    return (
        f'<div class="section-title">'
        f'  <span style="font-size:1.2rem">{icon}</span> {title}'
        f'</div>'
    )


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⛳ GolfPosePro")
    st.markdown("---")

    st.markdown("### 📁 Video Inputs")

    student_file = st.file_uploader(
        "Your Swing (student)",
        type=["mp4", "mov", "MP4", "MOV"],
        help="Upload your golf swing video.",
        key="student_upload",
    )

    # Built-in pro references
    PROJ_DIR  = Path(__file__).parent
    BUILTIN   = {}
    for name, path in [
        ("Max Homa — Iron",       PROJ_DIR / "input_videos" / "max_homa_iron_fixed.MP4"),
        ("Ludvig Åberg — Driver", PROJ_DIR / "input_videos" / "ludvig_aberg_driver_fixed.MP4"),
    ]:
        if path.exists():
            BUILTIN[name] = str(path)

    pro_source = st.radio(
        "Pro Reference",
        ["Upload custom", "Use built-in"],
        horizontal=True,
    )

    pro_file = None
    builtin_pro_path = None

    if pro_source == "Upload custom":
        pro_file = st.file_uploader(
            "Pro Reference Video",
            type=["mp4", "mov", "MP4", "MOV"],
            key="pro_upload",
        )
    elif BUILTIN:
        chosen_pro = st.selectbox("Choose built-in pro", list(BUILTIN.keys()))
        builtin_pro_path = BUILTIN[chosen_pro]
    else:
        st.warning("No built-in pro videos found.")

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")

    track_options = st.multiselect(
        "Track landmarks",
        ["wrist", "hip", "shoulder"],
        default=["wrist"],
        help="Select which body landmarks to track.",
    )
    if not track_options:
        track_options = ["wrist"]

    smoothing = st.slider("Smoothing window", 3, 20, 5, 1,
                          help="Bigger = smoother trajectory, less sensitive.")
    threshold_pct = st.slider("Motion threshold (%ile)", 70, 99, 90, 1,
                              help="Higher = only stronger motions trigger phase start.")

    st.markdown("---")
    st.markdown("### 🎬 Video Settings")

    slow_factor = st.slider(
        "Slow-motion factor",
        1.0, 4.0, 2.0, 0.5,
        help="How much to slow down the swing segment in the debug video.",
    )
    show_pose = st.checkbox("Show pose skeleton in comparison", value=True)
    use_dtw   = st.checkbox("Use DTW-aligned frame matching", value=True,
                            help="Use Dynamic Time Warping to match corresponding "
                                 "frames between student and pro.")

    st.markdown("---")
    analyze_btn = st.button("🚀 Run Analysis", use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# Hero header
# ────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <p class="hero-title">⛳ GolfPosePro</p>
  <p class="hero-sub">
    AI-powered golf swing analyzer · Pose tracking · Phase detection ·
    Pro comparison · DTW alignment
  </p>
</div>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state["results"] = None


# ────────────────────────────────────────────────────────────────────────────
# Main analysis pipeline
# ────────────────────────────────────────────────────────────────────────────

if analyze_btn:
    if student_file is None:
        st.error("⚠️  Please upload your student swing video first.")
        st.stop()

    # ── Save uploads to temp files ────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="golfposepro_")

    student_raw  = os.path.join(tmp_dir, "student_raw.mp4")
    student_path = os.path.join(tmp_dir, "student_fixed.mp4")
    with open(student_raw, "wb") as f:
        f.write(student_file.getbuffer())

    if pro_file is not None:
        pro_raw  = os.path.join(tmp_dir, "pro_raw.mp4")
        pro_path = os.path.join(tmp_dir, "pro_fixed.mp4")
        with open(pro_raw, "wb") as f:
            f.write(pro_file.getbuffer())
    else:
        pro_raw  = builtin_pro_path
        pro_path = os.path.join(tmp_dir, "pro_fixed.mp4")

    has_pro = (pro_raw is not None)

    # ── Status UI ────────────────────────────────────────────────────────
    progress_bar  = st.progress(0, text="Starting analysis…")
    status_text   = st.empty()

    def update_progress(pct: int, msg: str):
        progress_bar.progress(pct, text=msg)
        status_text.markdown(f"*{msg}*")

    # ── Step 1: Audio fix ─────────────────────────────────────────────────
    update_progress(5, "🔇 Adding silent audio track…")
    add_silent_audio(student_raw, student_path)
    if has_pro:
        add_silent_audio(pro_raw, pro_path)

    # ── Step 2: Pose extraction ───────────────────────────────────────────
    update_progress(10, "🧠 Extracting pose landmarks — student video…")

    frame_count_s = get_video_info(student_path)["total_frames"]

    def student_progress(f, t):
        pct = 10 + int(30 * f / max(t, 1))
        update_progress(pct, f"🧠 Pose extraction — student: frame {f}/{t}")

    frame_data_student = extract_pose_features(
        student_path,
        track_landmarks=track_options,
        progress_callback=student_progress,
    )

    if has_pro:
        update_progress(40, "🧠 Extracting pose landmarks — pro video…")

        def pro_progress(f, t):
            pct = 40 + int(20 * f / max(t, 1))
            update_progress(pct, f"🧠 Pose extraction — pro: frame {f}/{t}")

        frame_data_pro = extract_pose_features(
            pro_path,
            track_landmarks=track_options,
            progress_callback=pro_progress,
        )
    else:
        frame_data_pro = None

    # ── Step 3: Phase detection ───────────────────────────────────────────
    update_progress(62, "🔍 Detecting swing phases — student…")
    signal_key = f"{track_options[0]}_y"
    (phase_ranges_s, swing_start_s, swing_end_s, wrist_y_s, smoothed_s) = \
        detect_swing_phases(frame_data_student, signal_key=signal_key,
                            smoothing_window=smoothing,
                            threshold_percentile=threshold_pct)

    phase_ranges_p = swing_start_p = swing_end_p = wrist_y_p = smoothed_p = None
    if has_pro and frame_data_pro:
        update_progress(65, "🔍 Detecting swing phases — pro…")
        (phase_ranges_p, swing_start_p, swing_end_p, wrist_y_p, smoothed_p) = \
            detect_swing_phases(frame_data_pro, signal_key=signal_key,
                                smoothing_window=smoothing,
                                threshold_percentile=threshold_pct)

    # ── Step 4: DTW alignment + similarity ──────────────────────────────
    dtw_alignment    = None
    similarity_score = None

    if has_pro and phase_ranges_p is not None and use_dtw:
        update_progress(68, "📐 Computing DTW alignment…")
        dtw_alignment = align_phase_frames(
            phase_ranges_s, phase_ranges_p, smoothed_s, smoothed_p
        )
        similarity_score = compute_similarity_score(
            smoothed_s, smoothed_p,
            swing_start_s, swing_end_s,
            swing_start_p, swing_end_p,
        )

    elif has_pro and phase_ranges_p is not None:
        # Simple midpoint alignment
        dtw_alignment = {
            phase: ((s + e) // 2, (phase_ranges_p[phase][0] + phase_ranges_p[phase][1]) // 2, None)
            for phase, (s, e) in phase_ranges_s.items()
            if phase in phase_ranges_p
        }

    # ── Step 5: Build extra series dict ─────────────────────────────────
    extra_series_s = {}
    for lm in track_options:
        if lm != track_options[0]:            # wrist is primary
            key = f"{lm}_y"
            arr = extract_y_series(frame_data_student, key)
            if not np.all(np.isnan(arr)):
                extra_series_s[lm.capitalize()] = arr

    # ── Step 6: Comparison figure ────────────────────────────────────────
    comparison_fig = None
    if has_pro and phase_ranges_p is not None:
        update_progress(72, "📸 Generating phase comparison grid…")
        comparison_fig = compare_swing_phases(
            student_video_path=student_path,
            student_phases=phase_ranges_s,
            pro_video_path=pro_path,
            pro_phases=phase_ranges_p,
            dtw_alignment=dtw_alignment,
            show_pose=show_pose,
        )

    # ── Step 6b: 3D landmark extraction ─────────────────────────────────
    update_progress(75, "🎯 Extracting 3D landmarks — student…")
    frames_3d_s = extract_full_3d_landmarks(student_path)
    club_data_s = estimate_club_positions(frames_3d_s, phase_ranges_s)

    frames_3d_p = None
    club_data_p = None
    if has_pro:
        update_progress(76, "🎯 Extracting 3D landmarks — pro…")
        frames_3d_p = extract_full_3d_landmarks(pro_path)
        club_data_p = estimate_club_positions(frames_3d_p, phase_ranges_p)

    json_3d_bytes = export_3d_json(frames_3d_s, club_data_s, phase_ranges_s)

    # ── Step 7: Debug video ──────────────────────────────────────────────
    update_progress(78, "🎬 Generating debug video…")
    debug_output = os.path.join(tmp_dir, "debug.mp4")

    def video_progress(f, t):
        pct = 78 + int(18 * f / max(t, 1))
        update_progress(pct, f"🎬 Rendering debug video: frame {f}/{t}")

    playable_path = generate_debug_video(
        video_path=student_path,
        output_path=debug_output,
        wrist_y=wrist_y_s,
        smoothed=smoothed_s,
        phase_ranges=phase_ranges_s,
        swing_start=swing_start_s,
        swing_end=swing_end_s,
        slow_factor=slow_factor,
        extra_series=extra_series_s if extra_series_s else None,
        progress_callback=video_progress,
    )

    # ── Step 8: Export data ──────────────────────────────────────────────
    update_progress(97, "📄 Preparing export data…")
    csv_bytes  = export_to_csv(frame_data_student, phase_ranges_s)
    json_bytes = export_to_json(
        frame_data_student, phase_ranges_s,
        swing_start_s, swing_end_s,
        similarity_score=similarity_score,
    )

    # Load debug video bytes
    video_bytes = None
    if os.path.exists(playable_path):
        with open(playable_path, "rb") as f:
            video_bytes = f.read()

    # ── Store in session state ────────────────────────────────────────────
    update_progress(100, "✅ Analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    st.session_state["results"] = {
        # Student data
        "frame_data_s":   frame_data_student,
        "phase_ranges_s": phase_ranges_s,
        "swing_start_s":  swing_start_s,
        "swing_end_s":    swing_end_s,
        "wrist_y_s":      wrist_y_s,
        "smoothed_s":     smoothed_s,
        "extra_series_s": extra_series_s,
        "total_frames_s": len(frame_data_student),
        # Pro data
        "has_pro":        has_pro,
        "phase_ranges_p": phase_ranges_p,
        # DTW
        "dtw_alignment":     dtw_alignment,
        "similarity_score":  similarity_score,
        # Outputs
        "comparison_fig": comparison_fig,
        "video_bytes":    video_bytes,
        "csv_bytes":      csv_bytes,
        "json_bytes":     json_bytes,
        # 3D data
        "frames_3d_s":   frames_3d_s,
        "club_data_s":   club_data_s,
        "frames_3d_p":   frames_3d_p,
        "club_data_p":   club_data_p,
        "json_3d_bytes": json_3d_bytes,
        # Settings echo
        "track_options":  track_options,
        "slow_factor":    slow_factor,
        "use_dtw":        use_dtw,
    }
    st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# Results display
# ────────────────────────────────────────────────────────────────────────────

R = st.session_state.get("results")

if R is None:
    # Welcome state
    st.markdown("""
    <div class="section-card" style="text-align:center;padding:3rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">⛳</div>
        <h2 style="color:#e2e8f0;margin-bottom:0.5rem;">Welcome to GolfPosePro</h2>
        <p style="color:#64748b;max-width:500px;margin:auto;line-height:1.7;">
            Upload your golf swing video in the sidebar, configure analysis settings,
            and click <strong style="color:#4ade80">Run Analysis</strong> to get started.
        </p>
        <div style="display:flex;gap:1.5rem;justify-content:center;margin-top:2rem;flex-wrap:wrap;">
            <div style="color:#4ade80;font-size:0.85rem;">🧠 MediaPipe Pose</div>
            <div style="color:#38bdf8;font-size:0.85rem;">📐 DTW Alignment</div>
            <div style="color:#f472b6;font-size:0.85rem;">🦴 Multi-landmark</div>
            <div style="color:#fb923c;font-size:0.85rem;">🎬 Debug Video</div>
            <div style="color:#a78bfa;font-size:0.85rem;">📄 CSV / JSON Export</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Top KPI row ───────────────────────────────────────────────────────────────

phase_ranges_s  = R["phase_ranges_s"]
swing_start_s   = R["swing_start_s"]
swing_end_s     = R["swing_end_s"]
total_frames_s  = R["total_frames_s"]
similarity_score = R["similarity_score"]
use_dtw         = R["use_dtw"]

swing_duration  = swing_end_s - swing_start_s
num_phases      = len(phase_ranges_s)

kpi_cols = st.columns(4)
kpi_data = [
    (f"{num_phases}", "PHASES DETECTED", "Address → Follow Through"),
    (f"{swing_duration}", "SWING FRAMES", f"Frames {swing_start_s}–{swing_end_s}"),
    (f"{total_frames_s}", "TOTAL FRAMES", f"{total_frames_s} analyzed"),
    (
        f"{similarity_score:.0f}%" if similarity_score is not None else "N/A",
        "SWING SIMILARITY",
        "DTW vs. Pro" if similarity_score is not None else "No pro video",
    ),
]
for col, (val, label, sub) in zip(kpi_cols, kpi_data):
    col.markdown(_metric_card(val, label, sub), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Phase timeline ────────────────────────────────────────────────────────────

st.markdown(_section_title("📊", "Swing Phase Timeline"), unsafe_allow_html=True)
st.markdown(
    _phase_timeline_html(phase_ranges_s, total_frames_s),
    unsafe_allow_html=True,
)

# Phase table
phase_table_rows = ""
for phase, (s, e) in phase_ranges_s.items():
    color   = PHASE_COLORS.get(phase, "#888")
    dur     = e - s
    pct     = dur / total_frames_s * 100
    dtw_tag = ""
    if use_dtw and R["dtw_alignment"] and phase in R["dtw_alignment"]:
        d = R["dtw_alignment"][phase][2]
        if d is not None:
            dtw_tag = f'<span style="color:#4ade80;font-size:0.7rem">DTW: {d:.1f}</span>'
    phase_table_rows += (
        f"<tr>"
        f"  <td><span style='color:{color};font-weight:600'>{phase}</span></td>"
        f"  <td style='color:#94a3b8'>{s}</td>"
        f"  <td style='color:#94a3b8'>{e}</td>"
        f"  <td style='color:#e2e8f0'>{dur}</td>"
        f"  <td style='color:#64748b'>{pct:.1f}%</td>"
        f"  <td>{dtw_tag}</td>"
        f"</tr>"
    )

st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:0.85rem;margin-top:0.5rem;">
  <thead>
    <tr style="color:#64748b;border-bottom:1px solid rgba(255,255,255,0.08);">
      <th style="text-align:left;padding:6px 8px">Phase</th>
      <th style="text-align:left;padding:6px 8px">Start</th>
      <th style="text-align:left;padding:6px 8px">End</th>
      <th style="text-align:left;padding:6px 8px">Frames</th>
      <th style="text-align:left;padding:6px 8px">% of clip</th>
      <th style="text-align:left;padding:6px 8px">DTW</th>
    </tr>
  </thead>
  <tbody>
    {phase_table_rows}
  </tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_trajectory, tab_comparison, tab_3d, tab_live, tab_video, tab_export = st.tabs([
    "📈 Trajectory",
    "🔄 Phase Comparison",
    "🎯 3D Viewer",
    "📹 Live Tracking",
    "🎬 Debug Video",
    "📄 Export Data",
])


# ─── Tab 1: Trajectory ───────────────────────────────────────────────────────

with tab_trajectory:
    st.markdown(_section_title("📈", "Wrist Y-Trajectory with Phase Overlay"),
                unsafe_allow_html=True)

    # Build extra_series for multi-landmark
    extra_series_s = R.get("extra_series_s", {})

    traj_fig = plot_trajectory(
        wrist_y=R["wrist_y_s"],
        smoothed=R["smoothed_s"],
        phase_ranges=phase_ranges_s,
        swing_start=swing_start_s,
        swing_end=swing_end_s,
        title="Student Swing — Wrist Y-Trajectory",
        extra_series=extra_series_s if extra_series_s else None,
        figsize=(14, 5),
    )
    st.pyplot(traj_fig, use_container_width=True)
    plt.close(traj_fig)

    traj_bytes = _fig_to_bytes(traj_fig)
    st.download_button(
        "⬇️  Download trajectory chart (PNG)",
        data=traj_bytes,
        file_name="wrist_trajectory.png",
        mime="image/png",
    )

    if R["has_pro"] and R["phase_ranges_p"] and R.get("smoothed_p") is not None: # type: ignore
        pass  # Could add pro trajectory here too


# ─── Tab 2: Comparison ───────────────────────────────────────────────────────

with tab_comparison:
    if not R["has_pro"] or R["comparison_fig"] is None:
        st.info("📌 Upload a pro reference video to enable phase comparison.")
    else:
        dtw_badge = ""
        if use_dtw and R["dtw_alignment"]:
            dtw_badge = (
                '<span style="background:rgba(74,222,128,0.15);color:#4ade80;'
                'border:1px solid rgba(74,222,128,0.3);border-radius:6px;'
                'padding:2px 10px;font-size:0.78rem;font-weight:600;">DTW ALIGNED</span>'
            )
        st.markdown(
            f'{_section_title("🔄", "Phase-by-Phase Comparison")} {dtw_badge}',
            unsafe_allow_html=True,
        )

        if similarity_score is not None:
            score_color = (
                "#4ade80" if similarity_score >= 70 else
                "#facc15" if similarity_score >= 40 else "#f87171"
            )
            st.markdown(
                f'<div style="text-align:center;margin-bottom:1rem;">'
                f'  <span style="font-size:2rem;font-weight:700;color:{score_color}">'
                f'{similarity_score:.0f}%</span>'
                f'  <span style="color:#64748b;font-size:0.9rem;margin-left:0.5rem">'
                f'DTW swing similarity score</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        comp_fig = R["comparison_fig"]
        st.pyplot(comp_fig, use_container_width=True)

        comp_bytes = _fig_to_bytes(comp_fig)
        st.download_button(
            "⬇️  Download comparison grid (PNG)",
            data=comp_bytes,
            file_name="swing_phase_comparison.png",
            mime="image/png",
        )


# ─── Tab 3: 3D Viewer ────────────────────────────────────────────────────────

with tab_3d:
    st.markdown(_section_title("🎯", "3D Swing Viewer"), unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#94a3b8;font-size:0.85rem;margin-bottom:1rem;">
        Interactive 3D skeleton with estimated golf club. Drag to rotate, scroll to zoom.
        Phase colors update as the animation plays.
    </p>
    """, unsafe_allow_html=True)

    frames_3d_s = R.get("frames_3d_s")
    club_data_s = R.get("club_data_s")

    if frames_3d_s and club_data_s:
        is_comparison = R["has_pro"] and R.get("frames_3d_p") is not None
        viewer_html = build_3d_viewer_html(
            frames_3d=frames_3d_s,
            club_data=club_data_s,
            phase_ranges=phase_ranges_s,
            phase_colors=PHASE_COLORS,
            mode="comparison" if is_comparison else "replay",
            comparison_pose=R.get("frames_3d_p"),
            comparison_club=R.get("club_data_p"),
            dtw_alignment=R.get("dtw_alignment"),
            height=600,
        )
        st.components.v1.html(viewer_html, height=650, scrolling=False)
    else:
        st.info("📌 3D data not available. Re-run analysis to generate 3D landmarks.")


# ─── Tab 4: Live Tracking ────────────────────────────────────────────────────

with tab_live:
    st.markdown(_section_title("📹", "Live Webcam Tracking"), unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#94a3b8;font-size:0.85rem;margin-bottom:1rem;">
        Real-time pose tracking using your webcam. MediaPipe runs directly in the browser —
        no video upload needed. Click <strong style="color:#4ade80">Start Tracking</strong> to begin.
    </p>
    """, unsafe_allow_html=True)
    live_html = build_live_tracking_html(height=600)
    st.components.v1.html(live_html, height=650, scrolling=False)


# ─── Tab 5: Debug Video ───────────────────────────────────────────────────────

with tab_video:
    st.markdown(_section_title("🎬", "Annotated Debug Video"), unsafe_allow_html=True)

    video_bytes = R.get("video_bytes")
    if video_bytes:
        st.markdown(f"""
        <div style="color:#64748b;font-size:0.85rem;margin-bottom:1rem;">
            Slow-motion factor: <strong style="color:#4ade80">{R['slow_factor']}×</strong>
            applied to swing window (frames {swing_start_s}–{swing_end_s}).
        </div>
        """, unsafe_allow_html=True)
        st.video(io.BytesIO(video_bytes))
        st.download_button(
            "⬇️  Download debug video (MP4)",
            data=video_bytes,
            file_name="golf_debug_video.mp4",
            mime="video/mp4",
        )
    else:
        st.warning("⚠️  Debug video could not be generated. Check that ffmpeg is installed.")


# ─── Tab 6: Export ───────────────────────────────────────────────────────────

with tab_export:
    st.markdown(_section_title("📄", "Export Analysis Data"), unsafe_allow_html=True)

    col_csv, col_json, col_3d = st.columns(3)

    with col_csv:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📊 CSV Export</div>
            <p style="color:#94a3b8;font-size:0.85rem;line-height:1.6;">
                Per-frame data with phase labels.<br>
                Columns: <code>frame_idx</code>, <code>phase</code>,
                <code>wrist_y</code>, <code>hip_y</code>*, <code>shoulder_y</code>*
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            "⬇️  Download CSV",
            data=R["csv_bytes"],
            file_name="golf_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_json:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📦 JSON Export</div>
            <p style="color:#94a3b8;font-size:0.85rem;line-height:1.6;">
                Full structured analysis including metadata,
                phase boundaries, swing window, similarity score, and all frames.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            "⬇️  Download JSON",
            data=R["json_bytes"],
            file_name="golf_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_3d:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🎯 3D JSON Export</div>
            <p style="color:#94a3b8;font-size:0.85rem;line-height:1.6;">
                Full 3D landmark data with golf club estimation.
                Use with Three.js or other 3D tools.
            </p>
        </div>
        """, unsafe_allow_html=True)
        json_3d = R.get("json_3d_bytes", b"")
        if json_3d:
            st.download_button(
                "⬇️  Download 3D JSON",
                data=json_3d,
                file_name="golf_3d_landmarks.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("3D data not available.")

    # Preview first 10 rows of CSV
    st.markdown("##### Preview (first 10 frames)")
    csv_preview = R["csv_bytes"].decode("utf-8")
    lines   = csv_preview.split("\n")
    preview = "\n".join(lines[:11])
    st.code(preview, language="csv")
===
"""
GolfPosePro — Streamlit Web Application
A premium, locally-runnable golf swing analyzer.

Run with:
    streamlit run app.py
"""

import io
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ── Internal package imports ────────────────────────────────────────────────
from golf_pose_pro.pose_extraction import (
    add_silent_audio,
    extract_pose_features,
    extract_full_3d_landmarks,
    extract_y_series,
    get_video_info,
)
from golf_pose_pro.phase_detection import detect_swing_phases, plot_trajectory, PHASE_COLORS
from golf_pose_pro.comparison import compare_swing_phases
from golf_pose_pro.dtw_utils import align_phase_frames, compute_similarity_score
from golf_pose_pro.video_gen import generate_debug_video
from golf_pose_pro.export_utils import export_to_csv, export_to_json, export_3d_json
from golf_pose_pro.club_estimation import estimate_club_positions
from golf_pose_pro.threejs_component import build_3d_viewer_html, build_live_tracking_html
from golf_pose_pro.kinematics import (
    compute_all_joint_angles,
    compute_velocities_from_frame_data,
    compute_angular_velocity,
    find_peak_velocities,
)
from golf_pose_pro.smoothing import smooth_series


# ────────────────────────────────────────────────────────────────────────────
# Page config & global CSS
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GolfPosePro",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* ---- Background ---- */
.stApp {
    background: linear-gradient(135deg, #030712 0%, #0a1628 50%, #061a10 100%);
    min-height: 100vh;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: rgba(10, 22, 40, 0.95) !important;
    border-right: 1px solid rgba(74, 222, 128, 0.15);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ---- Hero header ---- */
.hero-header {
    background: linear-gradient(135deg, rgba(6,26,16,0.9), rgba(10,22,40,0.9));
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 60px rgba(74,222,128,0.08), 0 20px 60px rgba(0,0,0,0.4);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4ade80, #22d3ee, #4ade80);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s ease infinite;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.4rem;
    font-weight: 300;
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ---- Metric cards ---- */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(74,222,128,0.18);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(74,222,128,0.15);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #4ade80;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.metric-sub {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}

/* ---- Phase timeline ---- */
.phase-row {
    display: flex;
    border-radius: 10px;
    overflow: hidden;
    height: 28px;
    margin: 0.6rem 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.phase-seg {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: rgba(255,255,255,0.9);
    overflow: hidden;
    white-space: nowrap;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    transition: flex 0.5s ease;
}

/* ---- Section cards ---- */
.section-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(22,163,74,0.3) !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(22,163,74,0.45) !important;
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
}
.stDownloadButton > button {
    background: rgba(71,85,105,0.5) !important;
    border: 1px solid rgba(148,163,184,0.25) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
}

/* ---- Tabs ---- */
.stTabs [role="tablist"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [role="tab"] {
    border-radius: 9px;
    color: #64748b !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stTabs [role="tab"][aria-selected="true"] {
    background: rgba(74,222,128,0.12) !important;
    color: #4ade80 !important;
    font-weight: 600 !important;
}

/* ---- Progress ---- */
.stProgress > div > div {
    background: linear-gradient(90deg, #16a34a, #4ade80) !important;
    border-radius: 4px !important;
}

/* ---- Info / Alerts ---- */
.stAlert {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
}

/* ---- Inputs ---- */
.stSlider > div > div > div { background: #16a34a !important; }
.stSelectbox div[data-baseweb], .stMultiSelect div[data-baseweb] {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(74,222,128,0.2) !important;
}

/* ---- Divider ---- */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(74,222,128,0.3); border-radius: 3px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

PHASE_COLORS = {
    "Address":       "#6c757d",
    "Backswing":     "#3a86ff",
    "Top":           "#8338ec",
    "Downswing":     "#ff006e",
    "Impact":        "#fb5607",
    "Follow Through":"#06d6a0",
}


def _save_upload(uploaded_file, suffix: str = ".mp4") -> str:
    """Write a Streamlit UploadedFile to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def _fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def _phase_timeline_html(phase_ranges: dict, total_frames: int) -> str:
    """Build an HTML phase-timeline bar."""
    segs = ""
    for phase, (s, e) in phase_ranges.items():
        pct   = max(1.0, (e - s) / total_frames * 100)
        color = PHASE_COLORS.get(phase, "#888")
        segs += (
            f'<div class="phase-seg" style="flex:{pct:.2f};background:{color};">'
            f'{phase}</div>'
        )
    return f'<div class="phase-row">{segs}</div>'


def _metric_card(value, label, sub="") -> str:
    return (
        f'<div class="metric-card">'
        f'  <div class="metric-value">{value}</div>'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="metric-sub">{sub}</div>'
        f'</div>'
    )


def _section_title(icon: str, title: str) -> str:
    return (
        f'<div class="section-title">'
        f'  <span style="font-size:1.2rem">{icon}</span> {title}'
        f'</div>'
    )


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⛳ GolfPosePro")
    st.markdown("---")

    st.markdown("### 📁 Video Inputs")

    student_file = st.file_uploader(
        "Your Swing (student)",
        type=["mp4", "mov", "MP4", "MOV"],
        help="Upload your golf swing video.",
        key="student_upload",
    )

    # Built-in pro references
    PROJ_DIR  = Path(__file__).parent
    BUILTIN   = {}
    for name, path in [
        ("Max Homa — Iron",       PROJ_DIR / "input_videos" / "max_homa_iron_fixed.MP4"),
        ("Ludvig Åberg — Driver", PROJ_DIR / "input_videos" / "ludvig_aberg_driver_fixed.MP4"),
    ]:
        if path.exists():
            BUILTIN[name] = str(path)

    pro_source = st.radio(
        "Pro Reference",
        ["Upload custom", "Use built-in"],
        horizontal=True,
    )

    pro_file = None
    builtin_pro_path = None

    if pro_source == "Upload custom":
        pro_file = st.file_uploader(
            "Pro Reference Video",
            type=["mp4", "mov", "MP4", "MOV"],
            key="pro_upload",
        )
    elif BUILTIN:
        chosen_pro = st.selectbox("Choose built-in pro", list(BUILTIN.keys()))
        builtin_pro_path = BUILTIN[chosen_pro]
    else:
        st.warning("No built-in pro videos found.")

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")

    track_options = st.multiselect(
        "Track landmarks",
        ["wrist", "hip", "shoulder"],
        default=["wrist"],
        help="Select which body landmarks to track.",
    )
    if not track_options:
        track_options = ["wrist"]

    smoothing = st.slider("Smoothing window", 3, 20, 5, 1,
                          help="Bigger = smoother trajectory, less sensitive.")
    threshold_pct = st.slider("Motion threshold (%ile)", 70, 99, 90, 1,
                              help="Higher = only stronger motions trigger phase start.")

    st.markdown("---")
    st.markdown("### 🎬 Video Settings")

    slow_factor = st.slider(
        "Slow-motion factor",
        1.0, 4.0, 2.0, 0.5,
        help="How much to slow down the swing segment in the debug video.",
    )
    show_pose = st.checkbox("Show pose skeleton in comparison", value=True)
    use_dtw   = st.checkbox("Use DTW-aligned frame matching", value=True,
                            help="Use Dynamic Time Warping to match corresponding "
                                 "frames between student and pro.")

    st.markdown("---")
    st.markdown("### 🎯 Smoothing & Kinematics")

    use_one_euro = st.checkbox(
        "Use adaptive smoothing (One-Euro Filter)",
        value=True,
        help="Adaptive low-pass filter: heavy smoothing for slow motion, "
             "light smoothing for fast motion. Reduces landmark jitter.",
    )

    if use_one_euro:
        oe_min_cutoff = st.slider(
            "Min cutoff (smoothness)", 0.3, 5.0, 1.0, 0.1,
            help="Lower = more smoothing for slow signals. Typical: 0.5–3.0.",
        )
        oe_beta = st.slider(
            "Beta (speed adaptation)", 0.0, 0.1, 0.007, 0.001,
            help="Higher = cutoff adapts faster to speed changes. Typical: 0.001–0.05.",
            format="%.3f",
        )
    else:
        oe_min_cutoff = 1.0
        oe_beta = 0.007

    st.markdown("---")
    analyze_btn = st.button("🚀 Run Analysis", use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# Hero header
# ────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <p class="hero-title">⛳ GolfPosePro</p>
  <p class="hero-sub">
    AI-powered golf swing analyzer · Pose tracking · Phase detection ·
    Pro comparison · DTW alignment
  </p>
</div>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state["results"] = None


# ────────────────────────────────────────────────────────────────────────────
# Main analysis pipeline
# ────────────────────────────────────────────────────────────────────────────

if analyze_btn:
    if student_file is None:
        st.error("⚠️  Please upload your student swing video first.")
        st.stop()

    # ── Save uploads to temp files ────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="golfposepro_")

    student_raw  = os.path.join(tmp_dir, "student_raw.mp4")
    student_path = os.path.join(tmp_dir, "student_fixed.mp4")
    with open(student_raw, "wb") as f:
        f.write(student_file.getbuffer())

    if pro_file is not None:
        pro_raw  = os.path.join(tmp_dir, "pro_raw.mp4")
        pro_path = os.path.join(tmp_dir, "pro_fixed.mp4")
        with open(pro_raw, "wb") as f:
            f.write(pro_file.getbuffer())
    else:
        pro_raw  = builtin_pro_path
        pro_path = os.path.join(tmp_dir, "pro_fixed.mp4")

    has_pro = (pro_raw is not None)

    # ── Status UI ────────────────────────────────────────────────────────
    progress_bar  = st.progress(0, text="Starting analysis…")
    status_text   = st.empty()

    def update_progress(pct: int, msg: str):
        progress_bar.progress(pct, text=msg)
        status_text.markdown(f"*{msg}*")

    # ── Step 1: Audio fix ─────────────────────────────────────────────────
    update_progress(5, "🔇 Adding silent audio track…")
    add_silent_audio(student_raw, student_path)
    if has_pro:
        add_silent_audio(pro_raw, pro_path)

    # ── Step 2: Pose extraction ───────────────────────────────────────────
    update_progress(10, "🧠 Extracting pose landmarks — student video…")

    frame_count_s = get_video_info(student_path)["total_frames"]

    def student_progress(f, t):
        pct = 10 + int(30 * f / max(t, 1))
        update_progress(pct, f"🧠 Pose extraction — student: frame {f}/{t}")

    frame_data_student = extract_pose_features(
        student_path,
        track_landmarks=track_options,
        progress_callback=student_progress,
        apply_smoothing=use_one_euro,
        smoothing_min_cutoff=oe_min_cutoff,
        smoothing_beta=oe_beta,
    )

    if has_pro:
        update_progress(40, "🧠 Extracting pose landmarks — pro video…")

        def pro_progress(f, t):
            pct = 40 + int(20 * f / max(t, 1))
            update_progress(pct, f"🧠 Pose extraction — pro: frame {f}/{t}")

        frame_data_pro = extract_pose_features(
            pro_path,
            track_landmarks=track_options,
            progress_callback=pro_progress,
            apply_smoothing=use_one_euro,
            smoothing_min_cutoff=oe_min_cutoff,
            smoothing_beta=oe_beta,
        )
    else:
        frame_data_pro = None

    # ── Step 3: Phase detection ───────────────────────────────────────────
    update_progress(62, "🔍 Detecting swing phases — student…")
    signal_key = f"{track_options[0]}_y"
    video_info_s = get_video_info(student_path)
    video_fps = video_info_s.get("fps", 30.0) or 30.0
    (phase_ranges_s, swing_start_s, swing_end_s, wrist_y_s, smoothed_s) = \
        detect_swing_phases(frame_data_student, signal_key=signal_key,
                            smoothing_window=smoothing,
                            threshold_percentile=threshold_pct,
                            use_one_euro=use_one_euro,
                            fps=video_fps,
                            one_euro_min_cutoff=oe_min_cutoff,
                            one_euro_beta=oe_beta)

    phase_ranges_p = swing_start_p = swing_end_p = wrist_y_p = smoothed_p = None
    if has_pro and frame_data_pro:
        update_progress(65, "🔍 Detecting swing phases — pro…")
        (phase_ranges_p, swing_start_p, swing_end_p, wrist_y_p, smoothed_p) = \
            detect_swing_phases(frame_data_pro, signal_key=signal_key,
                                smoothing_window=smoothing,
                                threshold_percentile=threshold_pct,
                                use_one_euro=use_one_euro,
                                fps=video_fps,
                                one_euro_min_cutoff=oe_min_cutoff,
                                one_euro_beta=oe_beta)

    # ── Step 4: DTW alignment + similarity ──────────────────────────────
    dtw_alignment    = None
    similarity_score = None

    if has_pro and phase_ranges_p is not None and use_dtw:
        update_progress(68, "📐 Computing DTW alignment…")
        dtw_alignment = align_phase_frames(
            phase_ranges_s, phase_ranges_p, smoothed_s, smoothed_p
        )
        similarity_score = compute_similarity_score(
            smoothed_s, smoothed_p,
            swing_start_s, swing_end_s,
            swing_start_p, swing_end_p,
        )

    elif has_pro and phase_ranges_p is not None:
        # Simple midpoint alignment
        dtw_alignment = {
            phase: ((s + e) // 2, (phase_ranges_p[phase][0] + phase_ranges_p[phase][1]) // 2, None)
            for phase, (s, e) in phase_ranges_s.items()
            if phase in phase_ranges_p
        }

    # ── Step 5: Build extra series dict ─────────────────────────────────
    extra_series_s = {}
    for lm in track_options:
        if lm != track_options[0]:            # wrist is primary
            key = f"{lm}_y"
            arr = extract_y_series(frame_data_student, key)
            if not np.all(np.isnan(arr)):
                extra_series_s[lm.capitalize()] = arr

    # ── Step 5b: Compute kinematics ──────────────────────────────────────
    update_progress(55, "⚡ Computing velocities…")
    velocity_data_s = compute_velocities_from_frame_data(
        frame_data_student, fps=video_fps,
        landmark_groups=track_options,
    )

    # Compute peak velocities per phase (wrist is primary)
    primary_vel_key = f"{track_options[0]}_velocity"
    if primary_vel_key in velocity_data_s:
        peak_velocities_s = find_peak_velocities(
            velocity_data_s[primary_vel_key], phase_ranges_s
        )
    else:
        peak_velocities_s = {}

    # Angles and angular velocity computed after 3D extraction (Step 6c)
    angle_data_s = {}
    angular_velocity_data_s = {}
    peak_angular_vel_s = {}

    # ── Step 6: Comparison figure ────────────────────────────────────────
    comparison_fig = None
    if has_pro and phase_ranges_p is not None:
        update_progress(72, "📸 Generating phase comparison grid…")
        comparison_fig = compare_swing_phases(
            student_video_path=student_path,
            student_phases=phase_ranges_s,
            pro_video_path=pro_path,
            pro_phases=phase_ranges_p,
            dtw_alignment=dtw_alignment,
            show_pose=show_pose,
        )

    # ── Step 6b: 3D landmark extraction ─────────────────────────────────
    update_progress(75, "🎯 Extracting 3D landmarks — student…")
    frames_3d_s = extract_full_3d_landmarks(student_path)
    club_data_s = estimate_club_positions(frames_3d_s, phase_ranges_s)

    frames_3d_p = None
    club_data_p = None
    if has_pro:
        update_progress(76, "🎯 Extracting 3D landmarks — pro…")
        frames_3d_p = extract_full_3d_landmarks(pro_path)
        club_data_p = estimate_club_positions(frames_3d_p, phase_ranges_p)

    json_3d_bytes = export_3d_json(frames_3d_s, club_data_s, phase_ranges_s)

    # ── Step 6c: Compute joint angles from 3D landmarks ─────────────────
    update_progress(77, "📐 Computing joint angles…")
    if frames_3d_s:
        angle_data_s = compute_all_joint_angles(frames_3d_s)

        # Smooth angle series if One-Euro is enabled
        if use_one_euro:
            for k, arr in angle_data_s.items():
                angle_data_s[k] = smooth_series(
                    arr, fps=video_fps,
                    min_cutoff=oe_min_cutoff, beta=oe_beta,
                )

        # Angular velocities
        for k, arr in angle_data_s.items():
            angular_velocity_data_s[k] = compute_angular_velocity(arr, fps=video_fps)

        # Peak angular velocities per phase
        for k, arr in angular_velocity_data_s.items():
            peak_angular_vel_s[k] = find_peak_velocities(arr, phase_ranges_s)

    # ── Step 7: Debug video ──────────────────────────────────────────────
    update_progress(78, "🎬 Generating debug video…")
    debug_output = os.path.join(tmp_dir, "debug.mp4")

    def video_progress(f, t):
        pct = 78 + int(18 * f / max(t, 1))
        update_progress(pct, f"🎬 Rendering debug video: frame {f}/{t}")

    playable_path = generate_debug_video(
        video_path=student_path,
        output_path=debug_output,
        wrist_y=wrist_y_s,
        smoothed=smoothed_s,
        phase_ranges=phase_ranges_s,
        swing_start=swing_start_s,
        swing_end=swing_end_s,
        slow_factor=slow_factor,
        extra_series=extra_series_s if extra_series_s else None,
        progress_callback=video_progress,
    )

    # ── Step 8: Export data ──────────────────────────────────────────────
    update_progress(97, "📄 Preparing export data…")
    csv_bytes  = export_to_csv(
        frame_data_student, phase_ranges_s,
        velocity_data=velocity_data_s,
        angle_data=angle_data_s,
        angular_velocity_data=angular_velocity_data_s,
    )
    json_bytes = export_to_json(
        frame_data_student, phase_ranges_s,
        swing_start_s, swing_end_s,
        similarity_score=similarity_score,
        velocity_data=velocity_data_s,
        angle_data=angle_data_s,
        angular_velocity_data=angular_velocity_data_s,
        peak_velocities=peak_velocities_s,
    )

    # Load debug video bytes
    video_bytes = None
    if os.path.exists(playable_path):
        with open(playable_path, "rb") as f:
            video_bytes = f.read()

    # ── Store in session state ────────────────────────────────────────────
    update_progress(100, "✅ Analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    st.session_state["results"] = {
        # Student data
        "frame_data_s":   frame_data_student,
        "phase_ranges_s": phase_ranges_s,
        "swing_start_s":  swing_start_s,
        "swing_end_s":    swing_end_s,
        "wrist_y_s":      wrist_y_s,
        "smoothed_s":     smoothed_s,
        "extra_series_s": extra_series_s,
        "total_frames_s": len(frame_data_student),
        # Kinematics data
        "velocity_data_s":     velocity_data_s,
        "angle_data_s":        angle_data_s,
        "angular_velocity_s":  angular_velocity_data_s,
        "peak_velocities_s":   peak_velocities_s,
        "peak_angular_vel_s":  peak_angular_vel_s,
        "video_fps":           video_fps,
        # Pro data
        "has_pro":        has_pro,
        "phase_ranges_p": phase_ranges_p,
        # DTW
        "dtw_alignment":     dtw_alignment,
        "similarity_score":  similarity_score,
        # Outputs
        "comparison_fig": comparison_fig,
        "video_bytes":    video_bytes,
        "csv_bytes":      csv_bytes,
        "json_bytes":     json_bytes,
        # 3D data
        "frames_3d_s":   frames_3d_s,
        "club_data_s":   club_data_s,
        "frames_3d_p":   frames_3d_p,
        "club_data_p":   club_data_p,
        "json_3d_bytes": json_3d_bytes,
        # Settings echo
        "track_options":  track_options,
        "slow_factor":    slow_factor,
        "use_dtw":        use_dtw,
    }
    st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# Results display
# ────────────────────────────────────────────────────────────────────────────

R = st.session_state.get("results")

if R is None:
    # Welcome state
    st.markdown("""
    <div class="section-card" style="text-align:center;padding:3rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">⛳</div>
        <h2 style="color:#e2e8f0;margin-bottom:0.5rem;">Welcome to GolfPosePro</h2>
        <p style="color:#64748b;max-width:500px;margin:auto;line-height:1.7;">
            Upload your golf swing video in the sidebar, configure analysis settings,
            and click <strong style="color:#4ade80">Run Analysis</strong> to get started.
        </p>
        <div style="display:flex;gap:1.5rem;justify-content:center;margin-top:2rem;flex-wrap:wrap;">
            <div style="color:#4ade80;font-size:0.85rem;">🧠 MediaPipe Pose</div>
            <div style="color:#38bdf8;font-size:0.85rem;">📐 DTW Alignment</div>
            <div style="color:#f472b6;font-size:0.85rem;">🦴 Multi-landmark</div>
            <div style="color:#fb923c;font-size:0.85rem;">🎬 Debug Video</div>
            <div style="color:#a78bfa;font-size:0.85rem;">📄 CSV / JSON Export</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Top KPI row ───────────────────────────────────────────────────────────────

phase_ranges_s  = R["phase_ranges_s"]
swing_start_s   = R["swing_start_s"]
swing_end_s     = R["swing_end_s"]
total_frames_s  = R["total_frames_s"]
similarity_score = R["similarity_score"]
use_dtw         = R["use_dtw"]

swing_duration  = swing_end_s - swing_start_s
num_phases      = len(phase_ranges_s)

kpi_cols = st.columns(4)
kpi_data = [
    (f"{num_phases}", "PHASES DETECTED", "Address → Follow Through"),
    (f"{swing_duration}", "SWING FRAMES", f"Frames {swing_start_s}–{swing_end_s}"),
    (f"{total_frames_s}", "TOTAL FRAMES", f"{total_frames_s} analyzed"),
    (
        f"{similarity_score:.0f}%" if similarity_score is not None else "N/A",
        "SWING SIMILARITY",
        "DTW vs. Pro" if similarity_score is not None else "No pro video",
    ),
]
for col, (val, label, sub) in zip(kpi_cols, kpi_data):
    col.markdown(_metric_card(val, label, sub), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Phase timeline ────────────────────────────────────────────────────────────

st.markdown(_section_title("📊", "Swing Phase Timeline"), unsafe_allow_html=True)
st.markdown(
    _phase_timeline_html(phase_ranges_s, total_frames_s),
    unsafe_allow_html=True,
)

# Phase table
phase_table_rows = ""
for phase, (s, e) in phase_ranges_s.items():
    color   = PHASE_COLORS.get(phase, "#888")
    dur     = e - s
    pct     = dur / total_frames_s * 100
    dtw_tag = ""
    if use_dtw and R["dtw_alignment"] and phase in R["dtw_alignment"]:
        d = R["dtw_alignment"][phase][2]
        if d is not None:
            dtw_tag = f'<span style="color:#4ade80;font-size:0.7rem">DTW: {d:.1f}</span>'
    phase_table_rows += (
        f"<tr>"
        f"  <td><span style='color:{color};font-weight:600'>{phase}</span></td>"
        f"  <td style='color:#94a3b8'>{s}</td>"
        f"  <td style='color:#94a3b8'>{e}</td>"
        f"  <td style='color:#e2e8f0'>{dur}</td>"
        f"  <td style='color:#64748b'>{pct:.1f}%</td>"
        f"  <td>{dtw_tag}</td>"
        f"</tr>"
    )

st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:0.85rem;margin-top:0.5rem;">
  <thead>
    <tr style="color:#64748b;border-bottom:1px solid rgba(255,255,255,0.08);">
      <th style="text-align:left;padding:6px 8px">Phase</th>
      <th style="text-align:left;padding:6px 8px">Start</th>
      <th style="text-align:left;padding:6px 8px">End</th>
      <th style="text-align:left;padding:6px 8px">Frames</th>
      <th style="text-align:left;padding:6px 8px">% of clip</th>
      <th style="text-align:left;padding:6px 8px">DTW</th>
    </tr>
  </thead>
  <tbody>
    {phase_table_rows}
  </tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_trajectory, tab_kinematics, tab_comparison, tab_3d, tab_live, tab_video, tab_export = st.tabs([
    "📈 Trajectory",
    "📐 Kinematics",
    "🔄 Phase Comparison",
    "🎯 3D Viewer",
    "📹 Live Tracking",
    "🎬 Debug Video",
    "📄 Export Data",
])


# ─── Tab 1: Trajectory ───────────────────────────────────────────────────────

with tab_trajectory:
    st.markdown(_section_title("📈", "Wrist Y-Trajectory with Phase Overlay"),
                unsafe_allow_html=True)

    # Build extra_series for multi-landmark
    extra_series_s = R.get("extra_series_s", {})

    traj_fig = plot_trajectory(
        wrist_y=R["wrist_y_s"],
        smoothed=R["smoothed_s"],
        phase_ranges=phase_ranges_s,
        swing_start=swing_start_s,
        swing_end=swing_end_s,
        title="Student Swing — Wrist Y-Trajectory",
        extra_series=extra_series_s if extra_series_s else None,
        figsize=(14, 5),
    )
    st.pyplot(traj_fig, use_container_width=True)
    plt.close(traj_fig)

    traj_bytes = _fig_to_bytes(traj_fig)
    st.download_button(
        "⬇️  Download trajectory chart (PNG)",
        data=traj_bytes,
        file_name="wrist_trajectory.png",
        mime="image/png",
    )

    if R["has_pro"] and R["phase_ranges_p"] and R.get("smoothed_p") is not None: # type: ignore
        pass  # Could add pro trajectory here too


# ─── Tab 2: Kinematics ───────────────────────────────────────────────────────

with tab_kinematics:
    velocity_data_s = R.get("velocity_data_s", {})
    angle_data_s    = R.get("angle_data_s", {})
    angular_vel_s   = R.get("angular_velocity_s", {})
    peak_vel_s      = R.get("peak_velocities_s", {})
    peak_ang_vel_s  = R.get("peak_angular_vel_s", {})
    vid_fps         = R.get("video_fps", 30.0)

    # ── Peak stats cards ─────────────────────────────────────────────
    st.markdown(_section_title("⚡", "Swing Speed & Peak Velocity"), unsafe_allow_html=True)

    if peak_vel_s:
        peak_cols = st.columns(min(len(peak_vel_s), 6))
        for col, (phase, pv) in zip(peak_cols, peak_vel_s.items()):
            color = PHASE_COLORS.get(phase, "#888")
            speed_val = f"{pv['peak_speed']:.0f}"
            col.markdown(
                _metric_card(speed_val, f"Peak px/s", f"{phase} · frame {pv['peak_frame']}"),
                unsafe_allow_html=True,
            )
    else:
        st.info("📌 No velocity data available. Re-run analysis.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Velocity chart ───────────────────────────────────────────────
    if velocity_data_s:
        st.markdown(_section_title("📈", "Landmark Velocity Over Time"), unsafe_allow_html=True)

        vel_fig, vel_ax = plt.subplots(figsize=(14, 4), facecolor="#0e1117")
        vel_ax.set_facecolor("#0e1117")

        vel_palette = ["#4ade80", "#fb923c", "#38bdf8", "#f472b6"]
        for i, (name, arr) in enumerate(velocity_data_s.items()):
            label = name.replace("_velocity", "").capitalize() + " speed"
            vel_ax.plot(
                np.arange(len(arr)), arr,
                color=vel_palette[i % len(vel_palette)],
                linewidth=1.8, label=label,
            )

        # Phase shading
        for phase, (s, e) in phase_ranges_s.items():
            c = PHASE_COLORS.get(phase, "#888888")
            vel_ax.axvspan(s, e, alpha=0.12, color=c)

        # Mark peak at impact
        if peak_vel_s and "Impact" in peak_vel_s:
            pf = peak_vel_s["Impact"]["peak_frame"]
            ps = peak_vel_s["Impact"]["peak_speed"]
            vel_ax.annotate(
                f"Peak: {ps:.0f} px/s",
                xy=(pf, ps), xytext=(pf + 10, ps * 1.1),
                fontsize=9, color="#fb5607", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#fb5607", lw=1.5),
            )

        vel_ax.axvline(swing_start_s, color="#facc15", linestyle="--", linewidth=1, alpha=0.7)
        vel_ax.axvline(swing_end_s,   color="#f87171", linestyle="--", linewidth=1, alpha=0.7)

        vel_ax.set_title("Landmark Speed (px/s)", color="white", fontsize=12, pad=8)
        vel_ax.set_xlabel("Frame", color="#94a3b8")
        vel_ax.set_ylabel("Speed (px/s)", color="#94a3b8")
        vel_ax.tick_params(colors="#94a3b8")
        vel_ax.legend(loc="upper right", fontsize=8, facecolor="#1e293b",
                      labelcolor="white", framealpha=0.7)
        for spine in vel_ax.spines.values():
            spine.set_edgecolor("#334155")
        vel_fig.tight_layout()

        st.pyplot(vel_fig, use_container_width=True)
        plt.close(vel_fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Joint angle chart ─────────────────────────────────────────────
    if angle_data_s:
        st.markdown(_section_title("📐", "Joint Angles Over Time"), unsafe_allow_html=True)

        # Let user pick which angles to show
        available_angles = list(angle_data_s.keys())
        default_angles = [a for a in ["elbow_L", "elbow_R", "spine_tilt"]
                          if a in available_angles]
        selected_angles = st.multiselect(
            "Select angles to display",
            available_angles,
            default=default_angles if default_angles else available_angles[:3],
            key="angle_select",
        )

        if selected_angles:
            ang_fig, ang_ax = plt.subplots(figsize=(14, 4), facecolor="#0e1117")
            ang_ax.set_facecolor("#0e1117")

            angle_palette = ["#4ade80", "#fb923c", "#38bdf8", "#f472b6",
                             "#a78bfa", "#facc15", "#22d3ee"]
            for i, name in enumerate(selected_angles):
                arr = angle_data_s[name]
                label = name.replace("_", " ").title()
                ang_ax.plot(
                    np.arange(len(arr)), arr,
                    color=angle_palette[i % len(angle_palette)],
                    linewidth=1.8, label=label,
                )

            for phase, (s, e) in phase_ranges_s.items():
                c = PHASE_COLORS.get(phase, "#888888")
                ang_ax.axvspan(s, e, alpha=0.12, color=c)

            ang_ax.axvline(swing_start_s, color="#facc15", linestyle="--", linewidth=1, alpha=0.7)
            ang_ax.axvline(swing_end_s,   color="#f87171", linestyle="--", linewidth=1, alpha=0.7)

            ang_ax.set_title("Joint Angles (°)", color="white", fontsize=12, pad=8)
            ang_ax.set_xlabel("Frame", color="#94a3b8")
            ang_ax.set_ylabel("Angle (°)", color="#94a3b8")
            ang_ax.tick_params(colors="#94a3b8")
            ang_ax.legend(loc="upper right", fontsize=8, facecolor="#1e293b",
                          labelcolor="white", framealpha=0.7)
            for spine in ang_ax.spines.values():
                spine.set_edgecolor("#334155")
            ang_fig.tight_layout()

            st.pyplot(ang_fig, use_container_width=True)
            plt.close(ang_fig)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Angular velocity chart ───────────────────────────────────────
        if angular_vel_s and selected_angles:
            st.markdown(_section_title("🌀", "Angular Velocity"), unsafe_allow_html=True)

            av_fig, av_ax = plt.subplots(figsize=(14, 4), facecolor="#0e1117")
            av_ax.set_facecolor("#0e1117")

            for i, name in enumerate(selected_angles):
                if name in angular_vel_s:
                    arr = angular_vel_s[name]
                    label = name.replace("_", " ").title() + " (°/s)"
                    av_ax.plot(
                        np.arange(len(arr)), arr,
                        color=angle_palette[i % len(angle_palette)],
                        linewidth=1.8, label=label,
                    )

            for phase, (s, e) in phase_ranges_s.items():
                c = PHASE_COLORS.get(phase, "#888888")
                av_ax.axvspan(s, e, alpha=0.12, color=c)

            av_ax.set_title("Angular Velocity (°/s)", color="white", fontsize=12, pad=8)
            av_ax.set_xlabel("Frame", color="#94a3b8")
            av_ax.set_ylabel("°/s", color="#94a3b8")
            av_ax.tick_params(colors="#94a3b8")
            av_ax.legend(loc="upper right", fontsize=8, facecolor="#1e293b",
                          labelcolor="white", framealpha=0.7)
            for spine in av_ax.spines.values():
                spine.set_edgecolor("#334155")
            av_fig.tight_layout()

            st.pyplot(av_fig, use_container_width=True)
            plt.close(av_fig)

    elif not velocity_data_s:
        st.info("📌 No kinematics data. Run analysis to compute velocity and angle metrics.")


# ─── Tab 2: Comparison ───────────────────────────────────────────────────────

with tab_comparison:
    if not R["has_pro"] or R["comparison_fig"] is None:
        st.info("📌 Upload a pro reference video to enable phase comparison.")
    else:
        dtw_badge = ""
        if use_dtw and R["dtw_alignment"]:
            dtw_badge = (
                '<span style="background:rgba(74,222,128,0.15);color:#4ade80;'
                'border:1px solid rgba(74,222,128,0.3);border-radius:6px;'
                'padding:2px 10px;font-size:0.78rem;font-weight:600;">DTW ALIGNED</span>'
            )
        st.markdown(
            f'{_section_title("🔄", "Phase-by-Phase Comparison")} {dtw_badge}',
            unsafe_allow_html=True,
        )

        if similarity_score is not None:
            score_color = (
                "#4ade80" if similarity_score >= 70 else
                "#facc15" if similarity_score >= 40 else "#f87171"
            )
            st.markdown(
                f'<div style="text-align:center;margin-bottom:1rem;">'
                f'  <span style="font-size:2rem;font-weight:700;color:{score_color}">'
                f'{similarity_score:.0f}%</span>'
                f'  <span style="color:#64748b;font-size:0.9rem;margin-left:0.5rem">'
                f'DTW swing similarity score</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        comp_fig = R["comparison_fig"]
        st.pyplot(comp_fig, use_container_width=True)

        comp_bytes = _fig_to_bytes(comp_fig)
        st.download_button(
            "⬇️  Download comparison grid (PNG)",
            data=comp_bytes,
            file_name="swing_phase_comparison.png",
            mime="image/png",
        )


# ─── Tab 3: 3D Viewer ────────────────────────────────────────────────────────

with tab_3d:
    st.markdown(_section_title("🎯", "3D Swing Viewer"), unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#94a3b8;font-size:0.85rem;margin-bottom:1rem;">
        Interactive 3D skeleton with estimated golf club. Drag to rotate, scroll to zoom.
        Phase colors update as the animation plays.
    </p>
    """, unsafe_allow_html=True)

    frames_3d_s = R.get("frames_3d_s")
    club_data_s = R.get("club_data_s")

    if frames_3d_s and club_data_s:
        is_comparison = R["has_pro"] and R.get("frames_3d_p") is not None
        viewer_html = build_3d_viewer_html(
            frames_3d=frames_3d_s,
            club_data=club_data_s,
            phase_ranges=phase_ranges_s,
            phase_colors=PHASE_COLORS,
            mode="comparison" if is_comparison else "replay",
            comparison_pose=R.get("frames_3d_p"),
            comparison_club=R.get("club_data_p"),
            dtw_alignment=R.get("dtw_alignment"),
            height=600,
        )
        st.components.v1.html(viewer_html, height=650, scrolling=False)
    else:
        st.info("📌 3D data not available. Re-run analysis to generate 3D landmarks.")


# ─── Tab 4: Live Tracking ────────────────────────────────────────────────────

with tab_live:
    st.markdown(_section_title("📹", "Live Webcam Tracking"), unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#94a3b8;font-size:0.85rem;margin-bottom:1rem;">
        Real-time pose tracking using your webcam. MediaPipe runs directly in the browser —
        no video upload needed. Click <strong style="color:#4ade80">Start Tracking</strong> to begin.
    </p>
    """, unsafe_allow_html=True)
    live_html = build_live_tracking_html(height=600)
    st.components.v1.html(live_html, height=650, scrolling=False)


# ─── Tab 5: Debug Video ───────────────────────────────────────────────────────

with tab_video:
    st.markdown(_section_title("🎬", "Annotated Debug Video"), unsafe_allow_html=True)

    video_bytes = R.get("video_bytes")
    if video_bytes:
        st.markdown(f"""
        <div style="color:#64748b;font-size:0.85rem;margin-bottom:1rem;">
            Slow-motion factor: <strong style="color:#4ade80">{R['slow_factor']}×</strong>
            applied to swing window (frames {swing_start_s}–{swing_end_s}).
        </div>
        """, unsafe_allow_html=True)
        st.video(io.BytesIO(video_bytes))
        st.download_button(
            "⬇️  Download debug video (MP4)",
            data=video_bytes,
            file_name="golf_debug_video.mp4",
            mime="video/mp4",
        )
    else:
        st.warning("⚠️  Debug video could not be generated. Check that ffmpeg is installed.")


# ─── Tab 6: Export ───────────────────────────────────────────────────────────

with tab_export:
    st.markdown(_section_title("📄", "Export Analysis Data"), unsafe_allow_html=True)

    col_csv, col_json, col_3d = st.columns(3)

    with col_csv:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📊 CSV Export</div>
            <p style="color:#94a3b8;font-size:0.85rem;line-height:1.6;">
                Per-frame data with phase labels.<br>
                Columns: <code>frame_idx</code>, <code>phase</code>,
                <code>wrist_y</code>, <code>hip_y</code>*, <code>shoulder_y</code>*
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            "⬇️  Download CSV",
            data=R["csv_bytes"],
            file_name="golf_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_json:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📦 JSON Export</div>
            <p style="color:#94a3b8;font-size:0.85rem;line-height:1.6;">
                Full structured analysis including metadata,
                phase boundaries, swing window, similarity score, and all frames.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            "⬇️  Download JSON",
            data=R["json_bytes"],
            file_name="golf_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_3d:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🎯 3D JSON Export</div>
            <p style="color:#94a3b8;font-size:0.85rem;line-height:1.6;">
                Full 3D landmark data with golf club estimation.
                Use with Three.js or other 3D tools.
            </p>
        </div>
        """, unsafe_allow_html=True)
        json_3d = R.get("json_3d_bytes", b"")
        if json_3d:
            st.download_button(
                "⬇️  Download 3D JSON",
                data=json_3d,
                file_name="golf_3d_landmarks.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("3D data not available.")

    # Preview first 10 rows of CSV
    st.markdown("##### Preview (first 10 frames)")
    csv_preview = R["csv_bytes"].decode("utf-8")
    lines   = csv_preview.split("\n")
    preview = "\n".join(lines[:11])
    st.code(preview, language="csv")
```

**Sidebar:**
- New "🎯 Smoothing & Kinematics" section with:
  - Checkbox: "Use adaptive smoothing (One-Euro Filter)" — default ON
  - Sliders: Min cutoff (smoothness) and Beta (speed adaptation)

**Pipeline:**
- Step 5b: Compute 2D velocities from smoothed landmarks
- Step 6c: Compute joint angles from 3D world landmarks, smooth with One-Euro, derive angular velocities

**New Tab — "📐 Kinematics":**
- **Peak velocity cards** — one per swing phase showing peak px/s
- **Velocity chart** — landmark speed over time with phase overlays and impact annotation
- **Joint angle chart** — user-selectable angles (elbow, knee, shoulder, spine tilt)
- **Angular velocity chart** — °/s for selected joints

## Verification

| Test | Result |
|------|--------|
| Python syntax check (all 6 files) | ✅ Pass |
| Module imports | ✅ Pass |
| One-Euro filter smoke test | ✅ Pass (reduces noise std) |
| Joint angle computation (90° test) | ✅ Pass (90.0°) |
| Full app import chain | ✅ Pass |
