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
