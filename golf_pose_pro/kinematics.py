"""
kinematics.py — Velocity and joint angle computation from MediaPipe landmarks.

Computes:
  - Linear velocity of landmarks (px/s for 2D, m/s for 3D)
  - Joint angles (elbow, knee, shoulder, spine tilt)
  - Angular velocity of joint angles (°/s)
  - Peak velocity detection per swing phase
"""

import numpy as np
from typing import Optional


# ─── MediaPipe landmark indices ──────────────────────────────────────────────

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


# ─── Joint angle definitions ────────────────────────────────────────────────
# Each entry: (name, landmark_A, landmark_B_vertex, landmark_C)
# Angle is measured at B (the vertex) between rays BA and BC.

JOINT_ANGLE_DEFS = [
    ("elbow_L",    LEFT_SHOULDER,  LEFT_ELBOW,   LEFT_WRIST),
    ("elbow_R",    RIGHT_SHOULDER, RIGHT_ELBOW,  RIGHT_WRIST),
    ("knee_L",     LEFT_HIP,       LEFT_KNEE,    LEFT_ANKLE),
    ("knee_R",     RIGHT_HIP,      RIGHT_KNEE,   RIGHT_ANKLE),
    ("shoulder_L", LEFT_HIP,       LEFT_SHOULDER, LEFT_ELBOW),
    ("shoulder_R", RIGHT_HIP,      RIGHT_SHOULDER, RIGHT_ELBOW),
]


# ─── Geometry helpers ────────────────────────────────────────────────────────

def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the angle (in degrees) between two vectors.
    Returns NaN if either vector has zero length.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return np.nan
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_joint_angle(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> float:
    """
    Compute the angle at vertex B formed by points A-B-C (in degrees).

    Parameters
    ----------
    a, b, c : np.ndarray
        Coordinates (2D or 3D) of the three landmarks.
        B is the vertex (e.g., elbow for shoulder-elbow-wrist angle).

    Returns
    -------
    float
        Angle in degrees [0, 180]. NaN if any landmark is missing.
    """
    if a is None or b is None or c is None:
        return np.nan
    ba = a - b
    bc = c - b
    return _angle_between_vectors(ba, bc)


def _get_lm_pos(landmarks: list, idx: int) -> Optional[np.ndarray]:
    """Extract (x, y, z) for a landmark index, or None if missing."""
    for lm in landmarks:
        if lm["idx"] == idx and lm["x"] is not None:
            return np.array([lm["x"], lm["y"], lm["z"]])
    return None


# ─── Joint angles from 3D landmarks ─────────────────────────────────────────

def compute_all_joint_angles(frames_3d: list) -> dict:
    """
    Compute joint angles from full 3D landmark data for every frame.

    Parameters
    ----------
    frames_3d : list[dict]
        Output of extract_full_3d_landmarks(). Each entry has:
        {frame_idx, timestamp_ms, landmarks: [{idx, x, y, z, visibility}, ...]}

    Returns
    -------
    dict
        {angle_name: np.ndarray of shape (n_frames,)} — angles in degrees.
        Keys: 'elbow_L', 'elbow_R', 'knee_L', 'knee_R', 'shoulder_L',
              'shoulder_R', 'spine_tilt'.
    """
    n = len(frames_3d)
    angles = {name: np.full(n, np.nan) for name, *_ in JOINT_ANGLE_DEFS}
    angles["spine_tilt"] = np.full(n, np.nan)

    for i, frame in enumerate(frames_3d):
        landmarks = frame.get("landmarks", [])
        if not landmarks:
            continue

        # Standard joint angles
        for name, idx_a, idx_b, idx_c in JOINT_ANGLE_DEFS:
            a = _get_lm_pos(landmarks, idx_a)
            b = _get_lm_pos(landmarks, idx_b)
            c = _get_lm_pos(landmarks, idx_c)
            angles[name][i] = compute_joint_angle(a, b, c)

        # Spine tilt: angle between vertical (0, -1, 0) and
        # the vector from mid-hip to mid-shoulder
        ls = _get_lm_pos(landmarks, LEFT_SHOULDER)
        rs = _get_lm_pos(landmarks, RIGHT_SHOULDER)
        lh = _get_lm_pos(landmarks, LEFT_HIP)
        rh = _get_lm_pos(landmarks, RIGHT_HIP)

        if ls is not None and rs is not None and lh is not None and rh is not None:
            mid_shoulder = (ls + rs) / 2.0
            mid_hip = (lh + rh) / 2.0
            spine_vec = mid_shoulder - mid_hip
            vertical = np.array([0.0, -1.0, 0.0])  # MediaPipe: Y+ is down
            angles["spine_tilt"][i] = _angle_between_vectors(spine_vec, vertical)

    return angles


# ─── Velocity computation ───────────────────────────────────────────────────

def compute_landmark_velocity(
    x_series: np.ndarray,
    y_series: np.ndarray,
    fps: float,
    z_series: np.ndarray = None,
) -> np.ndarray:
    """
    Compute instantaneous speed of a landmark from its coordinate series.

    Uses central finite differences for interior points and forward/backward
    differences at endpoints.

    Parameters
    ----------
    x_series, y_series : np.ndarray
        Coordinate arrays (same length). Units: pixels or meters.
    fps : float
        Frames per second.
    z_series : np.ndarray, optional
        Z-coordinate for 3D velocity.

    Returns
    -------
    np.ndarray
        Speed array (same length) in units/second (px/s or m/s).
    """
    dx = np.gradient(x_series) * fps
    dy = np.gradient(y_series) * fps

    if z_series is not None:
        dz = np.gradient(z_series) * fps
        speed = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        speed = np.sqrt(dx**2 + dy**2)

    return speed


def compute_angular_velocity(
    angle_series: np.ndarray,
    fps: float,
) -> np.ndarray:
    """
    Compute angular velocity (°/s) from a joint angle time series.

    Parameters
    ----------
    angle_series : np.ndarray
        Angle values in degrees per frame.
    fps : float
        Frames per second.

    Returns
    -------
    np.ndarray
        Angular velocity in °/s (absolute value).
    """
    # Replace NaNs with interpolated values for gradient computation
    clean = angle_series.copy()
    nans = np.isnan(clean)
    if nans.all():
        return np.zeros_like(clean)
    if nans.any():
        idx = np.where(~nans, np.arange(len(clean)), 0)
        np.maximum.accumulate(idx, out=idx)
        clean = clean[idx]

    d_angle = np.gradient(clean) * fps
    return np.abs(d_angle)


# ─── Velocity from frame_data (2D, pixels) ──────────────────────────────────

def compute_velocities_from_frame_data(
    frame_data: list,
    fps: float,
    landmark_groups: list = None,
) -> dict:
    """
    Compute 2D velocity (px/s) for each tracked landmark group from frame_data.

    Parameters
    ----------
    frame_data : list[dict]
        Per-frame dicts with keys like 'wrist_x', 'wrist_y', etc.
    fps : float
        Video FPS.
    landmark_groups : list[str], optional
        Which groups to compute velocity for. Default: auto-detect from keys.

    Returns
    -------
    dict
        {group_name + '_velocity': np.ndarray} — speed in px/s per frame.
    """
    if not frame_data:
        return {}

    if landmark_groups is None:
        sample = frame_data[0]
        landmark_groups = list({k.rsplit("_", 1)[0] for k in sample
                                if k.endswith("_x") and k != "frame_idx"})

    velocities = {}
    n = len(frame_data)

    for group in landmark_groups:
        x_key = f"{group}_x"
        y_key = f"{group}_y"

        x_arr = np.array(
            [f.get(x_key, np.nan) if f.get(x_key) is not None else np.nan
             for f in frame_data], dtype=float
        )
        y_arr = np.array(
            [f.get(y_key, np.nan) if f.get(y_key) is not None else np.nan
             for f in frame_data], dtype=float
        )

        # Forward-fill NaNs
        for arr in [x_arr, y_arr]:
            nans = np.isnan(arr)
            if nans.any() and not nans.all():
                idx = np.where(~nans, np.arange(n), 0)
                np.maximum.accumulate(idx, out=idx)
                arr[:] = arr[idx]

        velocities[f"{group}_velocity"] = compute_landmark_velocity(x_arr, y_arr, fps)

    return velocities


# ─── Peak detection ──────────────────────────────────────────────────────────

def find_peak_velocities(
    velocity_array: np.ndarray,
    phase_ranges: dict,
) -> dict:
    """
    Find peak velocity within each swing phase.

    Parameters
    ----------
    velocity_array : np.ndarray
        Speed array (px/s or m/s) per frame.
    phase_ranges : dict
        {phase_name: (start_frame, end_frame)}

    Returns
    -------
    dict
        {phase_name: {"peak_speed": float, "peak_frame": int}}
    """
    peaks = {}
    for phase, (start, end) in phase_ranges.items():
        seg = velocity_array[start:end + 1]
        if len(seg) == 0:
            peaks[phase] = {"peak_speed": 0.0, "peak_frame": start}
            continue

        peak_local = int(np.nanargmax(seg))
        peaks[phase] = {
            "peak_speed": float(np.nanmax(seg)),
            "peak_frame": start + peak_local,
        }

    return peaks


def find_peak_angular_velocities(
    angular_velocities: dict,
    phase_ranges: dict,
) -> dict:
    """
    Find peak angular velocity per joint per phase.

    Parameters
    ----------
    angular_velocities : dict
        {angle_name + '_angular_vel': np.ndarray}
    phase_ranges : dict
        {phase_name: (start_frame, end_frame)}

    Returns
    -------
    dict
        {angle_name: {phase_name: {"peak_speed": float, "peak_frame": int}}}
    """
    result = {}
    for key, arr in angular_velocities.items():
        name = key.replace("_angular_vel", "")
        result[name] = find_peak_velocities(arr, phase_ranges)
    return result
