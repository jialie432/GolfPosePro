"""
club_estimation.py — Estimate golf club position from body landmarks.

Uses wrist and elbow positions to approximate grip point and shaft direction.
MediaPipe does not track the club directly, so this is a biomechanical heuristic.
"""

import math
import numpy as np


# MediaPipe landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


def _get_landmark_pos(landmarks, idx):
    """Extract (x, y, z) from a landmarks list, or None if missing."""
    for lm in landmarks:
        if lm["idx"] == idx and lm["x"] is not None:
            return np.array([lm["x"], lm["y"], lm["z"]])
    return None


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.array([0.0, 1.0, 0.0])


def estimate_club_positions(frames_3d: list, phase_ranges: dict = None) -> list:
    """
    Estimate golf club grip and shaft end positions for each frame.

    Args:
        frames_3d: Output of extract_full_3d_landmarks().
        phase_ranges: Optional phase dict for phase-dependent angle offsets.

    Returns:
        List of dicts per frame:
        [{grip: {x,y,z}, shaft_end: {x,y,z}, has_club: bool}, ...]
    """
    # Estimate body scale from first frame with valid landmarks
    body_scale = _estimate_body_scale(frames_3d)
    club_length = body_scale * 1.8 if body_scale > 0 else 1.1

    club_data = []
    for frame in frames_3d:
        landmarks = frame.get("landmarks", [])
        if not landmarks:
            club_data.append({"grip": None, "shaft_end": None, "has_club": False})
            continue

        lw = _get_landmark_pos(landmarks, LEFT_WRIST)
        rw = _get_landmark_pos(landmarks, RIGHT_WRIST)
        le = _get_landmark_pos(landmarks, LEFT_ELBOW)
        re = _get_landmark_pos(landmarks, RIGHT_ELBOW)

        if lw is None or rw is None:
            club_data.append({"grip": None, "shaft_end": None, "has_club": False})
            continue

        # Grip = midpoint of both wrists
        grip = (lw + rw) / 2.0

        # Shaft direction: blend both forearm vectors, extend beyond wrists
        shaft_dir = np.array([0.0, 1.0, 0.0])  # default: straight down
        if le is not None and re is not None:
            left_forearm = _normalize(lw - le)
            right_forearm = _normalize(rw - re)
            forearm_avg = _normalize(left_forearm + right_forearm)

            # Blend with a downward bias (gravity pulls the club down)
            down = np.array([0.0, 1.0, 0.0])  # MediaPipe Y+ is down
            shaft_dir = _normalize(forearm_avg * 0.6 + down * 0.4)

        shaft_end = grip + shaft_dir * club_length

        club_data.append({
            "grip": {"x": round(float(grip[0]), 5),
                     "y": round(float(grip[1]), 5),
                     "z": round(float(grip[2]), 5)},
            "shaft_end": {"x": round(float(shaft_end[0]), 5),
                          "y": round(float(shaft_end[1]), 5),
                          "z": round(float(shaft_end[2]), 5)},
            "has_club": True,
        })

    return club_data


def _estimate_body_scale(frames_3d: list) -> float:
    """Estimate body scale as shoulder-to-hip distance from the first valid frame."""
    for frame in frames_3d[:30]:  # check first 30 frames
        landmarks = frame.get("landmarks", [])
        ls = _get_landmark_pos(landmarks, LEFT_SHOULDER)
        lh = _get_landmark_pos(landmarks, LEFT_HIP)
        if ls is not None and lh is not None:
            return float(np.linalg.norm(ls - lh))
    return 0.0
