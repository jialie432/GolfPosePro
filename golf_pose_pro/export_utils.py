"""
export_utils.py — Export pose analysis results to CSV and JSON.
"""

import csv
import json
import io
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy int/float types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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

    json_bytes = json.dumps(payload, indent=2, cls=_NumpyEncoder).encode("utf-8")

    # Add kinematics section if peak velocities are provided
    if peak_velocities:
        payload["kinematics"] = {
            "peak_velocities": {
                phase: {"peak_speed_px_per_s": round(v["peak_speed"], 2),
                        "peak_frame": v["peak_frame"]}
                for phase, v in peak_velocities.items()
            }
        }
        json_bytes = json.dumps(payload, indent=2, cls=_NumpyEncoder).encode("utf-8")

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

    json_bytes = json.dumps(payload, cls=_NumpyEncoder).encode("utf-8")

    if output_path:
        Path(output_path).write_bytes(json_bytes)

    return json_bytes


# ─── helpers ─────────────────────────────────────────────────────────────────

def _get_phase(frame_idx: int, phase_ranges: dict) -> str:
    for name, (s, e) in phase_ranges.items():
        if s <= frame_idx <= e:
            return name
    return ""
