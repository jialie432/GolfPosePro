"""
threejs_component.py — Build self-contained HTML strings for Streamlit embedding.

Reads HTML templates and injects pose/club/phase data as JSON variables.
"""

import base64
import json
from pathlib import Path


_TEMPLATES_DIR = Path(__file__).parent / "templates"
_MODELS_DIR = Path(__file__).parent.parent / "models"


def _model_data_url(filename: str) -> str:
    """Read a local GLB file and return it as a base64 data URL."""
    path = _MODELS_DIR / filename
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:model/gltf-binary;base64,{data}"


def build_3d_viewer_html(
    frames_3d: list,
    club_data: list,
    phase_ranges: dict,
    phase_colors: dict,
    mode: str = "replay",
    comparison_pose: list = None,
    comparison_club: list = None,
    dtw_alignment: dict = None,
    height: int = 600,
) -> str:
    """
    Build a self-contained HTML string with Three.js 3D viewer.

    Args:
        frames_3d:        Output of extract_full_3d_landmarks().
        club_data:        Output of estimate_club_positions().
        phase_ranges:     Dict {phase_name: (start, end)}.
        phase_colors:     Dict {phase_name: "#hex"}.
        mode:             "replay" or "comparison".
        comparison_pose:  Pro's 3D landmarks (for comparison mode).
        comparison_club:  Pro's club data (for comparison mode).
        dtw_alignment:    DTW alignment dict (for comparison mode).
        height:           Component height in pixels.

    Returns:
        HTML string ready for st.components.v1.html().
    """
    template_path = _TEMPLATES_DIR / "viewer_3d.html"
    html = template_path.read_text(encoding="utf-8")

    # Serialize phase_ranges as {name: {start, end}}
    phase_data = {}
    for name, (s, e) in phase_ranges.items():
        phase_data[name] = {"start": int(s), "end": int(e)}

    # Serialize DTW alignment for JS
    dtw_js = "null"
    if dtw_alignment:
        dtw_js = json.dumps({
            k: [int(v[0]), int(v[1]), float(v[2]) if v[2] is not None else 0] if len(v) >= 3 else [int(v[0]), int(v[1]), 0]
            for k, v in dtw_alignment.items()
        })

    html = html.replace("__MODEL_URL__", _model_data_url("Xbot.glb"))
    html = html.replace("__POSE_DATA__", json.dumps(frames_3d))
    html = html.replace("__CLUB_DATA__", json.dumps(club_data))
    html = html.replace("__PHASE_DATA__", json.dumps(phase_data))
    html = html.replace("__PHASE_COLORS__", json.dumps(phase_colors))
    html = html.replace("__MODE__", mode)
    html = html.replace("__COMPARISON_POSE__", json.dumps(comparison_pose or []))
    html = html.replace("__COMPARISON_CLUB__", json.dumps(comparison_club or []))
    html = html.replace("__DTW_ALIGNMENT__", dtw_js)
    html = html.replace("__HEIGHT__", str(height))

    return html


def build_live_tracking_html(height: int = 650) -> str:
    """
    Build HTML string for browser-side MediaPipe + Three.js live tracking.

    Returns:
        HTML string ready for st.components.v1.html().
    """
    template_path = _TEMPLATES_DIR / "live_tracking.html"
    html = template_path.read_text(encoding="utf-8")
    html = html.replace("__HEIGHT__", str(height))
    return html
