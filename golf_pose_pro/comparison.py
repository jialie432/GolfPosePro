"""
comparison.py — Side-by-side student vs. pro swing phase comparison.
Supports optional pose overlay and DTW-aligned frame selection.
Uses MediaPipe Tasks API (mediapipe >= 0.10.14).
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        PoseLandmarksConnections,
        RunningMode,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

from .phase_detection import PHASE_COLORS

_DEFAULT_MODEL = Path(__file__).parent.parent / "pose_landmarker.task"

# Skeleton connections (landmark index pairs)
_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),            # right arm
    (11, 23), (12, 24),            # torso sides
    (23, 24), (23, 25), (24, 26),  # hips + legs
    (25, 27), (26, 28),
]


# ─── Frame extraction ────────────────────────────────────────────────────────

def _extract_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _draw_pose_overlay(frame: np.ndarray, model_path: str = None) -> np.ndarray:
    """Run MediaPipe PoseLandmarker (IMAGE mode) on a single frame and draw skeleton."""
    if not MP_AVAILABLE:
        return frame

    if model_path is None:
        model_path = str(_DEFAULT_MODEL)
    if not Path(model_path).exists():
        return frame

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return frame

    landmarks = result.pose_landmarks[0]
    out = frame.copy()

    # Draw connections
    for (a, b) in _POSE_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            lm_a = landmarks[a]
            lm_b = landmarks[b]
            if (lm_a.visibility or 1.0) > 0.3 and (lm_b.visibility or 1.0) > 0.3:
                x1, y1 = int(lm_a.x * w), int(lm_a.y * h)
                x2, y2 = int(lm_b.x * w), int(lm_b.y * h)
                cv2.line(out, (x1, y1), (x2, y2), (0, 255, 128), 2, cv2.LINE_AA)

    # Draw keypoints
    for lm in landmarks:
        if (lm.visibility or 1.0) > 0.3:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(out, (cx, cy), 4, (74, 222, 128), -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 4, (0, 0, 0), 1, cv2.LINE_AA)

    return out


# ─── Comparison grid ─────────────────────────────────────────────────────────

def compare_swing_phases(
    student_video_path: str,
    student_phases: dict,
    pro_video_path: str,
    pro_phases: dict,
    dtw_alignment: dict = None,
    show_pose: bool = True,
    thumb_height: int = 320,
    model_path: str = None,
) -> plt.Figure:
    """
    Generate a matplotlib grid comparing student vs. pro at each swing phase.
    """
    phases = list(student_phases.keys())
    n      = len(phases)

    fig, axes = plt.subplots(
        n, 2,
        figsize=(10, n * 3.2),
        facecolor="#0e1117",
    )
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "🏌️  Phase-by-Phase Comparison: Student vs. Pro",
        color="white", fontsize=14, y=1.01,
    )

    for row_idx, phase in enumerate(phases):
        ax_student, ax_pro = axes[row_idx]
        color = PHASE_COLORS.get(phase, "#888888")

        # ─ Student frame ────────────────────────────────────────────────────
        s_start, s_end = student_phases[phase]
        s_frame_idx = (
            dtw_alignment[phase][0]
            if dtw_alignment and phase in dtw_alignment
            else (s_start + s_end) // 2
        )
        student_frame = _extract_frame(student_video_path, s_frame_idx)

        # ─ Pro frame ────────────────────────────────────────────────────────
        if dtw_alignment and phase in dtw_alignment:
            p_frame_idx = dtw_alignment[phase][1]
            dtw_dist    = dtw_alignment[phase][2]
        else:
            p_start, p_end = pro_phases.get(phase, (0, 0))
            p_frame_idx    = (p_start + p_end) // 2
            dtw_dist       = None

        pro_frame = _extract_frame(pro_video_path, p_frame_idx)

        # ─ Pose overlay ─────────────────────────────────────────────────────
        if show_pose:
            if student_frame is not None:
                student_frame = _draw_pose_overlay(student_frame.copy(), model_path)
            if pro_frame is not None:
                pro_frame = _draw_pose_overlay(pro_frame.copy(), model_path)

        # ─ Render ───────────────────────────────────────────────────────────
        dtw_label = f"Frame {p_frame_idx}" + ("  (DTW ✓)" if dtw_dist is not None else "")

        _render_frame_in_ax(ax_student, student_frame, thumb_height,
                            f"Student — {phase}", color, f"Frame {s_frame_idx}")
        _render_frame_in_ax(ax_pro, pro_frame, thumb_height,
                            f"Pro — {phase}", color, dtw_label)

        for ax in (ax_student, ax_pro):
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

    fig.tight_layout(pad=0.5)
    return fig


def _render_frame_in_ax(ax, frame, thumb_height, title, title_color, subtitle):
    ax.set_facecolor("#0e1117")
    ax.set_xticks([])
    ax.set_yticks([])

    if frame is None:
        ax.text(0.5, 0.5, "Frame\nnot found", color="#94a3b8",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = thumb_height / h
        rgb = cv2.resize(rgb, (int(w * scale), thumb_height))
        ax.imshow(rgb)

    ax.set_title(title, color=title_color, fontsize=9, pad=4, fontweight="bold")
    ax.set_xlabel(subtitle, color="#64748b", fontsize=7, labelpad=2)
