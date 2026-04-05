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
