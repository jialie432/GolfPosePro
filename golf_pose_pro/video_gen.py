"""
video_gen.py — Annotated debug video generator.
Produces a side-by-side MP4: original video frames alongside a live
wrist-trajectory plot with phase labels. Optionally slow-motion in swing window.
"""

import gc
import time
import subprocess

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .phase_detection import PHASE_COLORS, get_phase_label


def generate_debug_video(
    video_path: str,
    output_path: str,
    wrist_y: np.ndarray,
    smoothed: np.ndarray,
    phase_ranges: dict,
    swing_start: int,
    swing_end: int,
    slow_factor: float = 2.0,
    extra_series: dict = None,
    progress_callback=None,
) -> str:
    """
    Generate an annotated debug video with a live wrist-trajectory overlay.

    Args:
        video_path:        Input video path.
        output_path:       Output MP4 path (will be re-encoded to *_playable.mp4).
        wrist_y:           Raw wrist-Y array.
        smoothed:          Smoothed wrist-Y array.
        phase_ranges:      Phase boundary dict.
        swing_start/end:   Swing window frame indices.
        slow_factor:       Slow-motion multiplier for swing segment. 1.0 = no slow-mo.
        extra_series:      Optional {label: np.ndarray} for hip/shoulder lines.
        progress_callback: Optional callable(frame_idx, total_frames).

    Returns:
        Path to the re-encoded *_playable.mp4 file.
    """
    cap         = cv2.VideoCapture(video_path)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0

    plot_width  = int(width * 2.0)
    output_size = (width + plot_width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    fig, ax = plt.subplots(
        figsize=(plot_width / 100.0, height / 100.0), dpi=100,
        facecolor="#0e1117",
    )
    ax.set_facecolor("#0e1117")
    canvas = FigureCanvas(fig)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame_idx >= len(wrist_y):
            break

        try:
            ax.clear()
            ax.set_facecolor("#0e1117")

            x = np.arange(len(wrist_y))
            ax.plot(x, wrist_y, color="#94a3b8", alpha=0.35, linewidth=1)
            ax.plot(x, smoothed, color="#4ade80", linewidth=2, label="Wrist Y")

            if extra_series:
                palette = ["#fb923c", "#38bdf8"]
                for i, (lbl, series) in enumerate(extra_series.items()):
                    ax.plot(
                        np.arange(len(series)), series,
                        color=palette[i % len(palette)],
                        linewidth=1.6, linestyle="--", label=lbl, alpha=0.85,
                    )

            # Phase shading
            for phase, (s, e) in phase_ranges.items():
                c = PHASE_COLORS.get(phase, "#888888")
                ax.axvspan(s, e, alpha=0.15, color=c)

            # Current frame cursor
            ax.axvline(frame_idx, color="white", linewidth=1.5, alpha=0.9)
            ax.axvspan(swing_start, swing_end, color="#22c55e", alpha=0.08)

            # Phase label
            phase_label = get_phase_label(frame_idx, phase_ranges)
            if phase_label:
                pcolor = PHASE_COLORS.get(phase_label, "#ffffff")
                ax.text(
                    0.02, 0.96, phase_label,
                    transform=ax.transAxes,
                    fontsize=20, color=pcolor, fontweight="bold",
                    va="top", ha="left",
                    bbox=dict(facecolor="#0e1117", alpha=0.6, boxstyle="round,pad=0.3"),
                )

            ax.set_title(f"Frame {frame_idx} / {total_frames}", color="white", fontsize=14, pad=8)
            ax.set_xlabel("Frame", color="#94a3b8", fontsize=11)
            ax.set_ylabel("Wrist Y (inverted)", color="#94a3b8", fontsize=11)
            ax.set_xlim(0, len(wrist_y))
            ax.set_ylim(np.nanmax(wrist_y) + 15, np.nanmin(wrist_y) - 15)
            ax.tick_params(colors="#64748b")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
            ax.legend(loc="upper right", fontsize=9, facecolor="#1e293b",
                      labelcolor="white", framealpha=0.7)

            fig.tight_layout()
            canvas.draw()

            plot_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(
                canvas.get_width_height()[1], canvas.get_width_height()[0], 4
            )[..., :3]

            plot_img    = cv2.resize(plot_img, (plot_width, height))
            frame_resized = cv2.resize(frame, (width, height))
            combined      = np.hstack((frame_resized, plot_img))
            out.write(combined)

            frame_idx += 1
            gc.collect()

            if progress_callback:
                progress_callback(frame_idx, total_frames)

        except Exception as e:
            print(f"⚠️  Frame {frame_idx} error: {e}")
            frame_idx += 1
            continue

    cap.release()
    out.release()
    plt.close(fig)

    # ─ Re-encode with slow-motion ─────────────────────────────────────────
    playable_path = output_path.replace(".mp4", "_playable.mp4")
    _reencode_with_slowmo(output_path, playable_path, fps, swing_start, swing_end, slow_factor)
    return playable_path


def _reencode_with_slowmo(
    input_path: str,
    output_path: str,
    fps: float,
    swing_start: int,
    swing_end: int,
    slow_factor: float,
):
    """FFmpeg re-encode with a slow-motion segment over the swing window."""
    start_t = swing_start / fps
    end_t   = swing_end   / fps

    filter_complex = (
        f"[0:v]trim=0:{start_t:.4f},setpts=PTS-STARTPTS[v1];"
        f"[0:v]trim={start_t:.4f}:{end_t:.4f},setpts={slow_factor:.2f}*(PTS-STARTPTS)[v2];"
        f"[0:v]trim={end_t:.4f},setpts=PTS-STARTPTS[v3];"
        f"[v1][v2][v3]concat=n=3:v=1:a=0[outv]"
    )

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-vcodec", "libx264", "-crf", "23", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    time.sleep(0.5)
