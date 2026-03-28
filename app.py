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
