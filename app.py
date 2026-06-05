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
from golf_pose_pro.club_estimation import estimate_club_positions, calculate_club_head_speed
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
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=Nunito:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

:root {
    --gold:         #c9943a;
    --gold-light:   #e8c068;
    --gold-dim:     rgba(201,148,58,0.12);
    --gold-border:  rgba(201,148,58,0.18);
    --gold-border2: rgba(201,148,58,0.32);
    --text-1: #f2eed8;
    --text-2: #9aaa95;
    --text-3: #5e7259;
    --card:   rgba(255,255,255,0.035);
}

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif !important;
}

/* ── Background ── */
.stApp {
    background: radial-gradient(ellipse at 30% 15%, #0e1f14 0%, #080d08 55%, #070a10 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(5,8,5,0.98) !important;
    border-right: 1px solid var(--gold-border);
}
[data-testid="stSidebar"] * { color: var(--text-1) !important; }

.sb-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 900;
    color: var(--text-1) !important;
    letter-spacing: -0.01em;
}
.sb-brand em { font-style: italic; color: var(--gold) !important; }

.sb-step {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.65rem 0 0.25rem;
    margin-top: 0.6rem;
    border-top: 1px solid rgba(201,148,58,0.1);
}
.sb-num {
    width: 21px;
    height: 21px;
    border-radius: 50%;
    background: var(--gold);
    color: #050805 !important;
    font-size: 0.66rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.sb-label {
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--gold) !important;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(140deg, rgba(14,31,20,0.92) 0%, rgba(8,13,8,0.9) 55%, rgba(11,14,22,0.92) 100%);
    border: 1px solid var(--gold-border2);
    border-radius: 18px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::after {
    content: '';
    position: absolute;
    top: -50%;
    right: 3%;
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(201,148,58,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Fira Code', monospace;
    font-size: 0.67rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.45rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.75rem;
    font-weight: 900;
    color: var(--text-1);
    line-height: 1.05;
    margin: 0;
}
.hero-title em { font-style: italic; color: var(--gold); }
.hero-sub {
    color: var(--text-2);
    font-size: 0.97rem;
    margin-top: 0.5rem;
    font-weight: 400;
    max-width: 580px;
    line-height: 1.65;
}
.hero-pills {
    display: flex;
    gap: 0.6rem;
    margin-top: 1.1rem;
    flex-wrap: wrap;
}
.hero-pill {
    background: var(--gold-dim);
    border: 1px solid rgba(201,148,58,0.22);
    border-radius: 100px;
    padding: 3px 11px;
    font-size: 0.71rem;
    font-weight: 600;
    color: var(--gold-light);
    letter-spacing: 0.04em;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--card);
    border: 1px solid var(--gold-border);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--gold-border2);
    box-shadow: 0 10px 35px rgba(201,148,58,0.12);
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.25rem;
    font-weight: 700;
    color: var(--gold);
    line-height: 1;
}
.metric-label {
    font-family: 'Fira Code', monospace;
    font-size: 0.62rem;
    color: var(--text-3);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.35rem;
}
.metric-sub {
    font-size: 0.78rem;
    color: var(--text-2);
    margin-top: 0.2rem;
}

/* ── Phase timeline ── */
.phase-row {
    display: flex;
    border-radius: 8px;
    overflow: hidden;
    height: 30px;
    margin: 0.7rem 0 0.4rem;
    box-shadow: 0 2px 14px rgba(0,0,0,0.4);
}
.phase-seg {
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Fira Code', monospace;
    font-size: 0.58rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: rgba(255,255,255,0.92);
    overflow: hidden;
    white-space: nowrap;
    text-shadow: 0 1px 3px rgba(0,0,0,0.55);
}
.phase-legend {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 0.4rem;
}
.phase-legend-item {
    display: flex;
    align-items: center;
    gap: 0.32rem;
    font-family: 'Fira Code', monospace;
    font-size: 0.67rem;
    color: var(--text-2);
}
.phase-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Section cards & titles ── */
.section-card {
    background: rgba(255,255,255,0.022);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.08rem;
    font-weight: 700;
    color: var(--text-1);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #c9943a, #9d7228) !important;
    color: #050805 !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 16px rgba(201,148,58,0.3) !important;
    letter-spacing: 0.04em !important;
    font-size: 0.9rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(201,148,58,0.44) !important;
    background: linear-gradient(135deg, #e8c068, #c9943a) !important;
}
.stDownloadButton > button {
    background: rgba(40,52,40,0.45) !important;
    border: 1px solid rgba(201,148,58,0.2) !important;
    color: var(--text-1) !important;
    border-radius: 8px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Tabs ── */
.stTabs [role="tablist"] {
    background: rgba(255,255,255,0.02);
    border-radius: 10px;
    padding: 3px;
    border: 1px solid rgba(255,255,255,0.05);
}
.stTabs [role="tab"] {
    border-radius: 7px;
    color: var(--text-3) !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    transition: all 0.2s !important;
}
.stTabs [role="tab"][aria-selected="true"] {
    background: var(--gold-dim) !important;
    color: var(--gold) !important;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #9d7228, #e8c068) !important;
    border-radius: 4px !important;
}

/* ── Alerts ── */
.stAlert {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
}

/* ── Inputs ── */
.stSlider > div > div > div { background: var(--gold) !important; }
.stSelectbox div[data-baseweb], .stMultiSelect div[data-baseweb] {
    background: rgba(255,255,255,0.03) !important;
    border-color: var(--gold-border) !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(201,148,58,0.22); border-radius: 3px; }

/* ── Welcome state ── */
.welcome-wrap {
    max-width: 680px;
    margin: 1.5rem auto;
    text-align: center;
}
.welcome-icon { font-size: 3.5rem; line-height: 1; margin-bottom: 0.8rem; }
.welcome-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--text-1);
    margin-bottom: 0.4rem;
}
.welcome-sub {
    color: var(--text-2);
    font-size: 0.95rem;
    line-height: 1.65;
    margin-bottom: 2rem;
}
.welcome-steps {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 1rem;
    text-align: left;
}
.w-step {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.1rem 1.15rem;
}
.w-step-num {
    font-family: 'Fira Code', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.1em;
    color: var(--gold);
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.w-step-title { font-weight: 700; font-size: 0.9rem; color: var(--text-1); margin-bottom: 0.22rem; }
.w-step-desc  { font-size: 0.8rem; color: var(--text-2); line-height: 1.5; }
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
    """Build an HTML phase-timeline bar with a color legend."""
    segs = ""
    legend = ""
    for phase, (s, e) in phase_ranges.items():
        pct   = max(1.0, (e - s) / total_frames * 100)
        color = PHASE_COLORS.get(phase, "#888")
        segs += (
            f'<div class="phase-seg" style="flex:{pct:.2f};background:{color};">'
            f'{phase}</div>'
        )
        legend += (
            f'<span class="phase-legend-item">'
            f'<span class="phase-dot" style="background:{color};"></span>'
            f'{phase}</span>'
        )
    return (
        f'<div class="phase-row">{segs}</div>'
        f'<div class="phase-legend">{legend}</div>'
    )


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
    st.markdown('<p class="sb-brand">Golf<em>Pose</em>Pro ⛳</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Step 1 ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-step"><span class="sb-num">1</span>'
        '<span class="sb-label">Upload Your Swing</span></div>',
        unsafe_allow_html=True,
    )
    student_file = st.file_uploader(
        "Your swing video",
        type=["mp4", "mov", "MP4", "MOV"],
        help="Record yourself hitting a shot and upload it here.",
        key="student_upload",
        label_visibility="collapsed",
    )
    if student_file:
        st.success("✓ Swing video loaded", icon=None)

    # ── Step 2 ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-step"><span class="sb-num">2</span>'
        '<span class="sb-label">Pick a Pro to Compare</span></div>',
        unsafe_allow_html=True,
    )

    PROJ_DIR  = Path(__file__).parent
    BUILTIN   = {}
    for name, path in [
        ("Max Homa — Iron",       PROJ_DIR / "input_videos" / "max_homa_iron_fixed.MP4"),
        ("Ludvig Åberg — Driver", PROJ_DIR / "input_videos" / "ludvig_aberg_driver_fixed.MP4"),
    ]:
        if path.exists():
            BUILTIN[name] = str(path)

    pro_source = st.radio(
        "Pro reference source",
        ["Use built-in pro", "Upload custom"],
        horizontal=True,
        label_visibility="collapsed",
    )

    pro_file = None
    builtin_pro_path = None

    if pro_source == "Upload custom":
        pro_file = st.file_uploader(
            "Upload pro reference",
            type=["mp4", "mov", "MP4", "MOV"],
            key="pro_upload",
        )
    elif BUILTIN:
        chosen_pro = st.selectbox("Choose a pro golfer", list(BUILTIN.keys()))
        builtin_pro_path = BUILTIN[chosen_pro]
        st.caption("Built-in reference loaded. You can skip pro comparison too.")
    else:
        st.warning("No built-in pro videos found.")

    # ── Step 3 ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-step"><span class="sb-num">3</span>'
        '<span class="sb-label">Analyze</span></div>',
        unsafe_allow_html=True,
    )
    analyze_btn = st.button("⚡ Analyze Swing", use_container_width=True)

    # ── Advanced settings (collapsed) ─────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.markdown("**Landmark Tracking**")
        track_options = st.multiselect(
            "Track landmarks",
            ["wrist", "hip", "shoulder"],
            default=["wrist"],
            help="Wrist is primary. Add hip/shoulder for multi-landmark overlay.",
        )
        if not track_options:
            track_options = ["wrist"]

        st.markdown("**Phase Detection**")
        smoothing = st.slider("Smoothing window", 3, 20, 5, 1,
                              help="Bigger = smoother trajectory, less reactive to noise.")
        threshold_pct = st.slider("Motion threshold (%ile)", 70, 99, 90, 1,
                                  help="Higher = only stronger motions define phase boundaries.")

        st.markdown("**Debug Video**")
        slow_factor = st.slider("Slow-motion factor", 1.0, 4.0, 2.0, 0.5,
                                help="Multiplier applied to the swing window in the debug video.")
        show_pose = st.checkbox("Show skeleton overlay in comparison", value=True)
        use_dtw   = st.checkbox("Use DTW frame matching", value=True,
                                help="Dynamic Time Warping aligns student/pro phases for fair comparison.")

        st.markdown("**Adaptive Smoothing**")
        use_one_euro = st.checkbox("One-Euro Filter", value=True,
                                   help="Reduces jitter: heavy smoothing for slow motion, light for fast.")
        if use_one_euro:
            oe_min_cutoff = st.slider("Min cutoff", 0.3, 5.0, 1.0, 0.1,
                                      help="Lower = more smoothing. Typical 0.5–3.0.")
            oe_beta = st.slider("Beta", 0.0, 0.1, 0.007, 0.001,
                                help="Cutoff adaptation speed. Typical 0.001–0.05.",
                                format="%.3f")
        else:
            oe_min_cutoff = 1.0
            oe_beta = 0.007

        st.markdown("**Club Head Speed**")
        capture_fps_input = st.number_input(
            "Capture frame rate (fps)",
            min_value=24.0, max_value=1000.0, value=30.0, step=60.0,
            help="Actual recording fps. For 480 fps slow-motion played back at 30 fps, enter 480.",
        )
        _shaft_presets = {
            "Driver (1.07 m)":  1.07,
            "3-Wood (0.97 m)":  0.97,
            "Iron (0.89 m)":    0.89,
        }
        shaft_preset = st.selectbox(
            "Club type",
            list(_shaft_presets.keys()) + ["Custom"],
            index=0,
            help="Sets the physical grip-to-head shaft length used for speed calibration.",
        )
        if shaft_preset == "Custom":
            physical_shaft_m_input = st.number_input(
                "Shaft length (m)", min_value=0.7, max_value=1.3,
                value=1.07, step=0.01,
            )
        else:
            physical_shaft_m_input = _shaft_presets[shaft_preset]


# ────────────────────────────────────────────────────────────────────────────
# Hero header
# ────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <div class="hero-eyebrow">AI-Powered Golf Swing Analysis</div>
  <h1 class="hero-title">Golf<em>Pose</em>Pro</h1>
  <p class="hero-sub">
    Upload your swing, compare against a professional, and get instant
    biomechanical feedback — powered by MediaPipe AI and Dynamic Time Warping.
  </p>
  <div class="hero-pills">
    <span class="hero-pill">⚡ Phase Detection</span>
    <span class="hero-pill">📐 DTW Alignment</span>
    <span class="hero-pill">🎯 3D Kinematics</span>
    <span class="hero-pill">📹 Live Tracking</span>
    <span class="hero-pill">📊 Export Data</span>
  </div>
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
    update_progress(76, "🏌️ Detecting golf club — student…")
    club_data_s = estimate_club_positions(
        frames_3d_s, phase_ranges_s, video_path=student_path,
    )
    club_speed_s = calculate_club_head_speed(
        club_data_s,
        actual_fps=capture_fps_input,
        physical_shaft_m=physical_shaft_m_input,
    )

    frames_3d_p = None
    club_data_p = None
    club_speed_p = None
    if has_pro:
        update_progress(76, "🎯 Extracting 3D landmarks — pro…")
        frames_3d_p = extract_full_3d_landmarks(pro_path)
        update_progress(77, "🏌️ Detecting golf club — pro…")
        club_data_p = estimate_club_positions(
            frames_3d_p, phase_ranges_p, video_path=pro_path,
        )
        club_speed_p = calculate_club_head_speed(
            club_data_p,
            actual_fps=capture_fps_input,
            physical_shaft_m=physical_shaft_m_input,
        )

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
        "club_speed_s":  club_speed_s,
        "frames_3d_p":   frames_3d_p,
        "club_data_p":   club_data_p,
        "club_speed_p":  club_speed_p,
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
    st.markdown("""
    <div class="welcome-wrap">
      <div class="welcome-icon">⛳</div>
      <h2 class="welcome-title">Analyze Your Golf Swing</h2>
      <p class="welcome-sub">
        Compare your swing against touring professionals frame by frame.<br>
        Get instant biomechanical insights — no golf expertise required.
      </p>
      <div class="welcome-steps">
        <div class="w-step">
          <div class="w-step-num">Step 01</div>
          <div class="w-step-title">Upload Your Video</div>
          <div class="w-step-desc">Film your swing and upload the video using the sidebar on the left.</div>
        </div>
        <div class="w-step">
          <div class="w-step-num">Step 02</div>
          <div class="w-step-title">Choose a Pro</div>
          <div class="w-step-desc">Pick Max Homa or Ludvig Åberg, or upload your own reference video.</div>
        </div>
        <div class="w-step">
          <div class="w-step-num">Step 03</div>
          <div class="w-step-title">Hit Analyze</div>
          <div class="w-step-desc">Click <strong style="color:var(--gold)">Analyze Swing</strong> — results appear in seconds.</div>
        </div>
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
video_fps_r     = R.get("video_fps", 30.0) or 30.0
swing_secs      = swing_duration / max(video_fps_r, 1)

club_speed_s = R.get("club_speed_s") or {}
club_speed_p = R.get("club_speed_p") or {}
speed_mph    = club_speed_s.get("speed_mph")

if speed_mph is not None:
    speed_val = f"{speed_mph:.0f}"
    pro_spd   = club_speed_p.get("speed_mph")
    speed_sub = f"vs pro {pro_spd:.0f} mph" if pro_spd is not None else "grip-to-head at impact"
else:
    speed_val = "—"
    speed_sub = "set capture fps in ⚙️"

kpi_cols = st.columns(5)
kpi_data = [
    (f"{num_phases}", "PHASES", "Address → Follow Through"),
    (f"{swing_secs:.2f}s", "SWING TIME", f"frames {swing_start_s}–{swing_end_s}"),
    (f"{total_frames_s}", "TOTAL FRAMES", f"@ {video_fps_r:.0f} fps"),
    (
        f"{similarity_score:.0f}%",
        "MATCH SCORE",
        "DTW vs. pro" if similarity_score is not None else "—",
    ) if similarity_score is not None else (
        "—", "MATCH SCORE", "no pro video",
    ),
    (f"{speed_val} mph", "CLUB HEAD SPEED", speed_sub),
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
            dtw_tag = f'<span style="color:var(--gold-light,#e8c068);font-size:0.7rem;font-family:\'Fira Code\',monospace">DTW: {d:.1f}</span>'
    phase_table_rows += (
        f"<tr>"
        f"  <td><span style='color:{color};font-weight:700'>{phase}</span></td>"
        f"  <td style='color:#9aaa95;font-family:\"Fira Code\",monospace;font-size:0.82rem'>{s}</td>"
        f"  <td style='color:#9aaa95;font-family:\"Fira Code\",monospace;font-size:0.82rem'>{e}</td>"
        f"  <td style='color:#f2eed8;font-family:\"Fira Code\",monospace;font-size:0.82rem'>{dur}</td>"
        f"  <td style='color:#5e7259;font-family:\"Fira Code\",monospace;font-size:0.82rem'>{pct:.1f}%</td>"
        f"  <td>{dtw_tag}</td>"
        f"</tr>"
    )

st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:0.85rem;margin-top:0.5rem;">
  <thead>
    <tr style="color:#5e7259;border-bottom:1px solid rgba(255,255,255,0.07);font-family:'Fira Code',monospace;font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;">
      <th style="text-align:left;padding:7px 8px">Phase</th>
      <th style="text-align:left;padding:7px 8px">Start</th>
      <th style="text-align:left;padding:7px 8px">End</th>
      <th style="text-align:left;padding:7px 8px">Frames</th>
      <th style="text-align:left;padding:7px 8px">Duration</th>
      <th style="text-align:left;padding:7px 8px">DTW</th>
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
    "⚡ Kinematics",
    "🔄 Compare",
    "🎯 3D View",
    "📹 Live",
    "🎬 Video",
    "📤 Export",
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
                '<span style="background:rgba(201,148,58,0.12);color:#e8c068;'
                'border:1px solid rgba(201,148,58,0.28);border-radius:6px;'
                'padding:2px 10px;font-size:0.72rem;font-weight:700;'
                'font-family:\'Fira Code\',monospace;letter-spacing:0.06em;">DTW ALIGNED</span>'
            )
        st.markdown(
            f'{_section_title("🔄", "Phase-by-Phase Comparison")} {dtw_badge}',
            unsafe_allow_html=True,
        )

        if similarity_score is not None:
            score_color = (
                "#c9943a" if similarity_score >= 70 else
                "#e8c068" if similarity_score >= 40 else "#f87171"
            )
            grade = "A" if similarity_score >= 85 else "B" if similarity_score >= 70 else "C" if similarity_score >= 50 else "D"
            st.markdown(
                f'<div style="text-align:center;margin:0.5rem 0 1.2rem;">'
                f'  <span style="font-family:\'Playfair Display\',serif;font-size:3rem;font-weight:900;color:{score_color}">'
                f'{similarity_score:.0f}%</span>'
                f'  <span style="background:rgba(201,148,58,0.12);border:1px solid rgba(201,148,58,0.25);'
                f'border-radius:6px;padding:2px 10px;font-size:0.9rem;font-weight:700;color:#e8c068;'
                f'margin-left:0.6rem;vertical-align:middle;">Grade {grade}</span>'
                f'  <div style="color:#5e7259;font-size:0.82rem;margin-top:0.3rem;font-family:\'Fira Code\',monospace;">'
                f'DTW swing similarity vs. pro</div>'
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
