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

# All MediaPipe Pose left/right landmark index pairs (left, right)
_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]

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
    apply_smoothing: bool = True,
    smoothing_min_cutoff: float = 1.0,
    smoothing_beta: float = 0.007,
) -> list:
    """
    Extract all 33 world landmarks (x, y, z in meters) per frame.

    Uses pose_world_landmarks which provides 3D coordinates relative
    to the hip center — suitable for Three.js rendering.

    A brief transient left/right identity swap (BlazePose occasionally
    mislabels a whole side for a few frames during fast rotation, e.g. a golf
    follow-through) is auto-corrected, an arm that the camera can't see
    (hidden behind the other arm at the top of the backswing / behind the
    body through the finish) is rebuilt from the visible arm, then a One-Euro
    filter smooths each landmark's position over time to remove per-frame
    depth-estimation jitter.

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
    _fix_left_right_swaps(frames_3d)
    _fix_occluded_arms(frames_3d)
    if apply_smoothing:
        from .smoothing import smooth_landmarks_3d
        smooth_landmarks_3d(
            frames_3d, fps=fps,
            min_cutoff=smoothing_min_cutoff, beta=smoothing_beta,
        )
    return frames_3d


def _dist2(p: tuple, q: tuple) -> float:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2


def _fix_left_right_swaps(
    frames_3d: list, min_pairs: int = 4, margin: float = 0.5, ref_alpha: float = 0.25,
) -> list:
    """
    Correct transient left/right identity swaps in MediaPipe world landmarks.

    During fast rotational motion (e.g. a golf follow-through, where the body
    turns through ~90°+), BlazePose occasionally mislabels an entire side for
    a handful of consecutive frames — the left leg/arm landmarks momentarily
    carry the right side's positions and vice versa, which renders as the
    legs/arms crossing through the body in the 3D viewer. For each frame,
    compare the cost of keeping vs. swapping every left/right landmark pair
    against a reference position, and swap back when that's clearly the
    better match (modifies frames_3d in place; also returned).

    The reference is a slow exponential moving average rather than the raw
    previous frame: right at the moment the legs cross, MediaPipe's own
    reading is briefly ambiguous (both sides land near the same position), so
    anchoring to the single prior frame lets one bad ambiguous frame corrupt
    the reference and permanently invert every frame after it. Blending
    slowly keeps a memory of the last confidently-labeled pose, so a single
    ambiguous frame nudges it only slightly instead of flipping it.
    """
    ref = {}  # idx -> (x, y, z), slow-moving reference

    for frame in frames_3d:
        lm_by_idx = {l["idx"]: l for l in frame["landmarks"] if l["x"] is not None}

        keep_cost = 0.0
        swap_cost = 0.0
        swappable = []
        for a, b in _LR_PAIRS:
            la, lb = lm_by_idx.get(a), lm_by_idx.get(b)
            ra, rb = ref.get(a), ref.get(b)
            if la is None or lb is None or ra is None or rb is None:
                continue
            swappable.append((a, b))
            pa, pb = (la["x"], la["y"], la["z"]), (lb["x"], lb["y"], lb["z"])
            keep_cost += _dist2(pa, ra) + _dist2(pb, rb)
            swap_cost += _dist2(pa, rb) + _dist2(pb, ra)

        if len(swappable) >= min_pairs and swap_cost < keep_cost * margin:
            for a, b in swappable:
                la, lb = lm_by_idx[a], lm_by_idx[b]
                la["x"], lb["x"] = lb["x"], la["x"]
                la["y"], lb["y"] = lb["y"], la["y"]
                la["z"], lb["z"] = lb["z"], la["z"]
                la["visibility"], lb["visibility"] = lb["visibility"], la["visibility"]

        for l in frame["landmarks"]:
            if l["x"] is None:
                continue
            p = (l["x"], l["y"], l["z"])
            prev = ref.get(l["idx"])
            ref[l["idx"]] = p if prev is None else (
                ref_alpha * p[0] + (1 - ref_alpha) * prev[0],
                ref_alpha * p[1] + (1 - ref_alpha) * prev[1],
                ref_alpha * p[2] + (1 - ref_alpha) * prev[2],
            )

    return frames_3d


# ─── Occluded-arm reconstruction ─────────────────────────────────────────────

# (shoulder, elbow, wrist) landmark indices, per arm
_ARM_CHAINS = {"left": (11, 13, 15), "right": (12, 14, 16)}

# (pinky, index, thumb) landmark indices, per hand. These ride along with the
# wrist when an arm is rebuilt: the 3D viewer reads them to work out how the
# hand is rolled around the club shaft, so leaving the originals behind at the
# hallucinated wrist would hand it a grip orientation from thin air.
_HAND_TIPS = {"left": (17, 19, 21), "right": (18, 20, 22)}


def _unit(v):
    """Normalize a vector, or None if it's too short to have a direction."""
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else None


def _torso_basis(pts: dict):
    """
    Orthonormal (right, up, fwd) basis rigidly attached to the torso, built
    from the shoulder and hip landmarks. None if any of them is missing.

    Occluded-arm geometry is carried across a gap in THIS frame rather than in
    world space: the golfer's torso turns through ~180° between the top of the
    backswing and the finish, so a relationship that is near-constant in torso
    coordinates ("the trail elbow points down and behind the chest") sweeps
    through an enormous arc in world coordinates.
    """
    ls, rs, lh, rh = pts.get(11), pts.get(12), pts.get(23), pts.get(24)
    if ls is None or rs is None or lh is None or rh is None:
        return None
    # MediaPipe world Y grows downward, so shoulders − hips points anatomically up.
    up = _unit((ls + rs) / 2.0 - (lh + rh) / 2.0)
    if up is None:
        return None
    right = rs - ls
    right = _unit(right - float(np.dot(right, up)) * up)
    if right is None:
        return None
    return right, up, np.cross(up, right)


def _to_local(v, basis):
    right, up, fwd = basis
    return np.array([float(np.dot(v, right)), float(np.dot(v, up)), float(np.dot(v, fwd))])


def _to_world(v, basis):
    right, up, fwd = basis
    return right * v[0] + up * v[1] + fwd * v[2]


def _elbow_pole(shoulder, elbow, wrist):
    """
    The direction the elbow juts out from the shoulder→wrist axis — the one
    degree of freedom a two-bone arm has left once both endpoints are fixed.
    """
    axis = _unit(wrist - shoulder)
    if axis is None:
        return None
    d = elbow - shoulder
    return _unit(d - float(np.dot(d, axis)) * axis)


def _solve_elbow(shoulder, wrist, upper_len, fore_len, pole):
    """
    Two-bone IK: place the elbow so the upper arm and forearm keep their
    measured lengths while spanning shoulder→wrist, with `pole` (a world-space
    direction) choosing which point on the resulting circle of solutions to
    use. Falls back to a straight arm when no pole direction is available.
    """
    reach = wrist - shoulder
    d = float(np.linalg.norm(reach))
    if d < 1e-6 or upper_len <= 0 or fore_len <= 0:
        return None
    axis = reach / d
    if d >= upper_len + fore_len:
        # Target sits past a straight arm. Straighten it and let both bones
        # share the overreach in proportion, rather than pinning the elbow at
        # the triangle's limit and leaving the entire stretch on the forearm.
        return shoulder + axis * d * (upper_len / (upper_len + fore_len))
    # Keep the triangle solvable when the wrist folds in close to the shoulder,
    # nearer than the two bones can ever bring it.
    d = max(d, abs(upper_len - fore_len) + 1e-4)
    along = (d * d + upper_len ** 2 - fore_len ** 2) / (2.0 * d)
    out = float(np.sqrt(max(upper_len ** 2 - along ** 2, 0.0)))

    base = shoulder + axis * along
    if pole is None:
        return base
    p = _unit(pole - float(np.dot(pole, axis)) * axis)
    return base if p is None else base + p * out


def _interp_vectors(series: list) -> list:
    """
    Fill None gaps in a list of vectors by linear interpolation, padding the
    ends with the nearest known value. Used to carry the arm's shape across an
    occluded stretch from the confident frames on either side of it.
    """
    known = [i for i, v in enumerate(series) if v is not None]
    if not known:
        return list(series)

    out = list(series)
    for i in range(known[0]):
        out[i] = series[known[0]].copy()
    for i in range(known[-1] + 1, len(series)):
        out[i] = series[known[-1]].copy()
    for a, b in zip(known, known[1:]):
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            out[i] = (1.0 - t) * series[a] + t * series[b]
    return out


def _set_landmark(frame: dict, idx: int, p) -> None:
    for l in frame["landmarks"]:
        if l["idx"] == idx:
            l["x"], l["y"], l["z"] = (round(float(p[0]), 5),
                                      round(float(p[1]), 5),
                                      round(float(p[2]), 5))
            return


def _fix_occluded_arms(
    frames_3d: list,
    vis_margin: float = 0.05,
    vis_reliable: float = 0.85,
    span_factor: float = 1.6,
    bone_tol: tuple = (0.7, 1.4),
    min_calibration_frac: float = 0.1,
    reach_tol: float = 1.25,
) -> list:
    """
    Rebuild an arm the camera cannot see from the arm it can.

    Twice per swing an arm disappears from view: at the top of the backswing
    the trail arm hides behind the lead arm and the chest, and through the
    finish the lead arm hides behind the body. BlazePose does not report those
    joints as missing — it invents them, and the invented arm drifts away from
    the club (on TigerWoodsDriver.mov the right wrist wanders ~0.5 m from the
    left from frame ~262 on, while both hands are in fact together on the
    grip). Downstream that shows up as a broken-looking right arm in the 3D
    viewer AND as a displaced club, since the grip point is the wrist midpoint.

    Detection leans on what is physically impossible rather than on the
    visibility score alone (which sags well before the pose actually breaks).
    An arm is treated as invented when it does something a real arm cannot —
      * both hands share a grip through a golf swing, so wrists further apart
        than a couple of hand widths cannot both be right;
      * bones do not change length, so an upper arm or forearm outside
        `bone_tol` × its own measured length is fabricated;
      * a joint MediaPipe dropped outright is missing —
    AND this arm is the less confidently tracked of the two, by at least
    `vis_margin`. That second half is what identifies WHICH of the two arms is
    the broken one; when both sides are equally confident there is no evidence
    for blaming either, so the frame is left alone.

    Such an arm is replaced rather than trusted:
      * the wrist is placed relative to the visible wrist using the
        hand-to-hand offset measured on the confident frames bracketing the
        occlusion, expressed in torso coordinates (see `_torso_basis`) so it
        rotates with the body;
      * the elbow is re-solved by two-bone IK at the arm's measured bone
        lengths, using an elbow-out direction interpolated the same way.

    Both the offset and the bone lengths are measured from this video's own
    confident frames, so no assumption about body size is baked in. That also
    means the whole pass needs a decent supply of them: on footage MediaPipe
    barely tracks at all (subject too small or too dark — no arm clearly seen
    for `min_calibration_frac` of the video) the proportions being
    reconstructed from would themselves be guesses, so the landmarks are left
    exactly as they are. Individual frames are likewise skipped when the
    rebuilt wrist lands beyond the arm's reach, which means the visible arm
    and the shoulders disagree and there is nothing solid to build on.

    Frames where the arm is genuinely visible are left completely untouched.

    Modifies frames_3d in place (also returned). Runs before smoothing so the
    One-Euro filter blends the seams at the ends of each reconstructed run.
    """
    n = len(frames_3d)
    if n == 0:
        return frames_3d

    # ── Per-frame geometry ──────────────────────────────────────────────────
    pts_per_frame, vis_per_frame, basis_per_frame = [], [], []
    for frame in frames_3d:
        pts, vis = {}, {}
        for l in frame["landmarks"]:
            vis[l["idx"]] = l.get("visibility") or 0.0
            if l["x"] is not None:
                pts[l["idx"]] = np.array([l["x"], l["y"], l["z"]], dtype=float)
        pts_per_frame.append(pts)
        vis_per_frame.append(vis)
        basis_per_frame.append(_torso_basis(pts))

    def side_vis(fi, side):
        _, e, w = _ARM_CHAINS[side]
        return min(vis_per_frame[fi].get(e, 0.0), vis_per_frame[fi].get(w, 0.0))

    def has_chain(fi, side):
        return all(i in pts_per_frame[fi] for i in _ARM_CHAINS[side])

    # ── Calibrate on frames where BOTH arms are clearly visible, so the
    #    hallucinated poses can't skew the proportions we reconstruct from ──
    spans, shoulder_widths = [], []
    bones = {side: {"upper": [], "fore": []} for side in _ARM_CHAINS}
    for fi in range(n):
        pts = pts_per_frame[fi]
        if 11 in pts and 12 in pts:
            shoulder_widths.append(float(np.linalg.norm(pts[11] - pts[12])))
        if not all(has_chain(fi, s) and side_vis(fi, s) >= vis_reliable
                   for s in _ARM_CHAINS):
            continue
        spans.append(float(np.linalg.norm(pts[15] - pts[16])))
        for side, (s, e, w) in _ARM_CHAINS.items():
            bones[side]["upper"].append(float(np.linalg.norm(pts[s] - pts[e])))
            bones[side]["fore"].append(float(np.linalg.norm(pts[e] - pts[w])))

    if len(spans) < max(15, min_calibration_frac * n):
        return frames_3d  # too little clean tracking to calibrate against

    shoulder_w = float(np.median(shoulder_widths)) if shoulder_widths else 0.0
    max_span = max(span_factor * float(np.median(spans)), 0.5 * shoulder_w)
    lengths = {side: (float(np.median(bones[side]["upper"])),
                      float(np.median(bones[side]["fore"])))
               for side in _ARM_CHAINS}

    # ── Reconstruct, one side at a time ─────────────────────────────────────
    for side, other in (("left", "right"), ("right", "left")):
        s_i, e_i, w_i = _ARM_CHAINS[side]
        o_w = _ARM_CHAINS[other][2]
        upper_len, fore_len = lengths[side]

        def impossible(fi):
            pts = pts_per_frame[fi]
            if w_i not in pts or e_i not in pts:
                return True   # MediaPipe gave up on the joint entirely
            if np.linalg.norm(pts[w_i] - pts[o_w]) > max_span:
                return True   # hands too far apart to be sharing one grip
            for a, b, ref in ((s_i, e_i, upper_len), (e_i, w_i, fore_len)):
                if a not in pts:
                    continue
                d = float(np.linalg.norm(pts[a] - pts[b]))
                if not (bone_tol[0] * ref <= d <= bone_tol[1] * ref):
                    return True   # bone squashed or stretched out of existence
            return False

        bad = [False] * n
        for fi in range(n):
            if (basis_per_frame[fi] is None or not has_chain(fi, other)
                    or side_vis(fi, side) > side_vis(fi, other) - vis_margin):
                continue
            bad[fi] = impossible(fi)

        if not any(bad):
            continue

        # Anchors: what this arm looks like on the frames we DO trust. A frame
        # that fails the physical checks is excluded even when its visibility
        # looks fine, so a half-corrupted frame at the edge of an occlusion
        # can't seed the reconstruction with its own error.
        offset_local = [None] * n   # this wrist relative to the other, torso coords
        pole_local   = [None] * n   # elbow-out direction, torso coords
        tips_local   = {t: [None] * n for t in _HAND_TIPS[side]}  # hand shape, torso coords
        for fi in range(n):
            pts, basis = pts_per_frame[fi], basis_per_frame[fi]
            if (basis is None or not has_chain(fi, side)
                    or side_vis(fi, side) < vis_reliable
                    or o_w not in pts or impossible(fi)):
                continue
            offset_local[fi] = _to_local(pts[w_i] - pts[o_w], basis)
            pole = _elbow_pole(pts[s_i], pts[e_i], pts[w_i])
            if pole is not None:
                pole_local[fi] = _to_local(pole, basis)
            for t in _HAND_TIPS[side]:
                if t in pts:
                    tips_local[t][fi] = _to_local(pts[t] - pts[w_i], basis)

        offsets = _interp_vectors(offset_local)
        poles   = _interp_vectors(pole_local)
        tips    = {t: _interp_vectors(v) for t, v in tips_local.items()}

        for fi in range(n):
            if not bad[fi]:
                continue
            pts, basis = pts_per_frame[fi], basis_per_frame[fi]
            if s_i not in pts or o_w not in pts:
                continue

            off = offsets[fi]
            wrist = pts[o_w] + (_to_world(off, basis) if off is not None
                                else np.zeros(3))
            # Out of reach means the visible wrist and this shoulder disagree;
            # forcing it would just trade a broken arm for a stretched one.
            # `reach_tol` sits well clear of a straight arm — MediaPipe's world
            # landmarks put a fully extended arm at up to ~1.18× the summed
            # median bone lengths, so a tighter bound would reject real poses.
            if np.linalg.norm(wrist - pts[s_i]) > (upper_len + fore_len) * reach_tol:
                continue
            pole = _to_world(poles[fi], basis) if poles[fi] is not None else None
            elbow = _solve_elbow(pts[s_i], wrist, upper_len, fore_len, pole)

            _set_landmark(frames_3d[fi], w_i, wrist)
            pts[w_i] = wrist
            if elbow is not None:
                _set_landmark(frames_3d[fi], e_i, elbow)
                pts[e_i] = elbow
            # Carry the hand's own shape (pinky/index/thumb) to the new wrist,
            # so the viewer can still read which way the hand is rolled.
            for t in _HAND_TIPS[side]:
                if tips[t][fi] is None:
                    continue
                tip = wrist + _to_world(tips[t][fi], basis)
                _set_landmark(frames_3d[fi], t, tip)
                pts[t] = tip

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
