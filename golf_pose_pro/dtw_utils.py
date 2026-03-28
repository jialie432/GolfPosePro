"""
dtw_utils.py — Dynamic Time Warping alignment for student vs. pro trajectories.

Uses fastdtw (O(n) approximation) if available, falls back to scipy cdist-based
vanilla DTW for pure-scipy environments.
"""

import numpy as np

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False


# ─── Core DTW ────────────────────────────────────────────────────────────────

def _vanilla_dtw_path(a: np.ndarray, b: np.ndarray) -> list:
    """
    O(n*m) vanilla DTW. Returns alignment path as list of (i, j) index pairs.
    Suitable for sequences of a few hundred frames.
    """
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(float(a[i - 1]) - float(b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Backtrack
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = {
            (i - 1, j - 1): cost[i - 1, j - 1],
            (i - 1, j):     cost[i - 1, j],
            (i,     j - 1): cost[i,     j - 1],
        }
        i, j = min(choices, key=choices.get)
    path.reverse()
    return path


def compute_dtw(a: np.ndarray, b: np.ndarray) -> tuple:
    """
    Compute DTW distance and alignment path between arrays a and b.

    Returns:
        (distance: float, path: list of (i_a, i_b) tuples)
    """
    a = np.nan_to_num(a.astype(float))
    b = np.nan_to_num(b.astype(float))

    if FASTDTW_AVAILABLE:
        dist, path = fastdtw(a.reshape(-1, 1), b.reshape(-1, 1), dist=euclidean)
        return float(dist), path
    else:
        path = _vanilla_dtw_path(a, b)
        dist = sum(abs(float(a[i]) - float(b[j])) for i, j in path)
        return float(dist), path


# ─── Phase-level alignment ───────────────────────────────────────────────────

def align_phase_frames(
    student_phases: dict,
    pro_phases: dict,
    student_smoothed: np.ndarray,
    pro_smoothed: np.ndarray,
) -> dict:
    """
    For each swing phase, use DTW to find the pro frame that best matches each
    student phase midpoint. Falls back to midpoint matching if DTW fails.

    Returns:
        dict: {phase_name: (student_frame, aligned_pro_frame, dtw_distance)}
    """
    result = {}

    for phase in student_phases:
        s_start, s_end = student_phases[phase]
        p_start, p_end = pro_phases.get(phase, (None, None))

        # Student midpoint
        s_mid = (s_start + s_end) // 2

        if p_start is None or p_end is None:
            # Phase doesn't exist in pro — use simple midpoint
            p_mid = 0
            result[phase] = (s_mid, p_mid, None)
            continue

        # Extract phase sub-trajectories
        s_seg = student_smoothed[s_start: s_end + 1]
        p_seg = pro_smoothed[p_start: p_end + 1]

        if len(s_seg) < 2 or len(p_seg) < 2:
            p_mid = (p_start + p_end) // 2
            result[phase] = (s_mid, p_mid, None)
            continue

        try:
            dist, path = compute_dtw(s_seg, p_seg)

            # Find student midpoint's local index in the segment
            s_local = s_mid - s_start
            s_local = max(0, min(s_local, len(s_seg) - 1))

            # Find best matching pro local index from path
            # Take the pro index that most often pairs with s_local
            p_local_candidates = [j for (i, j) in path if i == s_local]
            if p_local_candidates:
                p_local = int(np.median(p_local_candidates))
            else:
                p_local = (p_end - p_start) // 2

            p_mid = p_start + p_local
            result[phase] = (s_mid, p_mid, dist)

        except Exception:
            p_mid = (p_start + p_end) // 2
            result[phase] = (s_mid, p_mid, None)

    return result


def compute_similarity_score(
    student_smoothed: np.ndarray,
    pro_smoothed: np.ndarray,
    swing_start_s: int,
    swing_end_s: int,
    swing_start_p: int,
    swing_end_p: int,
) -> float:
    """
    Compute a 0–100 similarity score between student and pro swing trajectories.
    Uses normalized DTW distance over the swing window.
    """
    s_seg = student_smoothed[swing_start_s: swing_end_s + 1]
    p_seg = pro_smoothed[swing_start_p: swing_end_p + 1]

    if len(s_seg) < 2 or len(p_seg) < 2:
        return 0.0

    # Normalize both to 0-1 range
    def normalize(arr):
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if mx == mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    s_norm = normalize(s_seg)
    p_norm = normalize(p_seg)

    try:
        dist, _ = compute_dtw(s_norm, p_norm)
        # Scale: max possible distance is len of longer sequence
        max_dist = max(len(s_norm), len(p_norm))
        score = max(0.0, 100.0 * (1.0 - dist / max_dist))
        return round(score, 1)
    except Exception:
        return 0.0
