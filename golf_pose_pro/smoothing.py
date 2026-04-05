"""
smoothing.py — Adaptive smoothing filters for MediaPipe pose landmarks.

Implements the One-Euro Filter (Casiez et al., 2012), an adaptive low-pass
filter designed for real-time noisy human motion signals. It automatically
adjusts its cutoff frequency based on signal speed:
  - Slow motion → heavy smoothing (removes jitter)
  - Fast motion → light smoothing (preserves responsiveness)

Reference:
  "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in
   Interactive Systems" — Géry Casiez, Nicolas Roussel, Daniel Vogel (CHI 2012)
"""

import math
import numpy as np
from scipy.ndimage import uniform_filter1d


# ─── One-Euro Filter ────────────────────────────────────────────────────────

class _LowPassFilter:
    """Simple first-order exponential low-pass filter."""

    def __init__(self, alpha: float = 1.0):
        self._alpha = alpha
        self._y = None
        self._initialized = False

    def __call__(self, value: float, alpha: float = None) -> float:
        if alpha is not None:
            self._alpha = alpha
        if not self._initialized:
            self._y = value
            self._initialized = True
        else:
            self._y = self._alpha * value + (1.0 - self._alpha) * self._y
        return self._y

    @property
    def last(self) -> float:
        return self._y if self._y is not None else 0.0

    def reset(self):
        self._y = None
        self._initialized = False


class OneEuroFilter:
    """
    One-Euro Filter — adaptive low-pass filter for noisy signals.

    Parameters
    ----------
    freq : float
        Signal sampling frequency in Hz (e.g., video FPS).
    min_cutoff : float
        Minimum cutoff frequency in Hz. Lower = more smoothing for slow
        signals. Typical range: 0.5–3.0. Default: 1.0.
    beta : float
        Speed coefficient — controls how much the cutoff adapts to speed.
        Higher = less lag during fast motion but more jitter. Default: 0.007.
    d_cutoff : float
        Cutoff frequency for the derivative filter. Default: 1.0.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_filter = _LowPassFilter()
        self._dx_filter = _LowPassFilter()
        self._last_time = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        """Compute exponential smoothing factor from cutoff and sampling freq."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, timestamp: float = None) -> float:
        """
        Filter one sample.

        Parameters
        ----------
        x : float
            Current sample value.
        timestamp : float, optional
            Timestamp in seconds. If None, uses 1/freq increments.
        """
        if self._last_time is not None and timestamp is not None:
            dt = timestamp - self._last_time
            self.freq = 1.0 / max(dt, 1e-9)
        self._last_time = timestamp

        # Derivative (speed) of the signal
        prev = self._x_filter.last
        if prev is None or not self._x_filter._initialized:
            dx = 0.0
        else:
            dx = (x - prev) * self.freq

        # Smooth the derivative
        edx = self._dx_filter(dx, alpha=self._alpha(self.d_cutoff, self.freq))

        # Adaptive cutoff — higher speed → higher cutoff → less smoothing
        cutoff = self.min_cutoff + self.beta * abs(edx)

        # Filter the signal
        return self._x_filter(x, alpha=self._alpha(cutoff, self.freq))

    def reset(self):
        """Reset internal state (call between separate sequences)."""
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None


# ─── High-level smoothing functions ──────────────────────────────────────────

def smooth_series(
    data: np.ndarray,
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
    d_cutoff: float = 1.0,
) -> np.ndarray:
    """
    Apply the One-Euro Filter to a 1D time-series.

    Parameters
    ----------
    data : np.ndarray
        1D array of scalar values (e.g., an angle over time).
    fps : float
        Sampling frequency (video FPS).
    min_cutoff, beta, d_cutoff : float
        One-Euro filter parameters.

    Returns
    -------
    np.ndarray
        Smoothed 1D array, same length as input.
    """
    if len(data) == 0:
        return data.copy()

    filt = OneEuroFilter(freq=fps, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    out = np.empty_like(data, dtype=float)

    for i, val in enumerate(data):
        if np.isnan(val):
            out[i] = val
        else:
            out[i] = filt(float(val), timestamp=i / fps)

    return out


def smooth_landmarks(
    frame_data: list,
    landmark_keys: list = None,
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
    d_cutoff: float = 1.0,
) -> list:
    """
    Apply One-Euro smoothing to all landmark coordinate series in frame_data.

    Parameters
    ----------
    frame_data : list[dict]
        Per-frame dicts as returned by extract_pose_features().
        Each dict has keys like 'wrist_x', 'wrist_y', 'hip_x', 'hip_y', etc.
    landmark_keys : list[str], optional
        Which keys to smooth. If None, auto-discovers all keys ending in
        '_x' or '_y' (excluding 'frame_idx').
    fps : float
        Sampling frequency.
    min_cutoff, beta, d_cutoff : float
        One-Euro filter parameters.

    Returns
    -------
    list[dict]
        New frame_data with smoothed values (original frame_idx preserved).
    """
    if not frame_data:
        return frame_data

    # Discover coordinate keys
    if landmark_keys is None:
        sample = frame_data[0]
        landmark_keys = [k for k in sample if k != "frame_idx" and
                         (k.endswith("_x") or k.endswith("_y"))]

    if not landmark_keys:
        return frame_data

    n = len(frame_data)

    # Extract raw arrays, forward-fill NaNs, then filter
    smoothed_arrays = {}
    for key in landmark_keys:
        raw = np.array(
            [f.get(key) if f.get(key) is not None else np.nan for f in frame_data],
            dtype=float,
        )

        # Forward-fill NaN gaps
        nans = np.isnan(raw)
        if nans.any() and not nans.all():
            idx = np.where(~nans, np.arange(n), 0)
            np.maximum.accumulate(idx, out=idx)
            raw = raw[idx]

        # Apply One-Euro filter
        smoothed_arrays[key] = smooth_series(
            raw, fps=fps, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff
        )

    # Rebuild frame_data with smoothed values
    result = []
    for i, frame in enumerate(frame_data):
        new_frame = {"frame_idx": frame["frame_idx"]}
        for key in landmark_keys:
            new_frame[key] = float(smoothed_arrays[key][i])
        # Preserve any extra keys not in landmark_keys
        for k, v in frame.items():
            if k not in new_frame:
                new_frame[k] = v
        result.append(new_frame)

    return result


def smooth_series_uniform(
    data: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """
    Fallback: simple moving-average smoothing using uniform_filter1d.

    Provided for backward compatibility with existing smoothing_window param.
    """
    return uniform_filter1d(data.astype(float), size=window, mode="nearest")
