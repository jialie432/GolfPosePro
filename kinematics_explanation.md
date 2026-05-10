# `kinematics.py` — Detailed Code Explanation

> **File:** `golf_pose_pro/kinematics.py`  
> **Purpose:** Velocity and joint-angle computation from MediaPipe pose landmarks.

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Imports](#2-imports)
3. [Landmark Index Constants](#3-landmark-index-constants)
4. [Joint Angle Definitions — `JOINT_ANGLE_DEFS`](#4-joint-angle-definitions--joint_angle_defs)
5. [Geometry Helpers](#5-geometry-helpers)
   - [`_angle_between_vectors()`](#_angle_between_vectors)
   - [`compute_joint_angle()`](#compute_joint_angle)
   - [`_get_lm_pos()`](#_get_lm_pos)
6. [Joint Angles from 3D Landmarks — `compute_all_joint_angles()`](#6-joint-angles-from-3d-landmarks--compute_all_joint_angles)
7. [Velocity Computation](#7-velocity-computation)
   - [`compute_landmark_velocity()`](#compute_landmark_velocity)
   - [`compute_angular_velocity()`](#compute_angular_velocity)
8. [2D Velocity from Frame Data — `compute_velocities_from_frame_data()`](#8-2d-velocity-from-frame-data--compute_velocities_from_frame_data)
9. [Peak Detection](#9-peak-detection)
   - [`find_peak_velocities()`](#find_peak_velocities)
   - [`find_peak_angular_velocities()`](#find_peak_angular_velocities)
10. [Data Flow Summary](#10-data-flow-summary)

---

## 1. Module Overview

`kinematics.py` is a **pure-computation, stateless module** — it has no class definitions and no side effects. It accepts NumPy arrays or lists of landmark dicts (produced by `pose_extraction.py`) and returns numerical results ready for downstream visualisation or feedback.

The four main things it computes are:

| Metric | Output units | Key function |
|---|---|---|
| Linear speed of body landmarks | px/s (2D) or m/s (3D) | `compute_landmark_velocity()` |
| Joint angles (elbow, knee, shoulder, spine) | degrees | `compute_all_joint_angles()` |
| Angular velocity of joint angles | °/s | `compute_angular_velocity()` |
| Peak speed per swing phase | px/s or °/s | `find_peak_velocities()` / `find_peak_angular_velocities()` |

---

## 2. Imports

```python
import numpy as np
from typing import Optional
```

- **`numpy`** — All numerical operations (vector arithmetic, finite differences, trigonometry) are vectorised through NumPy for performance.
- **`Optional`** — Used only in type hints; `_get_lm_pos()` can return `None` when a landmark is occluded or missing.

---

## 3. Landmark Index Constants

```python
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_KNEE      = 25
RIGHT_KNEE     = 26
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28
```

These are **MediaPipe Pose's integer landmark indices** (from the official 33-landmark body model). Using named constants instead of raw integers everywhere:
- Makes code self-documenting.
- Prevents bugs from magic numbers.
- Makes it trivial to update if MediaPipe changes its model (change in one place, reflected everywhere).

---

## 4. Joint Angle Definitions — `JOINT_ANGLE_DEFS`

```python
JOINT_ANGLE_DEFS = [
    ("elbow_L",    LEFT_SHOULDER,  LEFT_ELBOW,   LEFT_WRIST),
    ("elbow_R",    RIGHT_SHOULDER, RIGHT_ELBOW,  RIGHT_WRIST),
    ("knee_L",     LEFT_HIP,       LEFT_KNEE,    LEFT_ANKLE),
    ("knee_R",     RIGHT_HIP,      RIGHT_KNEE,   RIGHT_ANKLE),
    ("shoulder_L", LEFT_HIP,       LEFT_SHOULDER, LEFT_ELBOW),
    ("shoulder_R", RIGHT_HIP,      RIGHT_SHOULDER, RIGHT_ELBOW),
]
```

Each tuple is `(name, landmark_A, landmark_B_vertex, landmark_C)`.  
**Convention:** the angle is measured **at the middle point (B)**, i.e., the angle formed by vectors `B→A` and `B→C`.

For example, `elbow_L` measures the angle at the left elbow — the opening between the upper arm (shoulder→elbow) and the forearm (elbow→wrist).

The list-of-tuples design makes it easy to:
- Add new joints (just append a tuple).
- Loop over all joints in one line inside `compute_all_joint_angles()`.

---

## 5. Geometry Helpers

### `_angle_between_vectors()`

```python
def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
```

**Purpose:** Computes the angle in degrees between two arbitrary vectors.

**Math:**
The angle θ between two vectors is given by the dot-product formula:

```
cos(θ) = (v1 · v2) / (|v1| × |v2|)
```

Steps:
1. Compute the L2 norms (`np.linalg.norm`).
2. **Guard against zero-length vectors** — returns `np.nan` if either norm is below `1e-9` (avoids division by zero for collapsed / missing landmarks).
3. Clamp the cosine to `[-1, 1]` with `np.clip` before calling `arccos` — floating-point rounding can occasionally produce values like `1.0000000002` that would cause `arccos` to return `NaN`.
4. Convert radians → degrees with `np.degrees`.

**Returns:** `float` in `[0°, 180°]`, or `NaN` on degenerate input.

---

### `compute_joint_angle()`

```python
def compute_joint_angle(a, b, c) -> float:
```

**Purpose:** High-level wrapper that computes the joint angle at vertex **B**.

Steps:
1. Returns `NaN` if any of the three coordinate arrays is `None` (landmark not detected).
2. Forms rays `ba = a - b` and `bc = c - b`.
3. Delegates to `_angle_between_vectors(ba, bc)`.

Works in both **2D** and **3D** — NumPy arithmetic is dimension-agnostic.

---

### `_get_lm_pos()`

```python
def _get_lm_pos(landmarks: list, idx: int) -> Optional[np.ndarray]:
```

**Purpose:** Looks up a specific landmark by index and returns its `(x, y, z)` as a NumPy array.

The `landmarks` list is the raw output from MediaPipe, structured as:

```python
[{"idx": 11, "x": 0.45, "y": 0.32, "z": -0.08, "visibility": 0.99}, ...]
```

It returns `None` if:
- No landmark with `idx` is found in the list, **or**
- The `x` value is `None` (landmark detected but coordinates invalid / occluded).

This `None` is propagated up to `compute_joint_angle()`, which gracefully returns `NaN` rather than crashing.

---

## 6. Joint Angles from 3D Landmarks — `compute_all_joint_angles()`

```python
def compute_all_joint_angles(frames_3d: list) -> dict:
```

**Input:** `frames_3d` — a list of per-frame dicts produced by `extract_full_3d_landmarks()`. Each dict looks like:

```python
{
  "frame_idx": 42,
  "timestamp_ms": 1400.0,
  "landmarks": [{"idx": 11, "x": ..., "y": ..., "z": ..., "visibility": ...}, ...]
}
```

**What it does:**

1. **Pre-allocates** result arrays filled with `NaN`:
   ```python
   angles = {name: np.full(n, np.nan) for name, *_ in JOINT_ANGLE_DEFS}
   angles["spine_tilt"] = np.full(n, np.nan)
   ```
   Pre-allocation is much faster than appending to Python lists.

2. **Loops over frames** and for each frame:
   - Iterates over `JOINT_ANGLE_DEFS`, looks up the three relevant landmarks with `_get_lm_pos()`, and fills `angles[name][i]`.
   - **Spine tilt** is computed separately:
     - Finds the midpoint of the two shoulders: `mid_shoulder = (ls + rs) / 2`
     - Finds the midpoint of the two hips: `mid_hip = (lh + rh) / 2`
     - Constructs the **spine vector**: `spine_vec = mid_shoulder - mid_hip`
     - Measures the angle between this vector and **vertical** `(0, -1, 0)`.

     > **Why `(0, -1, 0)` for vertical?**  
     > In MediaPipe's coordinate system, **Y increases downward**. So the "up" direction in world space is `(0, -1, 0)`. Spine tilt of 0° means perfectly upright; larger values mean the golfer is leaning forward or sideways.

**Returns:** `dict` of `{angle_name: np.ndarray}`, shape `(n_frames,)`, all angles in degrees.

---

## 7. Velocity Computation

### `compute_landmark_velocity()`

```python
def compute_landmark_velocity(
    x_series, y_series, fps, z_series=None
) -> np.ndarray:
```

**Purpose:** Given time series of coordinates, compute **instantaneous speed** (scalar) per frame.

**Math — Finite Differences:**  
`np.gradient` uses:
- **Central differences** for interior points: `dx[i] = (x[i+1] - x[i-1]) / 2`
- **Forward/backward differences** at endpoints to avoid edge-effects.

Multiplying by `fps` converts from "change per frame" → "change per second".

Speed is the **Euclidean norm** of the velocity vector:
```
speed = √(dx² + dy²)          # 2D
speed = √(dx² + dy² + dz²)    # 3D (when z_series provided)
```

**Returns:** `np.ndarray` of shape `(n_frames,)` in px/s (for 2D pixel coords) or m/s (for 3D world-space coords).

---

### `compute_angular_velocity()`

```python
def compute_angular_velocity(angle_series, fps) -> np.ndarray:
```

**Purpose:** Computes how quickly a joint angle is changing (°/s).

**NaN handling:**  
MediaPipe sometimes misses a landmark, leaving `NaN` in the angle series. Computing gradient over NaNs would propagate NaNs everywhere, ruining the output. The module uses a **forward-fill (LOCF — Last Observation Carried Forward)** strategy:

```python
idx = np.where(~nans, np.arange(len(clean)), 0)
np.maximum.accumulate(idx, out=idx)
clean = clean[idx]
```

This replaces each NaN index with the index of the last valid value, then re-indexes the array — effectively holding the last known angle constant until the next valid reading.

After cleaning: `d_angle = np.gradient(clean) * fps`, then `np.abs()` to return magnitude (direction of rotation is not used here).

**Edge case:** If the entire series is NaN (landmark never detected), returns all-zeros rather than crashing.

---

## 8. 2D Velocity from Frame Data — `compute_velocities_from_frame_data()`

```python
def compute_velocities_from_frame_data(
    frame_data: list, fps: float, landmark_groups: list = None
) -> dict:
```

**Purpose:** Batch-computes 2D pixel velocity for multiple landmark groups from `frame_data` — the flat per-frame dict format used in the Streamlit dashboard (keys like `"wrist_x"`, `"wrist_y"`, `"hip_x"`, etc.).

**Auto-detection of groups:**  
If `landmark_groups` is not provided, it scans the first frame's keys for anything ending in `_x` and strips the suffix:
```python
landmark_groups = list({k.rsplit("_", 1)[0] for k in sample
                        if k.endswith("_x") and k != "frame_idx"})
```

**NaN handling:**  
Uses the same forward-fill technique as `compute_angular_velocity()` — if a coordinate is missing for a frame, it holds the last known position, resulting in zero velocity for that gap rather than propagating NaN.

**Returns:** `dict` of `{group_name + "_velocity": np.ndarray}` in px/s.

---

## 9. Peak Detection

### `find_peak_velocities()`

```python
def find_peak_velocities(velocity_array, phase_ranges) -> dict:
```

**Purpose:** Given a speed array and a mapping of swing phases to frame ranges, find the **maximum speed and its frame index** within each phase.

`phase_ranges` has the shape:
```python
{
  "backswing":        (0, 45),
  "downswing":        (46, 92),
  "impact":           (93, 100),
  "follow_through":   (101, 160),
}
```

For each phase, it slices the velocity array, calls `np.nanargmax` (ignores NaN), and records:
- `peak_speed` — the maximum value found.
- `peak_frame` — the **global** frame index (start + local argmax).

**Returns:**
```python
{
  "backswing":      {"peak_speed": 342.1, "peak_frame": 38},
  "downswing":      {"peak_speed": 891.4, "peak_frame": 79},
  ...
}
```

---

### `find_peak_angular_velocities()`

```python
def find_peak_angular_velocities(angular_velocities, phase_ranges) -> dict:
```

**Purpose:** A thin wrapper around `find_peak_velocities()` that loops over multiple joints.

`angular_velocities` dict has keys like `"elbow_L_angular_vel"`. The function strips the `"_angular_vel"` suffix to use cleaner joint names as keys in the result, then delegates to `find_peak_velocities()` for the actual peak-finding logic.

**Returns:**
```python
{
  "elbow_L": {"backswing": {"peak_speed": 120.3, "peak_frame": 30}, ...},
  "knee_R":  {"backswing": ..., "downswing": ..., ...},
  ...
}
```

---

## 10. Data Flow Summary

```
pose_extraction.py
    │
    ├─ extract_full_3d_landmarks() ──► frames_3d (list of dicts)
    │                                        │
    │                                        ▼
    │                           compute_all_joint_angles()
    │                                        │
    │                                        ▼
    │                           {angle_name: np.ndarray (degrees)}
    │                                        │
    │                                        ▼
    │                           compute_angular_velocity()  ──► °/s arrays
    │                                        │
    │                                        ▼
    │                           find_peak_angular_velocities() ──► peaks per phase
    │
    └─ extract_frame_data()  ──► frame_data (flat per-frame dicts)
                                         │
                                         ▼
                           compute_velocities_from_frame_data()
                                         │
                                         ▼
                            {group_velocity: np.ndarray (px/s)}
                                         │
                                         ▼
                           find_peak_velocities() ──► peaks per phase
```

All outputs flow into the Streamlit dashboard (`app.py`) where they are plotted on the **Kinematics** tab.
