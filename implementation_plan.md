# Velocity & Angle Extraction with Smoothing Filters

Extract per-frame velocity (linear & angular) and joint angle data from MediaPipe landmarks, using smoothing filters to reduce landmark jitter.

## Background

Currently, GolfPosePro extracts only Y-coordinate trajectories from selected landmarks (wrist, hip, shoulder) and computes a simple velocity magnitude via `np.gradient(smoothed)` inside `phase_detection.py` for phase detection only. There is no dedicated velocity or angle data available for analysis, export, or display. The only smoothing is a `uniform_filter1d` moving average on the Y signal.

## User Review Required

> [!IMPORTANT]
> **Smoothing filter choice**: I propose using a **One-Euro Filter** — a well-known adaptive low-pass filter designed specifically for real-time noisy human motion signals. It adapts its cutoff frequency based on signal speed: slow motion gets heavy smoothing (removes jitter), fast motion gets light smoothing (preserves responsiveness). This is the gold standard for MediaPipe/pose-tracking pipelines. Alternative: a Butterworth low-pass filter (fixed cutoff). Please confirm this approach.

> [!IMPORTANT]
> **Angle definitions**: I plan to compute these specific joint angles per frame:
> - **Left/Right Elbow angle** — shoulder→elbow→wrist
> - **Left/Right Knee angle** — hip→knee→ankle
> - **Left/Right Shoulder angle** — hip→shoulder→elbow
> - **Spine tilt angle** — vertical vs. mid-shoulder→mid-hip line
>
> Are there specific angles you'd like added or removed?

## Proposed Changes

### New Module: Smoothing Filter

#### [NEW] [smoothing.py](file:///Users/jialielu/highschoolproj/GolfPosePro/golf_pose_pro/smoothing.py)

A dedicated smoothing module providing:

1. **One-Euro Filter class** — Adaptive low-pass filter with configurable `min_cutoff`, `beta`, and `d_cutoff` parameters. Processes scalar signals frame-by-frame.
2. **`smooth_landmarks()` function** — Takes raw per-frame landmark arrays and returns smoothed versions. Applies the One-Euro Filter independently to each coordinate (x, y, z) of each landmark.
3. **`smooth_series()` function** — Convenience wrapper for 1D time-series (e.g., an angle trajectory), applying the same One-Euro filter.
4. Falls back to `scipy.ndimage.uniform_filter1d` if the user disables adaptive smoothing.

---

### New Module: Kinematics Computation

#### [NEW] [kinematics.py](file:///Users/jialielu/highschoolproj/GolfPosePro/golf_pose_pro/kinematics.py)

Core computation module for velocity and angle extraction:

**Velocity computation:**
- `compute_landmark_velocity(positions, fps)` — From smoothed (x, y) positions per frame, compute instantaneous velocity (px/s or m/s for 3D) using central finite differences.
- `compute_angular_velocity(angles, fps)` — From smoothed angle series, compute angular velocity (°/s).
- Returns per-frame arrays that align with `frame_data`.

**Angle computation:**
- `compute_joint_angle(a, b, c)` — 3-point angle: vector BA · BC → angle in degrees.
- `compute_all_joint_angles(frame_3d_landmarks)` — Iterate through frames, compute elbow, knee, shoulder, and spine angles per frame.
- Returns a dict of angle-name → np.ndarray time series.

**Peak detection:**
- `find_peak_velocities(velocity_array, phase_ranges)` — Identify peak speeds per phase (e.g., peak club head speed at impact).

---

### Modify Pose Extraction

#### [MODIFY] [pose_extraction.py](file:///Users/jialielu/highschoolproj/GolfPosePro/golf_pose_pro/pose_extraction.py)

- Extend `extract_pose_features()` to also store **x-coordinates** for tracked landmarks (currently only stores `_y`). This is needed for 2D velocity computation (velocity = √(Δx² + Δy²)).
  - Add `{group}_x` fields alongside existing `{group}_y` fields in each frame dict.
- Apply the One-Euro filter to raw landmark positions before storing them, controlled by a new `apply_smoothing: bool = True` parameter.
- Store `fps` in the returned metadata so downstream consumers know the time scale.

---

### Modify Phase Detection

#### [MODIFY] [phase_detection.py](file:///Users/jialielu/highschoolproj/GolfPosePro/golf_pose_pro/phase_detection.py)

- Replace `uniform_filter1d` with the new `smooth_series()` function from `smoothing.py` when adaptive smoothing is enabled (keep `uniform_filter1d` as fallback for backward compatibility).
- No changes to phase detection logic itself.

---

### Modify Export Utilities

#### [MODIFY] [export_utils.py](file:///Users/jialielu/highschoolproj/GolfPosePro/golf_pose_pro/export_utils.py)

- `export_to_csv()`: Add columns for velocity and angle data when available (e.g., `wrist_velocity`, `elbow_angle_L`, `elbow_angle_R`, `angular_velocity_elbow_L`, etc.).
- `export_to_json()`: Add a `kinematics` section to the JSON output containing per-frame velocity and angle arrays, plus peak velocity summaries per phase.

---

### Modify Streamlit App

#### [MODIFY] [app.py](file:///Users/jialielu/highschoolproj/GolfPosePro/app.py)

**Sidebar additions:**
- Add a checkbox: "Use adaptive smoothing (One-Euro Filter)" — default on.
- Add sliders for One-Euro parameters (`min_cutoff`, `beta`) with sensible defaults and tooltips.

**New analysis pipeline steps (between Steps 3 and 4):**
- Step 3b: Apply One-Euro smoothing to all extracted landmark data.
- Step 3c: Compute joint angles from 3D landmarks.
- Step 3d: Compute linear velocities and angular velocities.
- Step 3e: Find peak velocities per phase.

**New UI tab — "📐 Kinematics":**
- **Velocity chart**: Plot wrist velocity over time with phase overlays, highlight peak speed at impact.
- **Angle chart**: Plot selected joint angles over time (user can toggle which angles to show).
- **Peak stats cards**: Show peak wrist speed, club head speed estimate, max elbow angle change, etc.
- **Angular velocity chart**: Plot angular velocity for key joints.

**Store new data in session state** for display and export.

---

## Open Questions

1. **Units**: For 2D velocity, should we display in **pixels/second** (from normalized MediaPipe coords × image height), or convert to an estimated real-world unit? The 3D world landmarks are in meters, so 3D velocity would naturally be in m/s.

2. **Which velocity to highlight**: Should the primary "swing speed" metric be based on the **wrist velocity** (most reliable from MediaPipe), or should we attempt a **club head speed estimate** by extrapolating from wrist + shaft direction? The latter is less accurate but more golfer-meaningful.

3. **Smoothing defaults**: The One-Euro filter has two key parameters:
   - `min_cutoff` (low = more smoothing, default ~1.0)
   - `beta` (high = faster adaptation to speed changes, default ~0.007)
   
   These defaults work well for 30fps webcam pose data. Should these be user-adjustable in the sidebar, or fixed to sensible defaults?

## Verification Plan

### Automated Tests
- Run `streamlit run app.py` and process a test video to ensure no regressions.
- Verify that the new Kinematics tab renders velocity and angle charts correctly.
- Confirm exported CSV/JSON contains the new velocity and angle columns.

### Manual Verification
- Compare smoothed vs. raw landmark trajectories visually — smoothed should reduce jitter without lagging behind fast motions.
- Check that peak velocity correctly aligns with the Impact phase.
- Validate angle computations against expected biomechanical ranges (e.g., elbow angle ~90-180°, spine tilt ~0-45°).
