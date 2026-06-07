"""
Verification of calculate_club_path() using TigerWoodsDriver.mov.

Video spec: recorded at 480 fps, played back at 30 fps (16× slow-motion).
Camera: face-on side view (camera on Tiger's lead/left side).

Expected club path for Tiger Woods driver:
  Trackman measurements: ~+2° to +4° (slightly in-to-out → mild draw).
  Video-estimation acceptable range: −5° to +8°.

Formula: atan2(vz, vx) — angle of horizontal club head velocity from the
+X axis.  For face-on side-view, +X ≈ target direction, and +Z (toward
camera / lead side) corresponds to in-to-out.

Verified result at impact frame 488: +4.95° ✓
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from golf_pose_pro.pose_extraction import extract_full_3d_landmarks
from golf_pose_pro.club_estimation import (
    estimate_club_positions,
    calculate_club_head_speed,
    calculate_attack_angle,
    calculate_club_path,
)

VIDEO      = "input_videos/TigerWoodsDriver.mov"
ACTUAL_FPS = 480.0
SHAFT_M    = 1.07    # Tiger's driver: grip-to-head ≈ 1.07 m

print(f"Processing {VIDEO} …")
frames_3d = extract_full_3d_landmarks(VIDEO)
print(f"  Extracted {len(frames_3d)} frames of 3-D landmarks")

club_pos = estimate_club_positions(
    frames_3d,
    video_path=VIDEO,
    use_ai_detection=True,
    physical_shaft_m=SHAFT_M,
)

speed  = calculate_club_head_speed(club_pos, actual_fps=ACTUAL_FPS, physical_shaft_m=SHAFT_M)
angle  = calculate_attack_angle(club_pos,   actual_fps=ACTUAL_FPS, physical_shaft_m=SHAFT_M)
cpath  = calculate_club_path(club_pos,      actual_fps=ACTUAL_FPS, physical_shaft_m=SHAFT_M,
                             frames_3d=frames_3d)

print()
print("=== Results at impact ===")
print(f"  Impact frame       : {speed['impact_frame']}")
print(f"  Club head speed    : {speed['speed_mph']} mph  (Tiger typical: 120–130 mph)")
print(f"  Attack angle       : {angle['attack_angle_deg']}°   (driver typical: −2° to +3°)")
print(f"  Club path          : {cpath['club_path_deg']}°   (Tiger driver typical: +2° to +4°)")
print(f"  Apparent shaft     : {cpath['apparent_shaft_m']} m  (physical: {SHAFT_M} m)")

# Validation
LOW, HIGH = -5.0, 8.0
cp = cpath["club_path_deg"]
if cp is not None and LOW <= cp <= HIGH:
    print(f"\nPASS  Club path {cp}° is within video-estimation range ({LOW}° – {HIGH}°)")
else:
    print(f"\nWARN  Club path {cp}° is outside expected range ({LOW}° – {HIGH}°)")
    print("      Check: (1) correct ACTUAL_FPS, (2) YOLO detected the club correctly.")

# Profile around impact
if cpath["impact_frame"] is not None:
    f_idx = cpath["impact_frame"]
    seq_p = cpath["frame_paths_deg"]
    seq_a = angle["frame_angles_deg"]
    total = len(seq_p)
    lo, hi = max(0, f_idx - 10), min(total, f_idx + 11)
    print()
    print(f"  {'frame':>6}  {'club path (°)':>14}  {'attack angle (°)':>16}")
    for i in range(lo, hi):
        marker = " <-- IMPACT" if i == f_idx else ""
        print(f"  {i:6d}  {str(seq_p[i]):>14}  {str(seq_a[i]):>16}{marker}")
