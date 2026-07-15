"""
Verification of calculate_swing_plane() using TigerWoodsDriver.mov.

Video spec: recorded at 480 fps, played back at 30 fps (16× slow-motion).

Swing Plane (TrackMan definition): the vertical angle, relative to the horizon,
of the plane in which the club head travels.

A radar fits a 3D plane to the club head path; that needs accurate depth.
From a single face-on camera the depth axis is unreliable, so we instead use
the projection geometry: the tilted swing circle projects to an ellipse in the
image (X-Y) plane whose minor/major axis ratio equals sin(plane tilt). This
recovers the plane angle from only the well-measured image-plane coordinates.

Typical values (TrackMan, driver):
  Male scratch:      48.1°
  Male 14.5 HCP:     49.0°
  Tour / elite:      ~45° to ~52°  (a driver sits the flattest of all clubs)

Monocular ellipse estimate carries a few degrees of bias; acceptable
video-estimation range used here: 45° - 62°.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from golf_pose_pro.pose_extraction import extract_full_3d_landmarks
from golf_pose_pro.club_estimation import (
    estimate_club_positions,
    calculate_club_head_speed,
    calculate_swing_plane,
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

speed = calculate_club_head_speed(club_pos, actual_fps=ACTUAL_FPS, physical_shaft_m=SHAFT_M)
plane = calculate_swing_plane(club_pos, actual_fps=ACTUAL_FPS, physical_shaft_m=SHAFT_M)

print()
print("=== Swing Plane ===")
print(f"  Impact frame       : {plane['impact_frame']}")
print(f"  Club head speed    : {speed['speed_mph']} mph  (Tiger typical: 120–130 mph)")
print(f"  Swing plane        : {plane['swing_plane_deg']}°   (driver typical: ~45°–52°)")
print(f"  Points in fit      : {plane['n_points']}  (club head positions over swing)")
print(f"  Apparent shaft     : {plane['apparent_shaft_m']} m  (physical: {SHAFT_M} m)")

# Validation
LOW, HIGH = 45.0, 62.0
sp = plane["swing_plane_deg"]
if sp is not None and LOW <= sp <= HIGH:
    print(f"\nPASS  Swing plane {sp}° is within video-estimation range ({LOW}° – {HIGH}°)")
    print("      TrackMan male-scratch driver reference: 48.1°.")
else:
    print(f"\nWARN  Swing plane {sp}° is outside expected range ({LOW}° – {HIGH}°)")
    print("      Check: (1) correct ACTUAL_FPS, (2) YOLO detected the club correctly.")
