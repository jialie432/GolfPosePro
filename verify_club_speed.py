"""
Quick verification of calculate_club_head_speed() using TigerWoodsDriver.mov.

Video spec: recorded at 480 fps, played back at 30 fps (16× slow-motion).
Expected club head speed for Tiger Woods driver: ~120–135 mph.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from golf_pose_pro.pose_extraction import extract_full_3d_landmarks
from golf_pose_pro.club_estimation import estimate_club_positions, calculate_club_head_speed

VIDEO = "input_videos/TigerWoodsDriver.mov"
ACTUAL_FPS = 480.0   # true capture rate (slow-motion)

print(f"Processing {VIDEO} …")
frames_3d = extract_full_3d_landmarks(VIDEO)
print(f"  Extracted {len(frames_3d)} frames of 3-D landmarks")

club_pos = estimate_club_positions(
    frames_3d,
    phase_ranges=None,
    video_path=VIDEO,
    use_ai_detection=True,
)
print(f"  Club positions estimated for {sum(1 for c in club_pos if c['has_club'])} frames")

# Tiger Woods' driver shaft: standard 45 in = 1.143 m; he used ~44.5 in ≈ 1.13 m.
# Grip is held at the very end, so effective grip-to-head ≈ 1.07 m.
DRIVER_SHAFT_M = 1.07

result = calculate_club_head_speed(club_pos, actual_fps=ACTUAL_FPS,
                                   physical_shaft_m=DRIVER_SHAFT_M)

print()
print("=== Club Head Speed at Impact ===")
print(f"  Impact frame       : {result['impact_frame']}")
print(f"  Speed              : {result['speed_mph']} mph  ({result['speed_ms']} m/s)")
print(f"  Apparent shaft len : {result['apparent_shaft_m']} m  (physical: {DRIVER_SHAFT_M} m)")
print(f"  Scale correction   : ×{DRIVER_SHAFT_M / result['apparent_shaft_m']:.3f}")
print()

# Sanity band for a Tour driver swing
LOW, HIGH = 110.0, 145.0
if result["speed_mph"] is not None and LOW <= result["speed_mph"] <= HIGH:
    print(f"PASS  ({LOW}–{HIGH} mph expected range for Tiger Woods driver)")
else:
    print(f"WARN  Speed {result['speed_mph']} mph is outside expected {LOW}–{HIGH} mph range")

# Print a speed profile around impact (±10 frames)
if result["impact_frame"] is not None:
    f = result["impact_frame"]
    n = len(result["frame_speeds_mph"])
    lo, hi = max(0, f - 10), min(n, f + 11)
    print()
    print("Speed profile (mph) around impact:")
    for i in range(lo, hi):
        marker = " <-- IMPACT" if i == f else ""
        spd = result["frame_speeds_mph"][i]
        print(f"  frame {i:4d}: {spd}{marker}")
