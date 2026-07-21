cd /path/to/GolfPosePro
source .venv/bin/activate
streamlit run app.py

Then open 
http://localhost:8501 in your browser.


1. _fix_left_right_swaps() — corrects a transient BlazePose bug where it mislabels left/right body-side landmarks for a few frames during fast rotation (follow-through). Compares each frame's left/right landmark pairs against a slow-moving reference to detect and undo the swap.

2. smooth_landmarks_3d() — applies the existing One-Euro filter to the 3D world landmarks over time, removing per-frame depth-estimation jitter that was making legs (and other joints) wobble even when the body was actually still.

3 Root cause: the forearm's rotation was computed using the elbow landmark position as the pivot (lm13), while the stretch magnitude was measured from the forearm bone's actual world position — and those two differ slightly because the model's fixed upper-arm bone length doesn't exactly match the mocap subject's proportions. For a target close to the arm's natural reach, that small positional mismatch translates into a large angular error, so the arm rotated toward the wrong direction and the stretched hand ended up nowhere near the grip (up near the chin, in your screenshots).

Fix: the forearm's rotation is now computed using its own actual current world position as the pivot (driveForeArmToGrip), matching what the stretch calculation already used — so rotation direction and stretch distance now agree.

reproduced exact bug by running the real pipeline (MediaPipe + YOLO club detection) on one of the project's sample videos, confirmed the club/wrist data itself was correct (grip is defined as the wrist midpoint, so it's always exactly right), isolated the bug to this rotation/stretch inconsistency, and re-verified across the address pose, backswing, and follow-through frames — hands now stay attached to the arms and gripped on the shaft throughout.

