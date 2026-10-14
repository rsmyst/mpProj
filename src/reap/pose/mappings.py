"""MediaPipe-33 -> COCO-17 index mapping.

MediaPipe Pose indices:
0 nose, 1-6 eyes, 7 left_ear, 8 right_ear, 11 L-shoulder, 12 R-shoulder,
13 L-elbow, 14 R-elbow, 15 L-wrist, 16 R-wrist,
23 L-hip, 24 R-hip, 25 L-knee, 26 R-knee, 27 L-ankle, 28 R-ankle.
"""
from __future__ import annotations

# COCO_idx -> MediaPipe_idx
MP_TO_COCO: dict[int, int] = {
    0: 0,    # nose
    1: 2,    # left_eye  (MP left_eye)
    2: 5,    # right_eye
    3: 7,    # left_ear
    4: 8,    # right_ear
    5: 11,   # left_shoulder
    6: 12,   # right_shoulder
    7: 13,   # left_elbow
    8: 14,   # right_elbow
    9: 15,   # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}
