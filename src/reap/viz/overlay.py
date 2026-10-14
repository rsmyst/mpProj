"""Skeleton overlay + HUD for the benchmark visualizer."""
from __future__ import annotations

import cv2
import numpy as np

from reap.kinematics.angles import pose_angles
from reap.schemas import COCO_EDGES, PoseResult


def draw_pose(frame: np.ndarray, pose: PoseResult,
              show_angles: bool = True) -> np.ndarray:
    h, w = frame.shape[:2]
    pts = pose.pixel(w, h)
    for a, b in COCO_EDGES:
        if pose.conf[a] > 0.3 and pose.conf[b] > 0.3:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 2)
    for i, (x, y) in enumerate(pts):
        if pose.conf[i] > 0.3:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
    y = 24
    for line in (f"{pose.source} conf={pose.avg_conf:.2f} {pose.latency_ms:.1f}ms",):
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        y += 24
    if show_angles and pose.present:
        angs = pose_angles(pose.kpts * np.array([w, h]))
        txt = f"knee L/R {angs['left_knee']:.0f}/{angs['right_knee']:.0f} " \
              f"hip L/R {angs['left_hip']:.0f}/{angs['right_hip']:.0f}"
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)
    return frame
