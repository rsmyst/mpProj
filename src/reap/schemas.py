"""Unified pose output in COCO-17 order (normalized xy in [0,1])."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton edges for drawing / angle chaining
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass
class PoseResult:
    kpts: np.ndarray  # (17, 2) normalized xy
    conf: np.ndarray  # (17,) in [0, 1]
    source: str = "unknown"  # mp | yolo | fused | none
    latency_ms: float = 0.0
    frame_id: int = 0

    @property
    def present(self) -> bool:
        return bool((self.conf > 0.05).any())

    @property
    def avg_conf(self) -> float:
        m = self.conf > 0.01
        return float(self.conf[m].mean()) if m.any() else 0.0

    def pixel(self, w: int, h: int) -> np.ndarray:
        pts = self.kpts.copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        return pts.astype(int)


def empty_pose(frame_id: int = 0, source: str = "none") -> PoseResult:
    return PoseResult(
        kpts=np.zeros((17, 2), dtype=np.float32),
        conf=np.zeros((17,), dtype=np.float32),
        source=source,
        frame_id=frame_id,
    )
