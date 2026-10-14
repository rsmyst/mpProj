"""Joint-angle utilities — feeds deviation analytics (SRS functional #2)."""
from __future__ import annotations

import numpy as np


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at joint b formed by a-b-c. Returns nan if degenerate."""
    ba = a - b
    bc = c - b
    na, nc = float(np.linalg.norm(ba)), float(np.linalg.norm(bc))
    if na < 1e-9 or nc < 1e-9:
        return float("nan")
    cosv = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def pose_angles(kpts: np.ndarray) -> dict[str, float]:
    """kpts: (17,2) COCO order. Returns key clinical angles."""
    g = lambda i: kpts[i]
    return {
        "left_knee": angle_at(g(11), g(13), g(15)),
        "right_knee": angle_at(g(12), g(14), g(16)),
        "left_elbow": angle_at(g(5), g(7), g(9)),
        "right_elbow": angle_at(g(6), g(8), g(10)),
        "left_hip": angle_at(g(5), g(11), g(13)),
        "right_hip": angle_at(g(6), g(12), g(14)),
    }
