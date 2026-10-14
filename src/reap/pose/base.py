"""Backend protocol — every pose estimator returns PoseResult (COCO-17)."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from reap.schemas import PoseResult


class PoseBackend(Protocol):
    name: str

    def process(self, bgr: np.ndarray, frame_id: int = 0) -> PoseResult: ...
    def close(self) -> None: ...
