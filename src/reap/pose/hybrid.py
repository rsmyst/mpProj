"""Hybrid fusion: MediaPipe every frame + YOLO fallback / periodic validator.

Policy (Sept milestone):
- mode mp    -> MediaPipe only
- mode yolo  -> YOLO only
- mode hybrid-> MP primary; run YOLO when MP weak/missing or every N frames,
                 fuse per-joint by confidence weighting.
"""
from __future__ import annotations

import numpy as np

from reap.pose.mediapipe_backend import MediaPipeBackend
from reap.pose.yolo_backend import YoloBackend
from reap.schemas import PoseResult, empty_pose


def fuse_confidence_weighted(a: PoseResult, b: PoseResult) -> PoseResult:
    wa = a.conf[:, None]
    wb = b.conf[:, None]
    tot = wa + wb
    tot = np.where(tot < 1e-6, 1.0, tot)
    kpts = (a.kpts * wa + b.kpts * wb) / tot
    # If only one backend sees a joint, take that one exactly
    only_a = (a.conf > 0.05) & (b.conf <= 0.05)
    only_b = (b.conf > 0.05) & (a.conf <= 0.05)
    kpts[only_a] = a.kpts[only_a]
    kpts[only_b] = b.kpts[only_b]
    conf = np.maximum(a.conf, b.conf)
    lat = max(a.latency_ms, b.latency_ms)
    return PoseResult(kpts=kpts.astype(np.float32), conf=conf.astype(np.float32),
                      source="fused", latency_ms=lat, frame_id=a.frame_id)


class HybridPose:
    def __init__(self, mode: str = "hybrid", mp_conf_threshold: float = 0.5,
                 validate_every_n: int = 15, mp_kwargs: dict | None = None,
                 yolo_kwargs: dict | None = None) -> None:
        assert mode in ("mp", "yolo", "hybrid")
        self.mode = mode
        self.mp_conf_threshold = mp_conf_threshold
        self.validate_every_n = max(1, validate_every_n)
        self._mp: MediaPipeBackend | None = None
        self._yolo: YoloBackend | None = None
        self._mp_kwargs = mp_kwargs or {}
        self._yolo_kwargs = yolo_kwargs or {}
        self._last_yolo: PoseResult | None = None

    def _mp_backend(self) -> MediaPipeBackend:
        if self._mp is None:
            self._mp = MediaPipeBackend(**self._mp_kwargs)
        return self._mp

    def _yolo_backend(self) -> YoloBackend:
        if self._yolo is None:
            self._yolo = YoloBackend(**self._yolo_kwargs)
        return self._yolo

    def process(self, bgr: np.ndarray, frame_id: int = 0) -> PoseResult:
        if self.mode == "mp":
            return self._mp_backend().process(bgr, frame_id)
        if self.mode == "yolo":
            return self._yolo_backend().process(bgr, frame_id)
        # hybrid
        mp_res = self._mp_backend().process(bgr, frame_id)
        need_yolo = (
            (not mp_res.present)
            or (mp_res.avg_conf < self.mp_conf_threshold)
            or (frame_id % self.validate_every_n == 0)
        )
        if not need_yolo:
            return mp_res
        yolo_res = self._yolo_backend().process(bgr, frame_id)
        self._last_yolo = yolo_res
        if not mp_res.present:
            return yolo_res
        if not yolo_res.present:
            return mp_res
        fused = fuse_confidence_weighted(mp_res, yolo_res)
        fused.latency_ms = mp_res.latency_ms + yolo_res.latency_ms
        return fused

    def close(self) -> None:
        if self._mp is not None:
            self._mp.close()
        if self._yolo is not None:
            self._yolo.close()
