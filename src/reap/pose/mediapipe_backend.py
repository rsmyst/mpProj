"""MediaPipe backend — fast primary path, runs every frame on CPU.

Uses mediapipe.tasks PoseLandmarker (mediapipe>=1.0 removed solutions.pose).
Model bundle auto-downloads once to ~/.cache/reap, then runs fully offline.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np

from reap.pose.mappings import MP_TO_COCO
from reap.schemas import PoseResult, empty_pose

_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def _ensure_bundle(model_complexity: int) -> str:
    url = _MODEL_URLS.get(model_complexity, _MODEL_URLS[1])
    name = url.rsplit("/", 1)[-1]  # e.g. pose_landmarker_lite.task
    dest = Path.home() / ".cache" / "reap" / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[mediapipe] downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
    return str(dest)


class MediaPipeBackend:
    name = "mp"

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_poses: int = 1,
    ) -> None:
        from mediapipe.tasks.python.core import base_options
        from mediapipe.tasks.python import vision

        bundle = _ensure_bundle(model_complexity)
        opts = vision.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=bundle),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)
        self._t0 = time.perf_counter()

    def process(self, bgr: np.ndarray, frame_id: int = 0) -> PoseResult:
        import cv2
        import mediapipe as mp

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - self._t0) * 1000) + frame_id * 33
        res = self._landmarker.detect_for_video(image, ts_ms)
        dt = (time.perf_counter() - t0) * 1000.0
        if not res.pose_landmarks:
            out = empty_pose(frame_id, source="mp")
            out.latency_ms = dt
            return out
        lm = res.pose_landmarks[0]  # 33 NormalizedLandmark
        kpts = np.zeros((17, 2), dtype=np.float32)
        conf = np.zeros((17,), dtype=np.float32)
        for coco_i, mp_i in MP_TO_COCO.items():
            p = lm[mp_i]
            kpts[coco_i, 0] = float(np.clip(p.x, 0, 1))
            kpts[coco_i, 1] = float(np.clip(p.y, 0, 1))
            c = float(getattr(p, "visibility", float("nan")))
            if not np.isfinite(c) or c <= 0:
                c = float(getattr(p, "presence", 0.9))
            conf[coco_i] = float(np.clip(c, 0, 1))
        return PoseResult(kpts=kpts, conf=conf, source="mp",
                          latency_ms=dt, frame_id=frame_id)

    def close(self) -> None:
        self._landmarker.close()
