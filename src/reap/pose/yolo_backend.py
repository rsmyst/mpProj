"""YOLOv8n-pose backend — robust fallback/validator (COCO-17 native)."""
from __future__ import annotations

import time

import numpy as np

from reap.schemas import PoseResult, empty_pose


class YoloBackend:
    name = "yolo"

    def __init__(self, weights: str = "yolov8n-pose.pt",
                 imgsz: int = 640, conf: float = 0.5,
                 device: str = "cpu") -> None:
        from ultralytics import YOLO

        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(weights)  # auto-downloads on first use
        self.device = device

    def process(self, bgr: np.ndarray, frame_id: int = 0) -> PoseResult:
        t0 = time.perf_counter()
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf,
                               device=self.device, verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000.0
        if r.keypoints is None or len(r.keypoints) == 0:
            out = empty_pose(frame_id, source="yolo")
            out.latency_ms = dt
            return out
        # Pick largest person box (or highest conf if no boxes)
        idx = 0
        if r.boxes is not None and len(r.boxes) > 0:
            wh = r.boxes.xywh.cpu().numpy()[:, 2:4]
            idx = int((wh[:, 0] * wh[:, 1]).argmax())
        xy = r.keypoints.xyn.cpu().numpy()[idx]  # (17,2) normalized
        kp_conf = r.keypoints.conf
        cf = kp_conf.cpu().numpy()[idx] if kp_conf is not None else np.ones(17)
        n = min(17, xy.shape[0])
        kpts = np.zeros((17, 2), dtype=np.float32)
        conf = np.zeros((17,), dtype=np.float32)
        kpts[:n] = np.clip(xy[:n], 0, 1).astype(np.float32)
        conf[:n] = np.clip(cf[:n], 0, 1).astype(np.float32)
        return PoseResult(kpts=kpts, conf=conf, source="yolo",
                          latency_ms=dt, frame_id=frame_id)

    def close(self) -> None:
        pass
