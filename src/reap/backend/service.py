"""Frame in -> canonical pose + stick overlay + optional rep/cue out."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from reap.audio.cues import CueEngine
from reap.framework.session import WorkoutSession
from reap.kinematics.angles import pose_angles
from reap.pose.filters import OneEuroFilter
from reap.pose.hybrid import HybridPose
from reap.schemas import COCO_NAMES, PoseResult, empty_pose
from reap.viz.overlay import draw_pose


@dataclass
class FrameOutput:
    frame_id: int
    pose: PoseResult
    angles: dict[str, float] = field(default_factory=dict)
    session: dict | None = None
    cue: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self, fps: float = 0.0, mode: str = "",
                stick_only: bool = False) -> dict[str, Any]:
        joints = [
            {
                "name": name,
                "x": round(float(self.pose.kpts[i, 0]), 4),
                "y": round(float(self.pose.kpts[i, 1]), 4),
                "confidence": round(float(self.pose.conf[i]), 3),
            }
            for i, name in enumerate(COCO_NAMES)
        ]
        angles = {
            k: (round(float(v), 1) if math.isfinite(float(v)) else None)
            for k, v in self.angles.items()
        }
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "mode": mode,
            "stick_only": stick_only,
            "fps": round(float(fps), 1),
            "pose": {
                "present": self.pose.present,
                "source": self.pose.source,
                "avg_confidence": round(self.pose.avg_conf, 3),
                "latency_ms": round(self.pose.latency_ms, 2),
            },
            "joints": joints,
            "angles": angles,
            "session": self.session,
            "cue": self.cue,
        }


class PoseService:
    """Thread-safe reusable backend for a live video pipeline."""

    def __init__(self, cfg: dict | None = None, mode: str | None = None,
                 exercise: str | None = None, smoothing: bool = True,
                 backend: Any | None = None,
                 cue_engine: CueEngine | None = None) -> None:
        cfg = cfg or {}
        hyb_cfg = cfg.get("hybrid", {})
        self.mode = mode or hyb_cfg.get("mode", "hybrid")
        self.smoothing = bool(smoothing)
        self.cue_engine = cue_engine
        self._lock = threading.Lock()
        self._smoother = OneEuroFilter()

        if backend is not None:
            self.pipe = backend
        else:
            self.pipe = HybridPose(
                mode=self.mode,
                mp_conf_threshold=hyb_cfg.get("mp_conf_threshold", 0.5),
                validate_every_n=hyb_cfg.get("validate_every_n", 15),
                mp_kwargs=cfg.get("mediapipe", {}),
                yolo_kwargs=cfg.get("yolo", {}),
            )

        sess_cfg = cfg.get("session", {})
        exercise = exercise if exercise is not None else sess_cfg.get("exercise", "chair_stand")
        self._session_cfg = {
            "exercise": exercise,
            "min_amplitude": sess_cfg.get("min_amplitude", 20.0),
            "min_gap": sess_cfg.get("min_gap", 10),
            "stand_threshold": sess_cfg.get("stand_threshold", 150.0),
        }
        self.session: WorkoutSession | None = None
        if exercise and exercise not in ("none", "off"):
            self.session = WorkoutSession(**self._session_cfg)

        self.last_output: FrameOutput | None = None

    def set_mode(self, mode: str) -> None:
        if mode not in ("mp", "yolo", "hybrid"):
            raise ValueError(f"unknown pose mode {mode!r}")
        if not hasattr(self.pipe, "set_mode"):
            raise RuntimeError("this backend does not support live mode switching")
        with self._lock:
            self.pipe.set_mode(mode)
            self.mode = mode

    def reset_session(self) -> None:
        with self._lock:
            self.session = None
            if self._session_cfg["exercise"] not in ("none", "off"):
                self.session = WorkoutSession(**self._session_cfg)

    def process(self, frame: np.ndarray, frame_id: int = 0) -> FrameOutput:
        with self._lock:
            pose = self.pipe.process(frame, frame_id)
            if self.smoothing and pose.present:
                pose.kpts = np.asarray(
                    self._smoother(pose.kpts, time.monotonic()), dtype=np.float32
                )
            elif self.smoothing:
                self._smoother.reset()

            h, w = frame.shape[:2]
            angles = (
                pose_angles(pose.kpts * np.array([w, h], dtype=np.float32))
                if pose.present else {}
            )
            session_out = None
            cue = ""
            if self.session is not None:
                session_out = self.session.update(angles, frame_id)
                if session_out.get("rep_done"):
                    cue = str(session_out.get("feedback", ""))
                    if cue and self.cue_engine is not None:
                        self.cue_engine.say(cue)

            out = FrameOutput(frame_id=frame_id, pose=pose, angles=angles,
                              session=session_out, cue=cue)
            self.last_output = out
            return out

    def render(self, frame: np.ndarray, out: FrameOutput,
               stick_only: bool = False, show_angles: bool = True,
               fps: float = 0.0) -> np.ndarray:
        canvas = np.zeros_like(frame) if stick_only else frame.copy()
        if out.pose.present or not stick_only:
            draw_pose(canvas, out.pose, show_angles=show_angles)
        reps = out.session.get("reps", 0) if out.session else 0
        label = f"mode={self.mode} fps={fps:.1f} reps={reps}"
        cv2.putText(canvas, label, (10, canvas.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return canvas

    def process_jpeg(self, frame: np.ndarray, frame_id: int = 0,
                     stick_only: bool = False, fps: float = 0.0,
                     quality: int = 80) -> tuple[FrameOutput, bytes]:
        out = self.process(frame, frame_id)
        rendered = self.render(frame, out, stick_only=stick_only, fps=fps)
        ok, buf = cv2.imencode(".jpg", rendered,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return out, (buf.tobytes() if ok else b"")

    def close(self) -> None:
        with self._lock:
            try:
                self.pipe.close()
            except Exception:  # noqa: BLE001
                pass


def dummy_output(frame_id: int = 0) -> FrameOutput:
    return FrameOutput(frame_id=frame_id, pose=empty_pose(frame_id, "none"))
