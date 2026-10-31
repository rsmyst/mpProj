"""Coordinates the live source, pose service, JPEG stream, and cue engine."""
from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

from reap.audio.cues import CueEngine
from reap.backend.service import FrameOutput, PoseService
from reap.backend.sources import FramePacket, VideoSource


class DemoController:
    def __init__(self, source: VideoSource | None, service: PoseService,
                 cue_engine: CueEngine | None = None,
                 stick_only: bool = False, jpeg_quality: int = 80) -> None:
        self.source = source
        self.service = service
        self.cue_engine = cue_engine
        self.stick_only = bool(stick_only)
        self.jpeg_quality = int(jpeg_quality)

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest_out: FrameOutput | None = None
        self._latest_jpeg = b""
        self._last_packet: FramePacket | None = None
        self._fps = 0.0
        self._last_time = time.monotonic()
        self._push_id = 1_000_000_000

    def start(self) -> "DemoController":
        if self._thread is not None:
            return self
        if self.source is not None:
            self.source.start()
        self._running = True
        if self.source is not None:
            self._thread = threading.Thread(target=self._loop, name="pose-loop",
                                            daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        assert self.source is not None
        last_id = -1
        while self._running:
            packet = self.source.latest(after_id=last_id)
            if packet is None:
                if self.source.eof:
                    break
                time.sleep(0.004)
                continue
            last_id = packet.frame_id
            try:
                self._process(packet.image, packet.frame_id, update_packet=False)
            except Exception as exc:  # noqa: BLE001 - keep the stream alive
                print(f"[backend] frame {packet.frame_id} failed: {exc}")
                time.sleep(0.05)

    def _process(self, frame: np.ndarray, frame_id: int,
                 update_packet: bool) -> tuple[FrameOutput, bytes]:
        fps = self._tick_fps()
        out, jpeg = self.service.process_jpeg(
            frame, frame_id, stick_only=self.stick_only, fps=fps,
            quality=self.jpeg_quality,
        )
        with self._lock:
            self._latest_out = out
            if jpeg:
                self._latest_jpeg = jpeg
            if update_packet:
                self._last_packet = FramePacket(frame_id, frame, time.time())
        return out, jpeg

    @property
    def fps(self) -> float:
        return self._fps

    def _tick_fps(self) -> float:
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now
        if dt > 1e-6:
            inst = 1.0 / dt
            self._fps = inst if self._fps <= 0 else 0.9 * self._fps + 0.1 * inst
        return self._fps

    def process_pushed_frame(self, frame: np.ndarray) -> tuple[FrameOutput, bytes]:
        """Synchronous entry point for an external client pushing frames."""
        frame_id = self._push_id
        self._push_id += 1
        return self._process(frame, frame_id, update_packet=True)

    def handle_jpeg(self, data: bytes) -> tuple[FrameOutput, bytes] | None:
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return self.process_pushed_frame(frame)

    def snapshot(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    def latest_output(self) -> FrameOutput | None:
        with self._lock:
            return self._latest_out

    def pose_payload(self) -> dict[str, Any]:
        out = self.latest_output()
        if out is None:
            return {
                "ready": False,
                "mode": self.service.mode,
                "stick_only": self.stick_only,
                "fps": 0.0,
                "cue": "",
            }
        payload = out.to_dict(fps=self.fps, mode=self.service.mode,
                              stick_only=self.stick_only)
        payload["ready"] = True
        payload["last_spoken"] = (
            self.cue_engine.last_spoken if self.cue_engine is not None else ""
        )
        return payload

    def speak(self, text: str, force: bool = True) -> bool:
        if self.cue_engine is None:
            return False
        return self.cue_engine.say(text, force=force)

    def set_mode(self, mode: str) -> None:
        self.service.set_mode(mode)

    def set_stick_only(self, value: bool) -> None:
        self.stick_only = bool(value)

    def reset_session(self) -> None:
        self.service.reset_session()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.source is not None:
            self.source.stop()
        self.service.close()
        if self.cue_engine is not None:
            self.cue_engine.close()
