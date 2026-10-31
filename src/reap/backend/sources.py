"""Live video sources: webcam, video file, or RTSP/HTTP stream.

A background grabber keeps only the newest frame, which avoids the growing
buffer lag that appears when inference is slower than the camera.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


def parse_source(value: str | int) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


@dataclass
class FramePacket:
    frame_id: int
    image: np.ndarray
    timestamp: float


class VideoSource:
    def __init__(self, source: str | int = 0, width: int = 640,
                 height: int = 480, mirror: bool = False,
                 loop: bool = False) -> None:
        self.source = parse_source(source)
        self.mirror = bool(mirror)
        self.loop = bool(loop)
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source {self.source!r}")

        self._lock = threading.Lock()
        self._packet: FramePacket | None = None
        self._frame_id = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._eof = False
        self._is_file = (
            isinstance(self.source, str)
            and not self.source.startswith(("rtsp://", "http://", "https://"))
        )
        file_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._frame_period = 1.0 / file_fps if self._is_file and file_fps > 0 else 0.0

    @property
    def opened(self) -> bool:
        return bool(self.cap.isOpened())

    def read(self) -> FramePacket | None:
        ok, frame = self.cap.read()
        if not ok:
            if self.loop and not isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                self._eof = True
                return None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        packet = FramePacket(self._frame_id, frame, time.time())
        self._frame_id += 1
        with self._lock:
            self._packet = packet
        return packet

    def start(self) -> "VideoSource":
        if self._thread is not None:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop,
                                        name="video-source", daemon=True)
        self._thread.start()
        return self

    def _grab_loop(self) -> None:
        while self._running:
            if self.read() is None:
                break
            if self._frame_period:
                time.sleep(self._frame_period)
            else:
                time.sleep(0)

    def current(self) -> FramePacket | None:
        with self._lock:
            return self._packet

    def latest(self, after_id: int = -1) -> FramePacket | None:
        with self._lock:
            packet = self._packet
        if packet is None or packet.frame_id <= after_id:
            return None
        return packet

    @property
    def eof(self) -> bool:
        return self._eof

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.cap.release()
