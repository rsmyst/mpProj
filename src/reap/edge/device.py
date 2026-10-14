"""Edge setup starter — Nov milestone (first ~30%).

Goal for Nov: make the app cheap enough to run fully on-device.
This starter covers the basics only:
  1. a lightweight mode (small frames, skip frames when idle),
  2. a per-frame stopwatch that logs where time goes,
  3. a simple rule that suggests lighter settings when running slow.

The remaining ~70% (GPU tuning, model quantization, full
battery/heat testing) is scheduled as next work.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EdgeProfile:
    """Lightweight on-device settings."""
    name: str = "balanced"       # balanced | low_power
    frame_width: int = 640
    frame_height: int = 480
    run_every_n: int = 1         # 1 = every frame, 2 = every other frame, ...
    model_size: str = "light"    # light | full

    @classmethod
    def low_power(cls) -> "EdgeProfile":
        return cls(name="low_power", frame_width=320, frame_height=240,
                   run_every_n=2, model_size="light")


def should_process(frame_id: int, profile: EdgeProfile) -> bool:
    """Frame-skip policy: only process 1 in every N frames."""
    return (frame_id % max(1, profile.run_every_n)) == 0


def target_frame_size(frame_shape: tuple[int, ...], profile: EdgeProfile) -> tuple[int, int]:
    """Target (width, height) the camera frame should be shrunk to."""
    return (profile.frame_width, profile.frame_height)


class LatencyLog:
    """Tiny stopwatch: record per-stage timings, report averages."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._t0: float = 0.0
        self._stage: str = ""

    def begin(self, stage: str) -> None:
        self._stage = stage
        self._t0 = time.perf_counter()

    def end(self) -> float:
        dt = (time.perf_counter() - self._t0) * 1000.0
        self._sums[self._stage] = self._sums.get(self._stage, 0.0) + dt
        self._counts[self._stage] = self._counts.get(self._stage, 0) + 1
        return dt

    def average(self, stage: str) -> float:
        n = self._counts.get(stage, 0)
        return self._sums.get(stage, 0.0) / n if n else 0.0

    def total_average(self) -> float:
        n = sum(self._counts.values())
        return sum(self._sums.values()) / n if n else 0.0

    def suggest(self, budget_ms: float = 100.0) -> str:
        """Plain-language suggestion if the device is too slow."""
        total = sum(self.average(s) for s in self._sums)
        if total > budget_ms:
            return ("Running slow — switch to low-power mode "
                    "(smaller picture, every other frame).")
        return "Speed looks fine — keep current settings."
