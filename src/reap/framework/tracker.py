"""Continuous rep + trajectory tracking — part of the Oct Core Framework.

Watches one bending signal (e.g. knee angle) over time and counts reps:
stand -> bend -> stand = one rep. Also records the range of motion so
the app can show progress and spot tiring, shallow reps.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RepEvent:
    frame_id: int
    deepest: float      # smallest angle seen during the rep (deepest bend)
    shallowest: float   # largest angle (most upright)
    rom: float          # shallowest - deepest (degrees of motion)


class RepTracker:
    """Peak/valley rep detector with noise guards.

    Args:
        min_amplitude: smallest bend depth (degrees) that counts as a rep.
        min_gap: minimum frames between two counted reps (debounce).
        stand_threshold: angle above which the user counts as "standing".
    """

    def __init__(self, min_amplitude: float = 20.0, min_gap: int = 10,
                 stand_threshold: float = 150.0) -> None:
        self.min_amplitude = min_amplitude
        self.min_gap = min_gap
        self.stand_threshold = stand_threshold
        self.reps = 0
        self.history: list[float] = []
        self.events: list[RepEvent] = []
        self._in_rep = False
        self._deepest = float("inf")
        self._shallowest = float("-inf")
        self._last_rep_frame = -10 ** 9
        self._frames_since_stand = 0

    def update(self, value: float, frame_id: int) -> RepEvent | None:
        import math
        if not math.isfinite(value):
            return None
        self.history.append(float(value))
        self._deepest = min(self._deepest, float(value))
        self._shallowest = max(self._shallowest, float(value))

        if value >= self.stand_threshold:
            self._frames_since_stand = 0
        else:
            self._frames_since_stand += 1

        # rep starts when we leave standing and bend clearly
        if not self._in_rep and value < self.stand_threshold - 5.0:
            self._in_rep = True
            self._deepest = float(value)
            self._shallowest = float(value)

        # rep ends when back upright after a real bend
        if self._in_rep and value >= self.stand_threshold:
            depth = self._shallowest - self._deepest
            gap_ok = (frame_id - self._last_rep_frame) >= self.min_gap
            self._in_rep = False
            deepest, shallowest = self._deepest, self._shallowest
            self._deepest = float("inf")
            self._shallowest = float("-inf")
            if depth >= self.min_amplitude and gap_ok:
                self.reps += 1
                self._last_rep_frame = frame_id
                ev = RepEvent(frame_id=frame_id, deepest=deepest,
                              shallowest=shallowest, rom=depth)
                self.events.append(ev)
                return ev
        return None

    @property
    def best_rom(self) -> float:
        return max((e.rom for e in self.events), default=0.0)

    @property
    def last_rom(self) -> float:
        return self.events[-1].rom if self.events else 0.0
