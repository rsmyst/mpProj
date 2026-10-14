"""Guided workout session — the app core from the Oct milestone.

Ties everything together: each video frame gives joint angles, the
session checks them against the chosen exercise guide, counts the rep,
and returns a short spoken-style message ("Good rep", "Bend a little
deeper", ...). No screen needed — messages are meant to be spoken aloud.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from reap.framework.exercises import ExerciseProfile, check_bend, get as get_exercise
from reap.framework.tracker import RepTracker


@dataclass
class SessionSummary:
    exercise: str
    reps: int = 0
    good: int = 0
    needs_fix: int = 0
    best_rom: float = 0.0
    messages: list[str] = field(default_factory=list)


class WorkoutSession:
    """Run one guided set of a single exercise."""

    def __init__(self, exercise: str = "chair_stand",
                 min_amplitude: float = 20.0, min_gap: int = 10,
                 stand_threshold: float = 150.0,
                 live_cue_cooldown: int = 45) -> None:
        self.profile: ExerciseProfile = get_exercise(exercise)
        self.tracker = RepTracker(min_amplitude=min_amplitude, min_gap=min_gap,
                                  stand_threshold=stand_threshold)
        self.good = 0
        self.needs_fix = 0
        self.messages: list[str] = []
        self._cooldown = live_cue_cooldown
        self._last_cue_frame = -10 ** 9

    def update(self, angles: dict[str, float], frame_id: int) -> dict:
        """Feed one frame of joint angles; get back coaching feedback."""
        signal = float(angles.get(self.profile.joint, float("nan")))
        event = self.tracker.update(signal, frame_id)
        feedback = ""
        rep_done = False
        if event is not None:
            rep_done = True
            ok, msg = check_bend(self.profile, event.deepest)
            if ok:
                self.good += 1
            else:
                self.needs_fix += 1
            feedback = msg
            self.messages.append(f"Rep {self.tracker.reps}: {msg}")
            self._last_rep_frame = frame_id
        return {
            "reps": self.tracker.reps,
            "good": self.good,
            "needs_fix": self.needs_fix,
            "rep_done": rep_done,
            "feedback": feedback,
            "rom": self.tracker.last_rom,
        }

    def summary(self) -> SessionSummary:
        return SessionSummary(
            exercise=self.profile.display_name,
            reps=self.tracker.reps,
            good=self.good,
            needs_fix=self.needs_fix,
            best_rom=self.tracker.best_rom,
            messages=list(self.messages),
        )
