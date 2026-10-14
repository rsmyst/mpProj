"""Exercise plug-in packs — the Oct milestone: Core Framework.

Each profile describes one mobility exercise in plain terms:
what joint to watch, what bend range is healthy, and what spoken
cue to give when form drifts. New exercises can be added without
touching the app core — just register another profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExerciseProfile:
    """Healthy-form guide for one exercise."""
    name: str                    # machine id, e.g. "chair_stand"
    display_name: str            # human label, e.g. "Chair Stand"
    joint: str                   # which angle to watch ("left_knee", ...)
    bend_min: float              # deepest healthy bend (degrees, smaller = deeper)
    bend_max: float              # shallowest acceptable bend (degrees)
    cue_too_deep: str = "Ease up — you are bending deeper than the safe range."
    cue_too_shallow: str = "Bend a little deeper for a full rep."
    cue_good: str = "Good rep — nice and controlled."
    tolerance: float = 8.0       # extra slack before flagging (degrees)


_REGISTRY: dict[str, ExerciseProfile] = {}


def register(profile: ExerciseProfile) -> None:
    _REGISTRY[profile.name] = profile


def get(name: str) -> ExerciseProfile:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown exercise {name!r}; available: {sorted(_REGISTRY)}") from None


def list_names() -> list[str]:
    return sorted(_REGISTRY)


def check_bend(profile: ExerciseProfile, angle: float) -> tuple[bool, str]:
    """Check one rep's deepest bend against the healthy range.

    Returns (ok, message). `angle` is in degrees; smaller = deeper bend.
    """
    import math
    if not math.isfinite(angle):
        return False, "I could not see your joints clearly — face the camera."
    if angle < profile.bend_min - profile.tolerance:
        return False, profile.cue_too_deep
    if angle > profile.bend_max + profile.tolerance:
        return False, profile.cue_too_shallow
    return True, profile.cue_good


def _builtins() -> list[ExerciseProfile]:
    return [
        ExerciseProfile(
            name="chair_stand",
            display_name="Chair Stand",
            joint="left_knee",
            bend_min=75.0,
            bend_max=115.0,
            cue_too_deep="That is deeper than needed — rise a little higher.",
            cue_too_shallow="Bend your knees a little more, keep your chest up.",
        ),
        ExerciseProfile(
            name="squat",
            display_name="Supported Squat",
            joint="left_knee",
            bend_min=70.0,
            bend_max=120.0,
            cue_too_deep="Too deep — stop a little higher to protect your knees.",
            cue_too_shallow="Sink a little lower for a full rep, back straight.",
        ),
        ExerciseProfile(
            name="leg_raise",
            display_name="Lying Leg Raise",
            joint="left_hip",
            bend_min=45.0,
            bend_max=85.0,
            cue_too_deep="Lower your leg a little — you lifted past the safe range.",
            cue_too_shallow="Lift your leg a bit higher for a full rep.",
        ),
    ]


for _p in _builtins():
    register(_p)
