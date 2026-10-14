"""Oct milestone tests — core framework (no camera needed)."""
import math

from reap.framework.exercises import check_bend, get, list_names
from reap.framework.session import WorkoutSession
from reap.framework.tracker import RepTracker


def test_three_exercise_guides_registered():
    names = list_names()
    assert "chair_stand" in names
    assert "squat" in names
    assert "leg_raise" in names


def test_healthy_bend_passes_shallow_fails():
    prof = get("chair_stand")
    ok, _ = check_bend(prof, 95.0)
    assert ok
    ok2, msg2 = check_bend(prof, 160.0)
    assert not ok2 and isinstance(msg2, str) and len(msg2) > 0


def _sine_like_reps(n_reps=3, frames_per_rep=30):
    """Synthetic knee-angle signal: 160 (stand) -> 95 (bend) -> 160."""
    vals = []
    for _ in range(n_reps):
        for i in range(frames_per_rep):
            frac = i / frames_per_rep
            # triangular wave down then up
            if frac < 0.5:
                vals.append(160.0 - (160.0 - 95.0) * (frac * 2))
            else:
                vals.append(95.0 + (160.0 - 95.0) * ((frac - 0.5) * 2))
    return vals


def test_tracker_counts_three_reps():
    tr = RepTracker(min_amplitude=20.0, min_gap=5, stand_threshold=150.0)
    vals = _sine_like_reps(3)
    events = 0
    for i, v in enumerate(vals):
        if tr.update(v, i) is not None:
            events += 1
    assert tr.reps == 3 and events == 3
    assert tr.best_rom > 40.0


def test_session_coaches_full_set():
    sess = WorkoutSession("chair_stand")
    vals = _sine_like_reps(2)
    for i, v in enumerate(vals):
        sess.update({"left_knee": v, "right_knee": v}, i)
    s = sess.summary()
    assert s.reps == 2
    assert s.good == 2 and s.needs_fix == 0
    assert len(s.messages) == 2
