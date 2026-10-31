"""Voice-cue tests — silent backend, no speakers required."""
from reap.audio.cues import CueEngine
from reap.audio.tts import NullTTS, create_tts


def test_null_tts_and_create_none():
    assert isinstance(create_tts("none"), NullTTS)


def test_cooldown_suppresses_repeats():
    now = [100.0]
    tts = NullTTS()
    cues = CueEngine(tts, min_gap_s=2.0, clock=lambda: now[0])
    assert cues.say("bend deeper") is True
    assert cues.say("bend deeper") is False
    now[0] += 3.0
    assert cues.say("bend deeper") is True
    assert cues.say("forced", force=True) is True
    cues.wait_idle()
    assert tts.spoken == ["bend deeper", "bend deeper", "forced"]
    cues.close()


def test_say_now_is_synchronous():
    tts = NullTTS()
    cues = CueEngine(tts, min_gap_s=10.0)
    cues.say_now("good rep")
    assert tts.spoken == ["good rep"]
    assert cues.last_spoken == "good rep"
    cues.close()
