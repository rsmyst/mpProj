"""Nov milestone starter tests — edge setup (~30%)."""
from reap.edge.device import EdgeProfile, LatencyLog, should_process


def test_low_power_profile_is_smaller():
    full = EdgeProfile()
    lite = EdgeProfile.low_power()
    assert lite.frame_width < full.frame_width
    assert lite.run_every_n >= 2


def test_frame_skip_policy():
    prof = EdgeProfile.low_power()
    assert should_process(0, prof)
    assert not should_process(1, prof)
    assert should_process(2, prof)


def test_latency_log_suggests_when_slow():
    log = LatencyLog()
    log._sums["pose"] = 150.0
    log._counts["pose"] = 1
    assert "low-power" in log.suggest(budget_ms=100.0)
    assert "fine" in LatencyLog().suggest()
