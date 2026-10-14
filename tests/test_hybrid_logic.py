"""Hybrid fusion unit tests — mocked backends, no camera needed."""
import numpy as np

from reap.pose.hybrid import HybridPose, fuse_confidence_weighted
from reap.schemas import PoseResult


def _pose(val: float, conf: float, source="mp") -> PoseResult:
    return PoseResult(kpts=np.full((17, 2), val, dtype=np.float32),
                      conf=np.full((17,), conf, dtype=np.float32),
                      source=source)


def test_fuse_prefers_confident_backend():
    a = _pose(0.2, 0.9, "mp")
    b = _pose(0.8, 0.1, "yolo")
    f = fuse_confidence_weighted(a, b)
    assert f.source == "fused"
    # weighted avg = (0.2*0.9+0.8*0.1)/1.0 = 0.26
    assert abs(float(f.kpts[0, 0]) - 0.26) < 1e-5
    assert abs(float(f.conf[0]) - 0.9) < 1e-6


def test_fuse_takes_only_visible_side():
    a = _pose(0.3, 0.9, "mp")
    b = _pose(0.7, 0.0, "yolo")
    f = fuse_confidence_weighted(a, b)
    assert abs(float(f.kpts[0, 0]) - 0.3) < 1e-6


def test_hybrid_mp_only_mode_never_touches_yolo():
    pipe = HybridPose(mode="mp")
    import numpy as np

    class FakeMP:
        def process(self, bgr, frame_id=0):
            return _pose(0.5, 0.9, "mp")
        def close(self): pass

    pipe._mp = FakeMP()  # type: ignore
    out = pipe.process(np.zeros((64, 64, 3), np.uint8), 0)
    assert out.source == "mp"
