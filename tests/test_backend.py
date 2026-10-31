"""Live backend tests — fake pose source, no camera required."""
import json
import threading
import urllib.request

import cv2
import numpy as np
import pytest

from reap.audio.cues import CueEngine
from reap.audio.tts import NullTTS
from reap.backend.controller import DemoController
from reap.backend.server import DemoHTTPServer
from reap.backend.service import PoseService
from reap.schemas import COCO_NAMES, PoseResult, empty_pose


def _pose_for_knee(bend: bool) -> PoseResult:
    pose = empty_pose(0, "fake")
    pose.conf[:] = 0.95
    # Standing: hip/knee/ankle are collinear (180 deg). Bend: ankle forward (90).
    pose.kpts[11] = [0.45, 0.30]
    pose.kpts[13] = [0.45, 0.50]
    pose.kpts[15] = [0.45, 0.70] if not bend else [0.65, 0.50]
    pose.kpts[12] = [0.55, 0.30]
    pose.kpts[14] = [0.55, 0.50]
    pose.kpts[16] = [0.55, 0.70] if not bend else [0.75, 0.50]
    return pose


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.mode = "hybrid"
        self.bend = False

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def process(self, bgr: np.ndarray, frame_id: int = 0) -> PoseResult:
        pose = _pose_for_knee(self.bend)
        pose.frame_id = frame_id
        return pose

    def close(self) -> None:
        pass


def _service(exercise="chair_stand", cue_engine=None):
    return PoseService(backend=FakeBackend(), cfg={"session": {"exercise": exercise}},
                       exercise=exercise, smoothing=False, cue_engine=cue_engine)


def test_service_returns_joints_angles_and_rep_cue():
    tts = NullTTS()
    cues = CueEngine(tts, min_gap_s=0.0)
    svc = _service(cue_engine=cues)
    backend = svc.pipe

    frame = np.zeros((480, 640, 3), np.uint8)
    backend.bend = False
    out1 = svc.process(frame, 0)
    assert out1.pose.present
    assert out1.session["rep_done"] is False
    assert out1.angles["left_knee"] > 150

    backend.bend = True
    out2 = svc.process(frame, 10)
    assert out2.angles["left_knee"] < 120
    assert out2.session["rep_done"] is False

    backend.bend = False
    out3 = svc.process(frame, 20)
    assert out3.session["rep_done"] is True
    assert out3.cue
    cues.wait_idle()
    assert tts.spoken == [out3.cue]
    cues.close()


def test_process_jpeg_and_stick_only_render():
    svc = _service(exercise="none")
    frame = np.full((240, 320, 3), 200, np.uint8)
    out, jpeg = svc.process_jpeg(frame, 1, stick_only=True)
    assert jpeg.startswith(b"\xff\xd8")
    rendered = svc.render(frame, out, stick_only=True)
    assert rendered.shape == frame.shape
    assert int(rendered.max()) > 0
    assert int(rendered[0, 0, 0]) == 0


def test_service_to_dict_has_coco17():
    svc = _service(exercise="none")
    out = svc.process(np.zeros((240, 320, 3), np.uint8), 3)
    payload = out.to_dict(fps=12.5, mode="mp")
    assert payload["frame_id"] == 3
    assert [j["name"] for j in payload["joints"]] == COCO_NAMES
    assert payload["pose"]["source"] == "fake"


def _fake_source():
    class Source:
        def __init__(self):
            self.source = 0
            self.eof = False
            self._id = -1
            self._packet = None
            self._running = False

        def start(self):
            self._running = True

        def latest(self, after_id=-1):
            if not self._running or self._id > 2:
                self.eof = True
                return None
            self._id += 1
            from reap.backend.sources import FramePacket
            self._packet = FramePacket(self._id, np.zeros((240, 320, 3), np.uint8), 0.0)
            return self._packet

        def stop(self):
            self._running = False

    return Source()


def test_controller_pushes_frames_and_serves():
    svc = _service(exercise="none")
    tts = NullTTS()
    cues = CueEngine(tts, min_gap_s=0.0)
    ctrl = DemoController(_fake_source(), svc, cue_engine=cues)
    ctrl.start()
    for _ in range(50):
        if ctrl.snapshot():
            break
        import time
        time.sleep(0.02)
    assert ctrl.snapshot().startswith(b"\xff\xd8")
    assert ctrl.pose_payload()["ready"] is True

    server = DemoHTTPServer(("127.0.0.1", 0), ctrl)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health") as r:
            assert json.loads(r.read())["status"] == "ok"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/pose") as r:
            data = json.loads(r.read())
            assert data["ready"] and len(data["joints"]) == 17
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/speak",
            data=json.dumps({"text": "hello demo"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["ok"] is True
        cues.wait_idle()
        assert "hello demo" in tts.spoken

        ok, buf = cv2.imencode(".jpg", np.full((120, 160, 3), 128, np.uint8))
        assert ok
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/frame",
            data=buf.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read())
            assert data["ready"] and len(data["joints"]) == 17

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/stick",
            data=json.dumps({"stick_only": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["stick_only"] is True
    finally:
        server.shutdown()
        server.server_close()
        ctrl.close()


def test_mode_switch_validation():
    svc = _service(exercise="none")
    svc.set_mode("yolo")
    assert svc.mode == "yolo"
    with pytest.raises(ValueError):
        svc.set_mode("bogus")

def test_demo_speaks_only_on_button_by_default():
    from pathlib import Path

    import yaml

    from reap.apps.demo_server import build_parser

    args = build_parser().parse_args([])
    assert args.auto_cues is False
    assert args.no_cues is False

    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/default.yaml").read_text()
    )
    assert cfg["audio"]["auto_cues"] is False
    assert cfg["audio"]["tts"] == "auto"
    assert cfg["audio"]["phonon_voice"] == "Pippa"
