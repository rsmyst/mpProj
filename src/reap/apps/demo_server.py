"""Live demo: webcam -> stick model + click-to-speak cues in the browser.

Run:
  uv run reap-demo --source 0 --mode hybrid
  uv run reap-demo --source clip.mp4 --loop
  uv run reap-demo --push-only          # accept JPEG frames at POST /api/frame

The page is served on http://127.0.0.1:8000 and shows the MJPEG stick model,
joint angles, rep counter, model switcher, and buttons that speak demo cues.
"""
from __future__ import annotations

import argparse
import threading
import time
import webbrowser

import yaml

from reap.audio.cues import CueEngine
from reap.audio.tts import create_tts
from reap.backend.controller import DemoController
from reap.backend.server import DemoHTTPServer
from reap.backend.service import PoseService
from reap.backend.sources import VideoSource


def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="REAP live stick-model demo")
    ap.add_argument("--source", default=None,
                    help="camera index, video path, or rtsp/http URL")
    ap.add_argument("--push-only", action="store_true",
                    help="no local camera; POST JPEGs to /api/frame")
    ap.add_argument("--mode", default=None, choices=["mp", "yolo", "hybrid"])
    ap.add_argument("--exercise", default=None,
                    help="chair_stand | squat | leg_raise | none")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--stick-only", action="store_true", default=None,
                    help="start with skeleton-only output")
    ap.add_argument("--loop", action="store_true", default=None,
                    help="loop video files")
    ap.add_argument("--no-smoothing", action="store_true",
                    help="disable OneEuro smoothing")
    ap.add_argument("--tts", default=None,
                    choices=["auto", "phonon", "piper", "espeak", "none"])
    ap.add_argument("--no-voice", action="store_true",
                    help="disable all spoken output")
    ap.add_argument("--auto-cues", action="store_true",
                    help="also speak rep feedback automatically (off by default)")
    ap.add_argument("--no-cues", action="store_true",
                    help="force-disable automatic rep cues")
    ap.add_argument("--no-open", action="store_true",
                    help="do not open a browser window")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    cam_cfg = cfg.get("camera", {})
    demo_cfg = cfg.get("demo", {})
    audio_cfg = cfg.get("audio", {})

    source_value = args.source if args.source is not None else cam_cfg.get("source", 0)
    mode = args.mode or cfg.get("hybrid", {}).get("mode", "hybrid")
    exercise = args.exercise if args.exercise is not None else cfg.get("session", {}).get("exercise", "chair_stand")
    stick_only = args.stick_only if args.stick_only is not None else bool(demo_cfg.get("stick_only", False))
    loop = args.loop if args.loop is not None else bool(cam_cfg.get("loop", False))
    host = args.host or demo_cfg.get("host", "127.0.0.1")
    port = args.port if args.port is not None else int(demo_cfg.get("port", 8000))

    source = None
    if not args.push_only:
        try:
            source = VideoSource(
                source_value,
                width=int(cam_cfg.get("width", 640)),
                height=int(cam_cfg.get("height", 480)),
                mirror=bool(cam_cfg.get("mirror", False)),
                loop=loop,
            )
        except RuntimeError as exc:
            print(f"[reap-demo] {exc}")
            print("[reap-demo] try --source 1, a video file, or --push-only")
            return 2

    if args.no_voice:
        tts = create_tts("none")
    else:
        tts = create_tts(
            args.tts or audio_cfg.get("tts", "auto"),
            phonon_model=audio_cfg.get("phonon_model", "hf://gradium/phonon-31m-en"),
            phonon_voice=audio_cfg.get("phonon_voice") or None,
            phonon_lang=audio_cfg.get("phonon_lang", "en"),
            phonon_quant=audio_cfg.get("phonon_quant", "q8"),
            phonon_device=audio_cfg.get("phonon_device", "cpu"),
            phonon_voice_catalog=audio_cfg.get("phonon_voice_catalog") or None,
            piper_model=audio_cfg.get("piper_model") or None,
            espeak_voice=audio_cfg.get("espeak_voice", "en"),
            espeak_speed=int(audio_cfg.get("espeak_speed", 165)),
        )
    cue_engine = CueEngine(tts, min_gap_s=float(audio_cfg.get("cue_gap_s", 1.2)))
    auto_cues_on = bool(audio_cfg.get("auto_cues", False)) or args.auto_cues
    auto_cues = None if (args.no_cues or not auto_cues_on) else cue_engine

    service = PoseService(
        cfg, mode=mode, exercise=exercise,
        smoothing=not args.no_smoothing, cue_engine=auto_cues,
    )
    controller = DemoController(
        source, service, cue_engine=cue_engine,
        stick_only=stick_only,
        jpeg_quality=int(demo_cfg.get("jpeg_quality", 80)),
    )
    server = DemoHTTPServer((host, port), controller, verbose=args.verbose)
    controller.start()

    url = f"http://{host}:{port}/"
    print(f"[reap-demo] source={source_value!r} mode={mode} exercise={exercise}")
    print(f"[reap-demo] open {url}  (Ctrl-C to stop)")
    if not args.no_open:
        threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)),
                         daemon=True).start()

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        controller.close()
        print("[reap-demo] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
