# R.E.A.P — Realtime Edge Agentic Physio-Buddy

Hybrid **MediaPipe + YOLO-Pose** tracking, live stick-model backend, rep counting,
and on-device spoken cues.

## Quickstart
```bash
uv sync --group dev
uv sync --extra voice                       # optional: Phonon + Piper TTS
uv run python scripts/download_weights.py   # fetch yolov8n-pose.pt once
uv run python scripts/download_phonon.py    # optional: Phonon-31M q8 weights (~64 MB)
uv run reap-demo --source 0 --mode hybrid   # browser demo, live stick model + speak buttons
uv run pytest
```

The demo server runs at <http://127.0.0.1:8000>. It shows the live MJPEG stick
model, joint angles, rep counter, model switcher (MediaPipe / YOLO / Hybrid), a
stick-only toggle, and buttons that speak demo cues. If Phonon cannot load it
falls back to Piper, then the system `espeak-ng`; without the voice extra it uses
`espeak-ng`, and with no audio device it stays silent. The demo speaks only when
you click a cue button; add `--auto-cues` if you want automatic rep feedback.
Test the voice directly with `uv run reap-speak "Good rep!"`.

Video file instead of webcam: `uv run reap-demo --source clip.mp4 --loop`.
No camera at all: `uv run reap-demo --push-only`, then POST JPEG frames to
`/api/frame` and read `/api/pose` (see the HTTP API below).

Benchmark visualizer (still available): `uv run reap-bench --source 0 --mode hybrid`
(keys 1/2/3 switch, `q` quits, `--save-csv run.csv`).

## Live backend

`reap.backend` is the reusable piece behind the demo:

- `VideoSource` — webcam, file, or RTSP/HTTP URL; background grabber keeps only
  the newest frame so inference lag does not build up.
- `PoseService` — frame in, COCO-17 `PoseResult` + OneEuro smoothing + joint
  angles + optional rep/session feedback out; renders the stick overlay.
- `DemoController` — wires source, service, JPEG stream, and cue engine together
  in threads.
- `DemoHTTPServer` — dependency-free HTTP API and demo page.

HTTP API:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/stream.mjpg` | live annotated stick-model stream |
| GET | `/snapshot.jpg` | latest annotated frame |
| GET | `/api/pose` | latest 17 normalized joints, angles, reps, fps |
| POST | `/api/frame` | body = raw JPEG; returns pose JSON (`?image=1` adds base64 JPEG) |
| POST | `/api/speak` | `{"text": "Good rep"}` → local TTS |
| POST | `/api/mode` | `{"mode": "mp" \| "yolo" \| "hybrid"}` |
| POST | `/api/stick` | `{"stick_only": true}` |
| POST | `/api/reset` | reset the current set |

Push-frame example:
```bash
curl --data-binary @frame.jpg -H 'Content-Type: image/jpeg' \
     http://127.0.0.1:8000/api/frame
```

## Voice I/O

- **Spoken output** uses local TTS with this priority:
  1. **Phonon-31M** from [`gradium/phonon-31m-en`](https://huggingface.co/gradium/phonon-31m-en)
     — 31M-parameter English TTS at 24 kHz, loaded through the `ptts` runtime.
     The q8 weights are only ~64 MB. Fetch them once with
     `scripts/download_phonon.py`. Defaults to the British female voice `Pippa`;
     set `audio.phonon_voice` to change it.
  2. **Piper** when a Piper voice bundle exists (`scripts/download_piper_voice.py`).
  3. **espeak-ng** as the zero-download fallback.
  No cloud calls are made during a set once the weights are cached.
- Choose the backend in `configs/default.yaml` (`audio.tts: auto | phonon | piper |
  espeak | none`) or per run: `uv run reap-demo --tts phonon`.
- Try another voice: `uv run reap-speak --voice Pippa "Good rep!"`. List catalog
  voices with `uv run ptts --list-voices`; `Pippa` is the default British female
  voice.
- Automatic rep cues are off by default. The demo only speaks when a cue button
  is clicked; opt in with `uv run reap-demo --auto-cues`.
- After pre-fetching, set `HF_HUB_OFFLINE=1` to guarantee the voice never touches
  the network during a set.

## Layout
- `src/reap/schemas.py` — canonical COCO-17 `PoseResult`
- `src/reap/pose/` — `mediapipe_backend`, `yolo_backend`, `hybrid` fusion, `filters` (OneEuro now, N-euro stub)
- `src/reap/kinematics/angles.py` — knee/hip/elbow angles
- `src/reap/framework/` — Oct milestone: exercise guides, rep tracker, guided session
- `src/reap/backend/` — live source → pose → stick model → HTTP/MJPEG API
- `src/reap/audio/` — local Phonon/Piper/espeak TTS + async spoken-cue queue
- `src/reap/edge/` — Nov starter (~30%): lite profile, frame policy, latency log
- `src/reap/apps/bench_pose.py` — side-by-side benchmark visualizer
- `src/reap/apps/demo_server.py` — live browser demo (`reap-demo`)
- `configs/default.yaml` — thresholds, model sizes, camera, audio, demo
- `present/` — progress-update deck sources (LaTeX + mermaid diagrams)

## Pipeline (per Architecture slide)
`Camera → HybridPose(MP+YOLO) → Predictive Kinematics → ST-GCN → Fatigue → Audio Engine`
This milestone implements the first box + stubs for the rest, plus a live
stick-model backend and local spoken-cue loop.
