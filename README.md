# R.E.A.P — Realtime Edge Agentic Physio-Buddy

Sept milestone: **Tracking Model Integration** — hybrid MediaPipe + YOLO-Pose scaffold.

## Quickstart
```bash
uv sync --group dev
uv run python scripts/download_weights.py   # fetch yolov8n-pose.pt once
uv run reap-bench --source 0 --mode hybrid # webcam demo, keys 1/2/3 switch, q quits
uv run pytest
```

Video file instead of webcam: `uv run reap-bench --source clip.mp4 --save-csv run.csv`

## Layout
- `src/reap/schemas.py` — canonical COCO-17 `PoseResult`
- `src/reap/pose/` — `mediapipe_backend`, `yolo_backend`, `hybrid` fusion, `filters` (OneEuro now, N-euro stub)
- `src/reap/kinematics/angles.py` — knee/hip/elbow angles
- `src/reap/apps/bench_pose.py` — side-by-side benchmark visualizer
- `configs/default.yaml` — thresholds, model sizes, camera

## Pipeline (per Architecture slide)
`Camera → HybridPose(MP+YOLO) → Predictive Kinematics → ST-GCN → Fatigue → Audio Engine`
This milestone implements the first box + stubs for the rest.
