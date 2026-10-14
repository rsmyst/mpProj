"""Live benchmark: MediaPipe vs YOLO vs Hybrid side-by-side.

Usage:
  uv run reap-bench --source 0 --mode hybrid --config configs/default.yaml
Keys: 1/2/3 switch mp/yolo/hybrid, q quits.
"""
from __future__ import annotations

import argparse
import csv
import time

import cv2
import yaml

from reap.pose.hybrid import HybridPose
from reap.viz.overlay import draw_pose


def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def parse_source(s: str):
    return int(s) if s.isdigit() else s


def main() -> None:
    ap = argparse.ArgumentParser(description="REAP pose benchmark visualizer")
    ap.add_argument("--source", default="0")
    ap.add_argument("--mode", default="hybrid", choices=["mp", "yolo", "hybrid"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--save-csv", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hyb_cfg = cfg.get("hybrid", {})
    mp_cfg = cfg.get("mediapipe", {})
    yolo_cfg = cfg.get("yolo", {})
    bench_cfg = cfg.get("bench", {})

    save_csv = args.save_csv or bench_cfg.get("save_csv", "")
    show_angles = bool(bench_cfg.get("show_angles", True))

    cap = cv2.VideoCapture(parse_source(args.source))
    cam_cfg = cfg.get("camera", {})
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.source!r}")

    mode = args.mode
    pipe = HybridPose(mode=mode,
                      mp_conf_threshold=hyb_cfg.get("mp_conf_threshold", 0.5),
                      validate_every_n=hyb_cfg.get("validate_every_n", 15),
                      mp_kwargs=mp_cfg, yolo_kwargs=yolo_cfg)
    csv_fh = open(save_csv, "w", newline="") if save_csv else None
    writer = csv.writer(csv_fh) if csv_fh else None
    if writer:
        writer.writerow(["frame", "mode", "source", "avg_conf", "latency_ms", "fps"])

    frame_id, t_prev, fps = 0, time.time(), 0.0
    print(f"[reap-bench] mode={mode} keys: 1=mp 2=yolo 3=hybrid q=quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pipe.process(frame, frame_id)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - t_prev)) if frame_id else 0.0
            t_prev = now
            vis = draw_pose(frame, res, show_angles=show_angles)
            cv2.putText(vis, f"mode={mode} fps={fps:.1f} (1/2/3,q)",
                        (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2)
            cv2.imshow("REAP hybrid-pose bench", vis)
            if writer:
                writer.writerow([frame_id, mode, res.source,
                                 f"{res.avg_conf:.3f}",
                                 f"{res.latency_ms:.2f}", f"{fps:.1f}"])
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("1"), ord("2"), ord("3")):
                mode = {"1": "mp", "2": "yolo", "3": "hybrid"}[chr(key)]
                pipe.close()
                pipe = HybridPose(mode=mode,
                                  mp_conf_threshold=hyb_cfg.get("mp_conf_threshold", 0.5),
                                  validate_every_n=hyb_cfg.get("validate_every_n", 15),
                                  mp_kwargs=mp_cfg, yolo_kwargs=yolo_cfg)
                print(f"[reap-bench] switched to {mode}")
            frame_id += 1
    finally:
        pipe.close()
        cap.release()
        cv2.destroyAllWindows()
        if csv_fh:
            csv_fh.close()
            print(f"[reap-bench] wrote {save_csv}")


if __name__ == "__main__":
    main()
