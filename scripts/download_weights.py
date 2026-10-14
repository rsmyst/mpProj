"""Pre-fetch YOLO pose weights so edge runs work offline (zero-cloud req)."""
import yaml
from ultralytics import YOLO

cfg = {}
try:
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f) or {}
except FileNotFoundError:
    pass

weights = cfg.get("yolo", {}).get("weights", "yolov8n-pose.pt")
print(f"[download] fetching {weights} ...")
YOLO(weights)
print("[download] done.")
