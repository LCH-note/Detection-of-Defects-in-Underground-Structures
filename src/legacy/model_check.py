
import torch
from ultralytics import YOLO

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "YOLO_model" / "yolov8s-seg.pt"

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

model = YOLO(MODEL_PATH)

print("model loaded")
