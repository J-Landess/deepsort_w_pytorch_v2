import os
import os.path as osp
from typing import Optional

import cv2
from ultralytics import YOLO

from .utils import ensure_dir


def is_video_path(path: str) -> bool:
    return osp.splitext(path)[1].lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main(source: str, weights: str = "runs/detect/train/weights/best.pt", imgsz: int = 1280, device: str = ""):
    out_dir = "runs/inference"
    ensure_dir(out_dir)

    model = YOLO(weights)
    kwargs = {"imgsz": imgsz, "save": True, "project": out_dir, "name": "exp", "exist_ok": True}
    if device:
        kwargs["device"] = device

    if is_video_path(source):
        model.predict(source, **kwargs)
    else:
        model.predict(source, **kwargs)


if __name__ == "__main__":
    # Example usage
    # main(source="path/to/image_or_video")
    pass
