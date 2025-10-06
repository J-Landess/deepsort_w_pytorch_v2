import os
import os.path as osp
import yaml
from typing import Any, Dict

import cv2


def ensure_dir(path: str) -> None:
    if path and not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def draw_box_with_label(frame, box, label: str, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        ((tw, th), baseline) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
