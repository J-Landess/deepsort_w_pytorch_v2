import os.path as osp
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from .utils import ensure_dir, draw_box_with_label


COCO_VEHICLE_CLASS_IDS = {2, 5, 7}  # car, bus, truck in COCO index space


def main(source: str, weights: str = "runs/detect/train/weights/best.pt", device: str = "", conf: float = 0.25):
    assert osp.exists(source), f"Source not found: {source}"
    out_dir = "runs/track"
    ensure_dir(out_dir)

    model = YOLO(weights)
    yolo_kwargs = {"conf": conf}
    if device:
        yolo_kwargs["device"] = device

    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), f"Could not open video: {source}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out_path = osp.join(out_dir, osp.splitext(osp.basename(source))[0] + "_tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, verbose=False, **yolo_kwargs)
        dets = []
        if results:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, cls in zip(boxes_xyxy, confs, classes):
                    # If trained on DETRAC (3 classes), accept all; otherwise filter COCO vehicles
                    if len(r.names) == 3 or cls in COCO_VEHICLE_CLASS_IDS:
                        dets.append(((x1, y1, x2 - x1, y2 - y1), c, cls))

        tracks = tracker.update_tracks(dets, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            draw_box_with_label(frame, (x1, y1, x2, y2), f"ID {tid}")

        writer.write(frame)

    cap.release()
    writer.release()


if __name__ == "__main__":
    # main(source="path/to/video.mp4")
    pass
