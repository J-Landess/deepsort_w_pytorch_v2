from typing import Optional
from ultralytics import YOLO


def main(data: str = "config/detrac.yaml", epochs: int = 50, imgsz: int = 1280, batch: int = 8, device: str = ""):
    # Load COCO-pretrained YOLOv8l
    model = YOLO("yolov8l.pt")

    # Restrict classes to car(2), bus(5), truck(7) in COCO mapping; during training use the custom dataset labels
    # Ultralytics uses dataset class mapping from the YAML; training will learn only those classes in dataset

    train_args = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
    }
    if device:
        train_args["device"] = device

    model.train(**train_args)


if __name__ == "__main__":
    main()
