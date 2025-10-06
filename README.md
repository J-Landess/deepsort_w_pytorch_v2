# deepsort-with-pytorch-v2

YOLOv8 + DeepSORT (PyTorch) package for vehicle detection and tracking (cars, buses, trucks).

Note: Model fine-tuned on UA-DETRAC, derived from YOLOv8 COCO weights.

## Installation (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

## CLI

```bash
deepsort-with-pytorch-v2 --help
deepsort-with-pytorch-v2 train --data config/detrac.yaml --epochs 50 --imgsz 1280
deepsort-with-pytorch-v2 infer --source path/to/video.mp4
deepsort-with-pytorch-v2 track --source path/to/video.mp4
```

## Training

- Uses `yolov8l.pt` (COCO pretrained) as init weights
- Data config: `config/detrac.yaml`
- Outputs: `runs/detect/train/weights/best.pt`

## Inference

- Run inference on images/videos
- Outputs annotated results to `runs/inference/`

## Tracking

- YOLOv8 detections + DeepSORT tracking
- Outputs tracked video with IDs

## Docker

```bash
docker build -t deepsort-with-pytorch-v2 .
# Show help
docker run --gpus all --rm -it -v "$PWD:/work" deepsort-with-pytorch-v2 --help
# Train
docker run --gpus all --rm -it -v "$PWD:/work" deepsort-with-pytorch-v2 train --data config/detrac.yaml --epochs 50 --imgsz 1280
# Infer
docker run --gpus all --rm -it -v "$PWD:/work" deepsort-with-pytorch-v2 infer --source path/to/video.mp4
# Track
docker run --gpus all --rm -it -v "$PWD:/work" deepsort-with-pytorch-v2 track --source path/to/video.mp4
```

## Dataset

Expected structure for UA-DETRAC after conversion:

```yaml
path: datasets/detrac
train: images/train
val: images/val
names:
  0: car
  1: bus
  2: truck
```

## License

MIT
