# deepsort-with-pytorch-v2

YOLOv8 + DeepSORT (PyTorch) package for vehicle detection and tracking (cars, buses, trucks).

Note: Fine-tuning targets UA-DETRAC-style data, starting from YOLOv8 COCO weights.

## Quickstart (Conda, Python 3.11, CPU)

```bash
# Ensure conda is available in your shell, then:
conda create -y -n dspv2 python=3.11
conda activate dspv2

# Install PyTorch (use a CUDA build if you have a GPU)
conda install -y -c pytorch -c conda-forge pytorch torchvision cpuonly

# Install this package (pulls ultralytics, deep-sort-realtime, opencv-python, PyYAML, typer)
pip install -U pip
pip install -e .

# CLI
deepsort-with-pytorch-v2 --help
```

After `pip install -e .`, console scripts `train`, `infer`, and `track` are also available (same modules as the Typer subcommands).

## CLI

Config lives under the package directory (there is no top-level `config/`):

```bash
deepsort-with-pytorch-v2 train \
  --data deepsort_with_pytorch_v2/config/detrac.yaml \
  --epochs 50 --imgsz 1280

deepsort-with-pytorch-v2 infer --source path/to/video_or_image
deepsort-with-pytorch-v2 track --source path/to/video.mp4
deepsort-with-pytorch-v2 iterate --source path/to/video.mp4 --stride 3 --mode extract
```

Equivalent entry points after install:

```bash
train --data deepsort_with_pytorch_v2/config/detrac.yaml --epochs 50
infer --source path/to/video_or_image
track --source path/to/video.mp4
```

`iterate` modes: `iterate` (count frames) or `extract` (write frames to `--outdir`, default `runs/frames`).

## Iterating and processing videos

`deepsort_with_pytorch_v2.iterator_tools` streams frames for long videos.

```python
from deepsort_with_pytorch_v2.iterator_tools import FrameIterator

for frame in FrameIterator("traffic.mp4", stride=2):
    # frame is RGB numpy array
    ...
```

```python
from deepsort_with_pytorch_v2.iterator_tools import FrameIterator, FrameSlicer

slicer = FrameSlicer(FrameIterator("traffic.mp4"), slice_length=150)
for chunk in slicer:  # list of frames
    ...
```

```python
from deepsort_with_pytorch_v2.iterator_tools import VideoInferenceIterator

vi = VideoInferenceIterator(
    "traffic.mp4", model_path="yolov8l.pt", batch_size=4, conf=0.25, imgsz=1280
)
for frame_idx, annotated_rgb, detections in vi:
    ...
```

```python
from deepsort_with_pytorch_v2.iterator_tools import FrameExtractor, VideoAssembler

paths = FrameExtractor("traffic.mp4", "runs/frames/traffic", stride=3)
VideoAssembler(sorted(paths), "runs/videos/traffic_assembled.mp4", fps=30)
```

## Training / inference / tracking

- Train init weights: `yolov8l.pt`
- Data YAML: `deepsort_with_pytorch_v2/config/detrac.yaml`
- Ultralytics train outputs typically under `runs/detect/train/weights/best.pt`
- `infer` writes annotated results under `runs/inference/`
- `track` runs YOLOv8 detections + DeepSORT and writes a tracked video

## Docker (optional)

```bash
docker build -t deepsort-with-pytorch-v2 .
docker run --rm -it -v "$PWD:/work" deepsort-with-pytorch-v2 --help
```

## Dataset

Expected structure for UA-DETRAC after conversion (see the package YAML):

```yaml
path: datasets/UA-DETRAC
train: images/train
val: images/val
names:
  0: car
  1: bus
  2: truck
```

## License

MIT
