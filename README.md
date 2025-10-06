# deepsort-with-pytorch-v2

YOLOv8 + DeepSORT (PyTorch) package for vehicle detection and tracking (cars, buses, trucks).

Note: Model fine-tuned on UA-DETRAC, derived from YOLOv8 COCO weights.

## Quickstart (Conda, Python 3.11, CPU)

```bash
# 1) Ensure conda is available in your shell
# For Anaconda:   source ~/anaconda3/etc/profile.d/conda.sh
# For Miniconda:  source ~/miniconda3/etc/profile.d/conda.sh
# For Miniforge:  source ~/miniforge3/etc/profile.d/conda.sh

# 2) Create env
conda create -y -n dspv2 python=3.11
conda activate dspv2

# 3) Install PyTorch CPU (use CUDA variant if you have GPU)
conda install -y -c pytorch -c conda-forge pytorch torchvision cpuonly

# 4) Install package deps and the package
pip install -U pip
pip install ultralytics deep-sort-realtime opencv-python PyYAML typer
pip install -e .

# 5) Run
deepsort-with-pytorch-v2 --help
```

## CLI

```bash
deepsort-with-pytorch-v2 train --data config/detrac.yaml --epochs 50 --imgsz 1280
deepsort-with-pytorch-v2 infer --source path/to/video_or_image
deepsort-with-pytorch-v2 track --source path/to/video.mp4
deepsort-with-pytorch-v2 iterate --source path/to/video.mp4 --stride 3 --mode extract
```

## Iterating and Processing Videos

The `iterator_tools` module streams frames efficiently for long videos.

- FrameIterator: lazily read frames with stride
```python
from deepsort_with_pytorch_v2.iterator_tools import FrameIterator

for frame in FrameIterator("traffic.mp4", stride=2):
    # frame is RGB numpy array
    ...
```

- FrameSlicer: break frames into fixed-length chunks
```python
from deepsort_with_pytorch_v2.iterator_tools import FrameIterator, FrameSlicer

slicer = FrameSlicer(FrameIterator("traffic.mp4"), slice_length=150)
for chunk in slicer:  # list of 150 frames
    ...
```

- VideoInferenceIterator: batched YOLOv8 inference over a video
```python
from deepsort_with_pytorch_v2.iterator_tools import VideoInferenceIterator

vi = VideoInferenceIterator("traffic.mp4", model_path="yolov8l.pt", batch_size=4, conf=0.25, imgsz=1280)
for frame_idx, annotated_rgb, detections in vi:
    ...
```

- FrameExtractor / VideoAssembler
```python
from deepsort_with_pytorch_v2.iterator_tools import FrameExtractor, VideoAssembler

paths = FrameExtractor("traffic.mp4", "runs/frames/traffic", stride=3)
VideoAssembler(sorted(paths), "runs/videos/traffic_assembled.mp4", fps=30)
```

Design philosophy: streaming, composability, and reuse across modules.

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

## Docker (optional)

```bash
docker build -t deepsort-with-pytorch-v2 .
docker run --rm -it -v "$PWD:/work" deepsort-with-pytorch-v2 --help
```

## Dataset

Expected structure for UA-DETRAC after conversion:

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
