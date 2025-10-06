from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

ArrayLike = np.ndarray
TensorLike = torch.Tensor
FrameType = Union[ArrayLike, TensorLike]


class FrameIterator:
    """Lazily reads frames from a video.

    - Supports stride: yield every Nth frame
    - Can return numpy arrays (default) or torch tensors
    - Streams frames without loading entire video into memory
    """

    def __init__(
        self,
        source: Union[str, Path],
        stride: int = 1,
        return_tensors: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.source = str(source)
        self.stride = max(1, int(stride))
        self.return_tensors = return_tensors
        self.device = torch.device(device) if device is not None else None

    def __iter__(self) -> Iterator[FrameType]:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.source}")
        try:
            idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if idx % self.stride != 0:
                    idx += 1
                    continue
                # BGR -> RGB for ML-friendly convention
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if self.return_tensors:
                    t = torch.from_numpy(frame).to(torch.uint8)
                    if self.device is not None:
                        t = t.to(self.device)
                    yield t
                else:
                    yield frame
                idx += 1
        finally:
            cap.release()


class FrameSlicer:
    """Yield fixed-length chunks (slices) from a frame iterable.

    This is a streaming chunker. It does not store all frames.
    """

    def __init__(self, frames: Iterable[FrameType], slice_length: int) -> None:
        self.frames = frames
        self.slice_length = int(slice_length)
        if self.slice_length <= 0:
            raise ValueError("slice_length must be > 0")

    def __iter__(self) -> Iterator[List[FrameType]]:
        buf: List[FrameType] = []
        for frame in self.frames:
            buf.append(frame)
            if len(buf) == self.slice_length:
                yield buf
                buf = []
        if buf:
            yield buf


class FrameConcatenator:
    """Concatenate multiple frame iterables into one sequence.

    Optionally write the concatenated output to a video file (BGR).
    Input frames are expected in RGB; will convert to BGR for writer.
    """

    def __init__(
        self,
        sequences: Sequence[Iterable[FrameType]],
        output_path: Optional[Union[str, Path]] = None,
        fps: Optional[float] = None,
        codec: str = "mp4v",
    ) -> None:
        self.sequences = sequences
        self.output_path = Path(output_path) if output_path else None
        self.fps = fps
        self.codec = codec

    def __iter__(self) -> Iterator[FrameType]:
        writer = None
        try:
            for seq in self.sequences:
                for frame in seq:
                    if isinstance(frame, torch.Tensor):
                        frame_np = frame.detach().cpu().numpy()
                    else:
                        frame_np = frame
                    if self.output_path and writer is None:
                        h, w = frame_np.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*self.codec)
                        writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps or 30.0, (w, h))
                    if writer is not None:
                        bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                        writer.write(bgr)
                    yield frame
        finally:
            if writer is not None:
                writer.release()


class VideoInferenceIterator:
    """High-level streaming iterator: frames -> YOLOv8 inference -> annotated frames.

    Yields tuples of (frame_index, annotated_frame_RGB, detections), one by one.
    """

    def __init__(
        self,
        source: Union[str, Path],
        model_path: Union[str, Path],
        batch_size: int = 1,
        conf: float = 0.25,
        imgsz: int = 1280,
        device: Optional[Union[str, torch.device]] = None,
        stride: int = 1,
    ) -> None:
        from ultralytics import YOLO  # local import to avoid heavy import unless needed

        self.source = str(source)
        self.model_path = str(model_path)
        self.batch_size = max(1, int(batch_size))
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device
        self.stride = max(1, int(stride))
        self.model = YOLO(self.model_path)

    def __iter__(self) -> Iterator[Tuple[int, ArrayLike, object]]:
        frame_iter = FrameIterator(self.source, stride=self.stride, return_tensors=False)
        batch: List[ArrayLike] = []
        frame_indices: List[int] = []
        frame_idx_global = 0

        def flush_batch() -> List[Tuple[int, ArrayLike, object]]:
            if not batch:
                return []
            results = self.model.predict(batch, imgsz=self.imgsz, conf=self.conf, verbose=False)
            out: List[Tuple[int, ArrayLike, object]] = []
            for (idx, frame_rgb), res in zip(frame_indices, results):
                annotated = res.plot()[:, :, ::-1]  # res.plot() returns BGR; convert to RGB
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                out.append((idx, annotated_rgb, res))
            batch.clear()
            frame_indices.clear()
            return out

        for frame in frame_iter:
            batch.append(frame)
            frame_indices.append(frame_idx_global)
            frame_idx_global += 1
            if len(batch) >= self.batch_size:
                for item in flush_batch():
                    yield item
        for item in flush_batch():
            yield item


def FrameExtractor(source: Union[str, Path], target_dir: Union[str, Path], stride: int = 1) -> List[Path]:
    """Extract all frames to target directory. Returns list of saved file paths.

    Frames are written as PNG files with zero-padded indices.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    idx = 0
    for i, frame in enumerate(FrameIterator(source, stride=stride, return_tensors=False)):
        out = target / f"frame_{i:06d}.png"
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out), bgr)
        paths.append(out)
        idx += 1
    return paths


def VideoAssembler(
    frame_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    fps: float = 30.0,
    resolution: Optional[Tuple[int, int]] = None,
    codec: str = "mp4v",
) -> Path:
    """Assemble frames into a video file. Frame paths should be sorted.

    If resolution is None, it is inferred from the first frame.
    """
    frames = [Path(p) for p in frame_paths]
    if not frames:
        raise ValueError("No frame paths provided")
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise FileNotFoundError(f"Could not read first frame: {frames[0]}")
    h0, w0 = first.shape[:2]
    w, h = (w0, h0) if resolution is None else resolution

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    try:
        for p in frames:
            img = cv2.imread(str(p))
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != (w, h):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()
    return out_path
