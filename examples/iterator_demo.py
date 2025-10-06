from pathlib import Path

from deepsort_with_pytorch_v2.iterator_tools import (
    FrameIterator,
    FrameSlicer,
    FrameConcatenator,
    VideoInferenceIterator,
    FrameExtractor,
    VideoAssembler,
)


def main():
    video = "traffic.mp4"

    # Iterate frames with stride
    total = 0
    for _ in FrameIterator(video, stride=2):
        total += 1
    print(f"Iterated {total} frames (stride=2)")

    # Slice frames into chunks of 150
    slicer = FrameSlicer(FrameIterator(video), slice_length=150)
    for i, chunk in enumerate(slicer):
        print(f"Chunk {i} length: {len(chunk)}")
        if i == 1:
            break

    # Run inference streaming
    vi = VideoInferenceIterator(video, model_path="yolov8l.pt", batch_size=4)
    for idx, annotated, det in vi:
        if idx % 100 == 0:
            print(f"Frame {idx} detections: {len(getattr(det, 'boxes', []) or [])}")
        if idx > 300:
            break

    # Extract frames and re-assemble
    out_dir = Path("runs/frames/demo")
    paths = FrameExtractor(video, out_dir, stride=5)
    assembled = VideoAssembler(sorted(paths), "runs/videos/assembled_demo.mp4", fps=30)
    print(f"Assembled video: {assembled}")


if __name__ == "__main__":
    main()
