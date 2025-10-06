FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /work

# System deps for OpenCV and others
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx git && \
    rm -rf /var/lib/apt/lists/*

COPY . /work

# Install package
RUN pip install --upgrade pip && pip install .

# Default command
ENTRYPOINT ["deepsort-with-pytorch-v2"]
CMD ["--help"]
