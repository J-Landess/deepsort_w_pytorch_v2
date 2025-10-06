import typer
from .train import main as train_main
from .infer import main as infer_main
from .track import main as track_main

app = typer.Typer(help="YOLOv8 + DeepSORT (PyTorch) for vehicle detection and tracking")

@app.command()
def train(
    data: str = typer.Option("config/detrac.yaml", help="Dataset YAML"),
    epochs: int = typer.Option(50, help="Training epochs"),
    imgsz: int = typer.Option(1280, help="Image size"),
    batch: int = typer.Option(8, help="Batch size"),
    device: str = typer.Option("", help="CUDA device id(s), e.g. '0' or '0,1'"),
):
    train_main(data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device)

@app.command()
def infer(
    source: str = typer.Option(..., help="Image or video file path"),
    weights: str = typer.Option("runs/detect/train/weights/best.pt", help="Weights path"),
    imgsz: int = typer.Option(1280, help="Image size"),
    device: str = typer.Option("", help="CUDA device id(s)"),
):
    infer_main(source=source, weights=weights, imgsz=imgsz, device=device)

@app.command()
def track(
    source: str = typer.Option(..., help="Video file path"),
    weights: str = typer.Option("runs/detect/train/weights/best.pt", help="Weights path"),
    device: str = typer.Option("", help="CUDA device id(s)"),
    conf: float = typer.Option(0.25, help="Confidence threshold"),
):
    track_main(source=source, weights=weights, device=device, conf=conf)

if __name__ == "__main__":
    app()
