from __future__ import annotations
import time
import torch
import typer
from yolov1.model.yolov1 import YOLOv1

app = typer.Typer()


@app.command()
def benchmark(
    checkpoint: str = typer.Option("", help="Optional checkpoint path"),
    batch_size: int = 1,
    warmup: int = 20,
    runs: int = 200,
    img_size: int = 448,
    device: str = "cpu",
) -> None:
    dev = torch.device(device)
    model = YOLOv1().to(dev)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=dev))
    model.eval()

    dummy = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_images = runs * batch_size
    fps = total_images / elapsed
    latency_ms = (elapsed / runs) * 1000

    typer.echo(f"Device       : {device}")
    typer.echo(f"Batch size   : {batch_size}")
    typer.echo(f"Runs         : {runs}")
    typer.echo(f"Total images : {total_images}")
    typer.echo(f"Elapsed      : {elapsed:.2f}s")
    typer.echo(f"FPS          : {fps:.1f}")
    typer.echo(f"Latency/batch: {latency_ms:.2f}ms")


if __name__ == "__main__":
    app()
