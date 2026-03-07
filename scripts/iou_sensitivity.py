from __future__ import annotations
import json
import torch
import typer
from pathlib import Path
from yolov1.model.yolov1 import YOLOv1
from yolov1.eval.nms import decode_predictions
from yolov1.eval.map import compute_map
from yolov1.data.voc_dataset import VOCDataset
from torch.utils.data import DataLoader

app = typer.Typer()


@app.command()
def sensitivity(
    checkpoint: str = typer.Argument(...),
    data_root: str = "data/VOC",
    max_samples: int = 200,
    output: str = "logs/iou_sensitivity.json",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ds = VOCDataset(data_root, year="2007", split="test")
    dl = DataLoader(ds, batch_size=1, num_workers=0)

    all_preds: list[list] = []
    all_targets: list[dict] = []

    for idx, (imgs, targets) in enumerate(dl):
        if idx >= max_samples:
            break
        imgs = imgs.to(device)
        with torch.no_grad():
            preds = model(imgs)
        dets = decode_predictions(preds, conf_thresh=0.2, iou_thresh=0.5)[0]
        all_preds.append(dets)

        gt_boxes: list[list[float]] = []
        gt_labels: list[int] = []
        target = targets[0]
        for i in range(7):
            for j in range(7):
                if target[i, j, 20] == 1:
                    cx = (target[i, j, 21].item() + j) / 7
                    cy = (target[i, j, 22].item() + i) / 7
                    bw = target[i, j, 23].item()
                    bh = target[i, j, 24].item()
                    cls = int(target[i, j, :20].argmax().item())
                    gt_boxes.append([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
                    gt_labels.append(cls)
        all_targets.append({"boxes": gt_boxes, "labels": gt_labels})

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75]
    results: dict[str, float] = {}
    for thresh in thresholds:
        metrics = compute_map(all_preds, all_targets, iou_thresh=thresh)
        results[str(thresh)] = round(metrics["mAP"], 4)
        typer.echo(f"IoU@{thresh:.2f} → mAP: {metrics['mAP']:.4f}")

    Path(output).parent.mkdir(exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2))
    typer.echo(f"\nIoU sensitivity saved to {output}")


if __name__ == "__main__":
    app()
