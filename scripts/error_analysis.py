from __future__ import annotations
import json
import torch
import typer
from collections import defaultdict
from pathlib import Path
from yolov1.model.yolov1 import YOLOv1
from yolov1.eval.nms import decode_predictions
from yolov1.eval.iou import box_iou_xywh
from yolov1.data.voc_dataset import VOCDataset, VOC_CLASSES
from torch.utils.data import DataLoader

app = typer.Typer()


@app.command()
def analyze(
    checkpoint: str = typer.Argument(...),
    data_root: str = "data/VOC",
    conf_thresh: float = 0.3,
    iou_thresh: float = 0.5,
    max_samples: int = 200,
    output: str = "logs/error_analysis.json",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ds = VOCDataset(data_root, year="2007", split="test")
    dl = DataLoader(ds, batch_size=1, num_workers=0)

    per_class_tp: dict[int, int] = defaultdict(int)
    per_class_fp: dict[int, int] = defaultdict(int)
    per_class_fn: dict[int, int] = defaultdict(int)
    per_class_loc_errors: dict[int, list[float]] = defaultdict(list)

    for idx, (imgs, targets) in enumerate(dl):
        if idx >= max_samples:
            break
        imgs = imgs.to(device)
        with torch.no_grad():
            preds = model(imgs)
        dets = decode_predictions(preds, conf_thresh=conf_thresh, iou_thresh=iou_thresh)[0]

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

        matched_gt: set[int] = set()
        for det in dets:
            x1, y1, x2, y2, conf, cls = det
            best_iou, best_j = 0.0, -1
            for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if gl != int(cls) or j in matched_gt:
                    continue
                iou = box_iou_xywh(
                    torch.tensor([[x1, y1, x2 - x1, y2 - y1]]),
                    torch.tensor([[gb[0], gb[1], gb[2] - gb[0], gb[3] - gb[1]]]),
                ).item()
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                per_class_tp[int(cls)] += 1
                per_class_loc_errors[int(cls)].append(1.0 - best_iou)
                matched_gt.add(best_j)
            else:
                per_class_fp[int(cls)] += 1

        for j, gl in enumerate(gt_labels):
            if j not in matched_gt:
                per_class_fn[gl] += 1

    results: dict[str, dict[str, float]] = {}
    for c in range(20):
        tp = per_class_tp[c]
        fp = per_class_fp[c]
        fn = per_class_fn[c]
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        avg_loc = float(sum(per_class_loc_errors[c]) / len(per_class_loc_errors[c])) if per_class_loc_errors[c] else 0.0
        results[VOC_CLASSES[c]] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "avg_loc_error": round(avg_loc, 4),
        }

    Path(output).parent.mkdir(exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2))
    typer.echo(f"\nPer-class error analysis saved to {output}")
    typer.echo(f"\n{'Class':<20} {'Precision':>10} {'Recall':>10} {'AvgLocErr':>12}")
    typer.echo("-" * 55)
    for cls_name, stats in results.items():
        typer.echo(f"{cls_name:<20} {stats['precision']:>10.4f} {stats['recall']:>10.4f} {stats['avg_loc_error']:>12.4f}")


if __name__ == "__main__":
    app()
