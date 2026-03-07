from __future__ import annotations
import torch
from torchvision.ops import nms as torchvision_nms


def _nms_cpu(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        x1 = torch.max(boxes[i, 0], boxes[rest, 0])
        y1 = torch.max(boxes[i, 1], boxes[rest, 1])
        x2 = torch.min(boxes[i, 2], boxes[rest, 2])
        y2 = torch.min(boxes[i, 3], boxes[rest, 3])
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long)


def nms_dispatch(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
    if boxes.is_cuda:
        return torchvision_nms(boxes, scores, iou_thresh)
    return _nms_cpu(boxes, scores, iou_thresh)
