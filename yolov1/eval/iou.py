from __future__ import annotations
import torch


def box_iou_xywh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p_x1 = pred[..., 0] - pred[..., 2] / 2
    p_y1 = pred[..., 1] - pred[..., 3] / 2
    p_x2 = pred[..., 0] + pred[..., 2] / 2
    p_y2 = pred[..., 1] + pred[..., 3] / 2
    t_x1 = target[..., 0] - target[..., 2] / 2
    t_y1 = target[..., 1] - target[..., 3] / 2
    t_x2 = target[..., 0] + target[..., 2] / 2
    t_y2 = target[..., 1] + target[..., 3] / 2
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = (p_x2 - p_x1) * (p_y2 - p_y1) + (t_x2 - t_x1) * (t_y2 - t_y1) - inter
    return inter / (union + 1e-6)


def box_iou_xyxy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.max(pred[..., 0], target[..., 0])
    inter_y1 = torch.max(pred[..., 1], target[..., 1])
    inter_x2 = torch.min(pred[..., 2], target[..., 2])
    inter_y2 = torch.min(pred[..., 3], target[..., 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area_p = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    area_t = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
    return inter / (area_p + area_t - inter + 1e-6)
