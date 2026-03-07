from __future__ import annotations
import torch
import torch.nn as nn
from ..eval.iou import box_iou_xywh


class YOLOv1Loss(nn.Module):
    def __init__(self, S: int = 7, B: int = 2, C: int = 20,
                 lambda_coord: float = 5.0, lambda_noobj: float = 0.5) -> None:
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        N = pred.shape[0]
        obj_mask = target[..., self.C].unsqueeze(-1)
        noobj_mask = 1 - obj_mask
        pred_boxes = pred[..., self.C:self.C + 5 * self.B].reshape(N, self.S, self.S, self.B, 5)
        tgt_box = target[..., self.C + 1:self.C + 5]
        best_ious = torch.zeros(N, self.S, self.S, device=pred.device)
        best_box = torch.zeros(N, self.S, self.S, device=pred.device, dtype=torch.long)
        for b in range(self.B):
            iou = box_iou_xywh(pred_boxes[..., b, 1:5], tgt_box)
            mask = iou > best_ious
            best_ious[mask] = iou[mask]
            best_box[mask] = b
        best_mask = torch.zeros_like(pred_boxes[..., 0])
        for b in range(self.B):
            best_mask[..., b] = (best_box == b).float()
        obj = (obj_mask * best_mask).unsqueeze(-1)
        coord_loss = self.lambda_coord * (
            obj * (pred_boxes[..., 1:3] - tgt_box.unsqueeze(-2)[..., :2]).pow(2)
        ).sum()
        coord_loss += self.lambda_coord * (
            obj * (
                pred_boxes[..., 3:5].clamp(min=0).sqrt()
                - tgt_box.unsqueeze(-2)[..., 2:].clamp(min=0).sqrt()
            ).pow(2)
        ).sum()
        conf_pred = pred_boxes[..., 0]
        obj_conf_loss = (obj_mask * best_mask * (conf_pred - best_ious.unsqueeze(-1)).pow(2)).sum()
        noobj_conf_loss = self.lambda_noobj * (noobj_mask * conf_pred.pow(2)).sum()
        class_loss = (obj_mask * (pred[..., :self.C] - target[..., :self.C]).pow(2)).sum()
        return (coord_loss + obj_conf_loss + noobj_conf_loss + class_loss) / N
