from __future__ import annotations
import torch
import torch.nn as nn
from .backbone import YOLOv1Backbone
from .head import YOLOv1Head


class YOLOv1(nn.Module):
    def __init__(self, S: int = 7, B: int = 2, C: int = 20) -> None:
        super().__init__()
        self.backbone = YOLOv1Backbone()
        self.head = YOLOv1Head(S, B, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
