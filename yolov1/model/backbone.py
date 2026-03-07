from __future__ import annotations
import torch
import torch.nn as nn


def conv_bn_relu(in_c: int, out_c: int, k: int, s: int = 1, p: int = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1, inplace=True),
    )


class YOLOv1Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            conv_bn_relu(3, 64, 7, 2, 3),
            nn.MaxPool2d(2, 2),
            conv_bn_relu(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            conv_bn_relu(192, 128, 1),
            conv_bn_relu(128, 256, 3, 1, 1),
            conv_bn_relu(256, 256, 1),
            conv_bn_relu(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            *[nn.Sequential(conv_bn_relu(512, 256, 1), conv_bn_relu(256, 512, 3, 1, 1)) for _ in range(4)],
            conv_bn_relu(512, 512, 1),
            conv_bn_relu(512, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            conv_bn_relu(1024, 512, 1),
            conv_bn_relu(512, 1024, 3, 1, 1),
            conv_bn_relu(1024, 512, 1),
            conv_bn_relu(512, 1024, 3, 1, 1),
            conv_bn_relu(1024, 1024, 3, 1, 1),
            conv_bn_relu(1024, 1024, 3, 2, 1),
            conv_bn_relu(1024, 1024, 3, 1, 1),
            conv_bn_relu(1024, 1024, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
