from __future__ import annotations
import torch
import torch.nn as nn


class YOLOv1Head(nn.Module):
    def __init__(self, S: int = 7, B: int = 2, C: int = 20) -> None:
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (C + 5 * B)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).reshape(-1, self.S, self.S, self.C + 5 * self.B)
