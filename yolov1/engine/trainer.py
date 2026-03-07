from __future__ import annotations
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from ..model.yolov1 import YOLOv1
from ..model.loss import YOLOv1Loss
from ..data.voc_dataset import VOCDataset
from ..data.augmentations import TrainTransform


class YOLOv1Module(pl.LightningModule):
    def __init__(self, S=7, B=2, C=20, lr=1e-3, weight_decay=5e-4,
                 data_root="data/VOC", batch_size=16, num_workers=4):
        super().__init__()
        self.save_hyperparameters()
        self.model = YOLOv1(S, B, C)
        self.criterion = YOLOv1Loss(S, B, C)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        loss = self.criterion(self(imgs), targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        loss = self.criterion(self(imgs), targets)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr,
            momentum=0.9, weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 105], gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        ds = VOCDataset(self.hparams.data_root, year="2007", split="trainval", transform=TrainTransform())
        num_workers = self.hparams.num_workers
        kwargs = dict(prefetch_factor=2, persistent_workers=True) if num_workers > 0 else {}
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True,
                         num_workers=num_workers, pin_memory=True, **kwargs)

    def val_dataloader(self):
        ds = VOCDataset(self.hparams.data_root, year="2007", split="test")
        return DataLoader(ds, batch_size=self.hparams.batch_size,
                         num_workers=self.hparams.num_workers, pin_memory=True)
