import pytest
import torch
from yolov1.engine.trainer import YOLOv1Module
from yolov1.engine import YOLOv1Module as ImportedModule


def test_trainer_instantiation():
    module = YOLOv1Module()
    assert module.model is not None
    assert module.criterion is not None


def test_trainer_hparams():
    module = YOLOv1Module(lr=1e-4, batch_size=8, S=7, B=2, C=20)
    assert module.hparams.lr == 1e-4
    assert module.hparams.batch_size == 8


def test_trainer_forward():
    module = YOLOv1Module()
    x = torch.randn(1, 3, 448, 448)
    out = module(x)
    assert out.shape == (1, 7, 7, 30)


def test_configure_optimizers():
    module = YOLOv1Module()
    optimizers, schedulers = module.configure_optimizers()
    assert len(optimizers) == 1
    assert len(schedulers) == 1
    assert isinstance(optimizers[0], torch.optim.SGD)


def test_engine_import():
    assert ImportedModule is YOLOv1Module


def test_training_step():
    import torch
    module = YOLOv1Module()
    imgs = torch.randn(2, 3, 448, 448)
    targets = torch.zeros(2, 7, 7, 30)
    loss = module.training_step((imgs, targets), 0)
    assert not torch.isnan(loss)
    assert loss.item() >= 0


def test_validation_step():
    import torch
    module = YOLOv1Module()
    imgs = torch.randn(2, 3, 448, 448)
    targets = torch.zeros(2, 7, 7, 30)
    module.validation_step((imgs, targets), 0)
