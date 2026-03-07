import torch
import pytest
from yolov1.model.yolov1 import YOLOv1
from yolov1.model.loss import YOLOv1Loss
from yolov1.eval.nms import decode_predictions
from yolov1.eval.iou import box_iou_xywh


def test_backbone_output_shape():
    model = YOLOv1()
    out = model(torch.randn(2, 3, 448, 448))
    assert out.shape == (2, 7, 7, 30)


def test_loss_forward():
    model = YOLOv1()
    criterion = YOLOv1Loss()
    pred = model(torch.randn(2, 3, 448, 448))
    target = torch.zeros(2, 7, 7, 30)
    loss = criterion(pred, target)
    assert not torch.isnan(loss)
    assert loss.item() >= 0


def test_iou_perfect():
    box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    assert abs(box_iou_xywh(box, box).item() - 1.0) < 1e-4


def test_iou_no_overlap():
    a = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    b = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    assert box_iou_xywh(a, b).item() < 1e-4


def test_nms_output():
    pred = torch.zeros(1, 7, 7, 30)
    pred[0, 3, 3, 0] = 1.0
    pred[0, 3, 3, 20] = 0.9
    pred[0, 3, 3, 21:25] = torch.tensor([0.5, 0.5, 0.3, 0.3])
    assert isinstance(decode_predictions(pred, conf_thresh=0.1), list)


def test_box_iou_xyxy_perfect():
    from yolov1.eval.iou import box_iou_xyxy
    box = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
    assert abs(box_iou_xyxy(box, box).item() - 1.0) < 1e-4


def test_box_iou_xyxy_no_overlap():
    from yolov1.eval.iou import box_iou_xyxy
    a = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
    b = torch.tensor([[0.8, 0.8, 1.0, 1.0]])
    assert box_iou_xyxy(a, b).item() < 1e-4


def test_box_iou_xyxy_partial():
    from yolov1.eval.iou import box_iou_xyxy
    a = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    b = torch.tensor([[0.25, 0.25, 0.75, 0.75]])
    iou = box_iou_xyxy(a, b).item()
    assert 0.0 < iou < 1.0
