import torch
import pytest
from yolov1.eval.nms import decode_predictions


def test_no_detections_above_thresh():
    pred = torch.zeros(1, 7, 7, 30)
    dets = decode_predictions(pred, conf_thresh=0.99)
    assert dets == [[]]


def test_single_detection():
    pred = torch.zeros(1, 7, 7, 30)
    pred[0, 3, 3, 0] = 1.0
    pred[0, 3, 3, 20] = 0.9
    pred[0, 3, 3, 21] = 0.5
    pred[0, 3, 3, 22] = 0.5
    pred[0, 3, 3, 23] = 0.3
    pred[0, 3, 3, 24] = 0.3
    dets = decode_predictions(pred, conf_thresh=0.1)
    assert len(dets[0]) >= 1


def test_batch_decode():
    pred = torch.zeros(3, 7, 7, 30)
    dets = decode_predictions(pred, conf_thresh=0.5)
    assert len(dets) == 3


def test_detection_coords_valid():
    pred = torch.zeros(1, 7, 7, 30)
    pred[0, 0, 0, 0] = 1.0
    pred[0, 0, 0, 20] = 1.0
    pred[0, 0, 0, 21] = 0.5
    pred[0, 0, 0, 22] = 0.5
    pred[0, 0, 0, 23] = 0.2
    pred[0, 0, 0, 24] = 0.2
    dets = decode_predictions(pred, conf_thresh=0.1)
    for det in dets[0]:
        x1, y1, x2, y2, conf, cls = det
        assert 0 <= x1 <= 1
        assert 0 <= y1 <= 1
        assert 0 <= x2 <= 1
        assert 0 <= y2 <= 1
