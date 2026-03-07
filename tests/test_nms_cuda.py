import torch
import pytest
from unittest.mock import patch, MagicMock
from yolov1.eval.nms_cuda import _nms_cpu, nms_dispatch
import yolov1.eval.nms_cuda as nms_mod


def make_boxes():
    return torch.tensor([
        [0.1, 0.1, 0.5, 0.5],
        [0.1, 0.1, 0.5, 0.5],
        [0.8, 0.8, 0.9, 0.9],
    ])


def make_scores():
    return torch.tensor([0.9, 0.8, 0.7])


def test_nms_cpu_removes_duplicate():
    keep = _nms_cpu(make_boxes(), make_scores(), iou_thresh=0.5)
    assert 0 in keep.tolist()
    assert 1 not in keep.tolist()


def test_nms_cpu_keeps_separate_boxes():
    keep = _nms_cpu(make_boxes(), make_scores(), iou_thresh=0.5)
    assert 2 in keep.tolist()


def test_nms_cpu_single_box():
    boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
    scores = torch.tensor([0.9])
    keep = _nms_cpu(boxes, scores, iou_thresh=0.5)
    assert keep.tolist() == [0]


def test_nms_dispatch_cpu():
    keep = nms_dispatch(make_boxes(), make_scores(), iou_thresh=0.5)
    assert isinstance(keep, torch.Tensor)
    assert len(keep) >= 1


def test_nms_dispatch_high_thresh_keeps_all():
    keep = nms_dispatch(make_boxes(), make_scores(), iou_thresh=1.0)
    assert len(keep) == 3


def test_nms_dispatch_cuda_branch():
    expected = torch.tensor([0, 2])
    boxes_mock = MagicMock()
    boxes_mock.is_cuda = True
    scores = make_scores()
    with patch.object(nms_mod, "torchvision_nms", return_value=expected) as mock_tv:
        result = nms_mod.nms_dispatch(boxes_mock, scores, 0.5)
        mock_tv.assert_called_once_with(boxes_mock, scores, 0.5)
        assert torch.equal(result, expected)
