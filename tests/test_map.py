import numpy as np
import pytest
from yolov1.eval.map import compute_ap, compute_map


def test_compute_ap_perfect():
    rec = np.array([0.5, 1.0])
    prec = np.array([1.0, 1.0])
    ap = compute_ap(rec, prec)
    assert ap == pytest.approx(1.0)


def test_compute_ap_zero():
    rec = np.array([0.0])
    prec = np.array([0.0])
    ap = compute_ap(rec, prec)
    assert ap == pytest.approx(0.0)


def test_compute_map_single_class():
    all_preds = [
        [[0.1, 0.1, 0.4, 0.4, 0.9, 0]],
        [[0.5, 0.5, 0.9, 0.9, 0.8, 0]],
    ]
    all_targets = [
        {"boxes": [[0.1, 0.1, 0.4, 0.4]], "labels": [0]},
        {"boxes": [[0.5, 0.5, 0.9, 0.9]], "labels": [0]},
    ]
    result = compute_map(all_preds, all_targets)
    assert "mAP" in result
    assert 0.0 <= result["mAP"] <= 1.0


def test_compute_map_no_predictions():
    all_preds = [[], []]
    all_targets = [
        {"boxes": [[0.1, 0.1, 0.4, 0.4]], "labels": [0]},
        {"boxes": [[0.5, 0.5, 0.9, 0.9]], "labels": [1]},
    ]
    result = compute_map(all_preds, all_targets)
    assert result["mAP"] == pytest.approx(0.0)


def test_compute_map_returns_all_classes():
    all_preds = [[]]
    all_targets = [{"boxes": [], "labels": []}]
    result = compute_map(all_preds, all_targets)
    assert len(result) == 21


def test_compute_map_with_matched_detection():
    all_preds = [
        [[0.1, 0.1, 0.5, 0.5, 0.95, 0]],
        [[0.1, 0.1, 0.5, 0.5, 0.90, 1]],
    ]
    all_targets = [
        {"boxes": [[0.1, 0.1, 0.5, 0.5]], "labels": [0]},
        {"boxes": [[0.1, 0.1, 0.5, 0.5]], "labels": [1]},
    ]
    result = compute_map(all_preds, all_targets)
    assert result["aeroplane"] > 0.0
    assert result["bicycle"] > 0.0
    assert result["mAP"] > 0.0


def test_compute_map_duplicate_detection_hits_matched_branch():
    all_preds = [
        [
            [0.3, 0.3, 0.5, 0.5, 0.95, 0],
            [0.3, 0.3, 0.5, 0.5, 0.85, 0],
        ],
    ]
    all_targets = [
        {"boxes": [[0.3, 0.3, 0.5, 0.5]], "labels": [0]},
    ]
    result = compute_map(all_preds, all_targets)
    assert result["aeroplane"] > 0.0
    assert result["mAP"] >= 0.0
