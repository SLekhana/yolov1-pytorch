import numpy as np
import pytest
from yolov1.data.augmentations import hsv_jitter, random_crop, mosaic, TrainTransform


def make_img():
    return np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)


def test_hsv_jitter_values_change():
    img = make_img()
    out = hsv_jitter(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_random_crop_no_boxes():
    img = make_img()
    out_img, out_boxes = random_crop(img, np.zeros((0, 4), dtype=np.float32))
    assert out_img.shape == img.shape
    assert len(out_boxes) == 0


def test_random_crop_boxes_clipped():
    img = make_img()
    boxes = np.array([[0.5, 0.5, 0.3, 0.3]], dtype=np.float32)
    out_img, out_boxes = random_crop(img, boxes)
    assert out_boxes[:, 0].max() <= 1.0
    assert out_boxes[:, 0].min() >= 0.0


def test_mosaic_output_shape():
    imgs = [make_img() for _ in range(4)]
    boxes_list = [np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32) for _ in range(4)]
    labels_list = [np.array([0]) for _ in range(4)]
    out_img, out_boxes, out_labels = mosaic(imgs, boxes_list, labels_list)
    assert out_img.shape == (448, 448, 3)
    assert len(out_boxes) == 4


def test_mosaic_empty_boxes():
    imgs = [make_img() for _ in range(4)]
    boxes_list = [np.zeros((0, 4), dtype=np.float32) for _ in range(4)]
    labels_list = [np.zeros(0, dtype=np.int64) for _ in range(4)]
    out_img, out_boxes, out_labels = mosaic(imgs, boxes_list, labels_list)
    assert out_img.shape == (448, 448, 3)
    assert len(out_boxes) == 0


def test_train_transform_flip():
    img = make_img()
    boxes = np.array([[0.3, 0.5, 0.2, 0.2]], dtype=np.float32)
    labels = np.array([1])
    tf = TrainTransform()
    for _ in range(20):
        out_img, out_boxes, out_labels = tf(img, boxes, labels)
        assert out_img.shape == img.shape
        assert out_boxes.shape == boxes.shape
