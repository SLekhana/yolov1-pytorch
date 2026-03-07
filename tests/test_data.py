import numpy as np
from yolov1.data.augmentations import hsv_jitter, random_crop, TrainTransform


def test_hsv_jitter_shape():
    img = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
    assert hsv_jitter(img).shape == img.shape


def test_random_crop_shape():
    img = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
    boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
    out_img, _ = random_crop(img, boxes)
    assert out_img.shape == img.shape


def test_train_transform():
    img = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
    boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
    out_img, _, _ = TrainTransform()(img, boxes, np.array([0]))
    assert out_img.shape == img.shape
