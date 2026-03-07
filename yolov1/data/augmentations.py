from __future__ import annotations
import random
import cv2
import numpy as np


def hsv_jitter(img, h=0.1, s=0.7, v=0.4):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-h, h) * 180) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(1 - s, 1 + s), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(1 - v, 1 + v), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_crop(img, boxes, min_scale=0.8):
    h, w = img.shape[:2]
    scale = random.uniform(min_scale, 1.0)
    new_h, new_w = int(h * scale), int(w * scale)
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    img = img[top:top + new_h, left:left + new_w]
    img = cv2.resize(img, (w, h))
    if len(boxes):
        boxes[:, 0] = np.clip((boxes[:, 0] * w - left) / new_w, 0, 1)
        boxes[:, 1] = np.clip((boxes[:, 1] * h - top) / new_h, 0, 1)
        boxes[:, 2] = boxes[:, 2] * w / new_w
        boxes[:, 3] = boxes[:, 3] * h / new_h
    return img, boxes


def mosaic(imgs, boxes_list, labels_list, size=448):
    h, w = size // 2, size // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    out_boxes, out_labels = [], []
    positions = [(0, 0), (0, w), (h, 0), (h, w)]
    for k, (img, boxes, labels) in enumerate(zip(imgs[:4], boxes_list[:4], labels_list[:4])):
        img_r = cv2.resize(img, (w, h))
        r, c = positions[k]
        canvas[r:r + h, c:c + w] = img_r
        if len(boxes):
            b = boxes.copy()
            b[:, 0] = (b[:, 0] * w + c) / size
            b[:, 1] = (b[:, 1] * h + r) / size
            b[:, 2] = b[:, 2] * w / size
            b[:, 3] = b[:, 3] * h / size
            out_boxes.append(b)
            out_labels.append(labels)
    if out_boxes:
        return canvas, np.concatenate(out_boxes), np.concatenate(out_labels)
    return canvas, np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)


class TrainTransform:
    def __call__(self, img, boxes, labels):
        img = hsv_jitter(img)
        img, boxes = random_crop(img, boxes)
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            if len(boxes):
                boxes[:, 0] = 1 - boxes[:, 0]
        return img, boxes, labels
