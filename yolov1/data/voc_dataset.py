from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
    "chair","cow","diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}


class VOCDataset(Dataset):
    def __init__(self, root, year="2007", split="trainval", S=7, B=2, C=20, img_size=448, transform=None):
        self.root = Path(root) / f"VOCdevkit/VOC{year}"
        self.S = S
        self.B = B
        self.C = C
        self.img_size = img_size
        self.transform = transform
        split_file = self.root / "ImageSets" / "Main" / f"{split}.txt"
        self.ids = split_file.read_text().strip().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = self._load_image(img_id)
        boxes, labels = self._load_annotation(img_id)
        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)
        target = self._encode(boxes, labels)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_tensor, target

    def _load_image(self, img_id):
        path = self.root / "JPEGImages" / f"{img_id}.jpg"
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (self.img_size, self.img_size))

    def _load_annotation(self, img_id):
        path = self.root / "Annotations" / f"{img_id}.xml"
        tree = ET.parse(path)
        root = tree.getroot()
        size = root.find("size")
        w = float(size.find("width").text)
        h = float(size.find("height").text)
        boxes, labels = [], []
        for obj in root.findall("object"):
            if obj.find("difficult").text == "1":
                continue
            cls = CLASS_TO_IDX[obj.find("name").text]
            bb = obj.find("bndbox")
            x1 = float(bb.find("xmin").text) / w
            y1 = float(bb.find("ymin").text) / h
            x2 = float(bb.find("xmax").text) / w
            y2 = float(bb.find("ymax").text) / h
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
            boxes.append([cx, cy, bw, bh])
            labels.append(cls)
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _encode(self, boxes, labels):
        target = torch.zeros(self.S, self.S, self.C + 5 * self.B)
        for box, label in zip(boxes, labels):
            cx, cy, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            i = min(int(cy * self.S), self.S - 1)
            j = min(int(cx * self.S), self.S - 1)
            if target[i, j, self.C] == 0:
                target[i, j, self.C] = 1
                target[i, j, self.C + 1] = cx * self.S - j
                target[i, j, self.C + 2] = cy * self.S - i
                target[i, j, self.C + 3] = bw
                target[i, j, self.C + 4] = bh
                target[i, j, int(label)] = 1
        return target
