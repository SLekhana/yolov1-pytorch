import numpy as np
import pytest
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch, MagicMock
from yolov1.data.voc_dataset import VOCDataset, VOC_CLASSES, CLASS_TO_IDX


def make_fake_annotation(img_id, width=448, height=448):
    root = ET.Element("annotation")
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = "cat"
    ET.SubElement(obj, "difficult").text = "0"
    bb = ET.SubElement(obj, "bndbox")
    ET.SubElement(bb, "xmin").text = "100"
    ET.SubElement(bb, "ymin").text = "100"
    ET.SubElement(bb, "xmax").text = "200"
    ET.SubElement(bb, "ymax").text = "200"
    return ET.tostring(root)


def make_fake_difficult_annotation():
    root = ET.Element("annotation")
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "448"
    ET.SubElement(size, "height").text = "448"
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = "dog"
    ET.SubElement(obj, "difficult").text = "1"
    bb = ET.SubElement(obj, "bndbox")
    ET.SubElement(bb, "xmin").text = "50"
    ET.SubElement(bb, "ymin").text = "50"
    ET.SubElement(bb, "xmax").text = "100"
    ET.SubElement(bb, "ymax").text = "100"
    return ET.tostring(root)


@pytest.fixture
def fake_voc_root(tmp_path):
    voc_dir = tmp_path / "VOCdevkit" / "VOC2007"
    (voc_dir / "ImageSets" / "Main").mkdir(parents=True)
    (voc_dir / "JPEGImages").mkdir(parents=True)
    (voc_dir / "Annotations").mkdir(parents=True)

    ids = ["000001", "000002"]
    (voc_dir / "ImageSets" / "Main" / "trainval.txt").write_text("\n".join(ids))

    import cv2
    for img_id in ids:
        img = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
        cv2.imwrite(str(voc_dir / "JPEGImages" / f"{img_id}.jpg"), img)
        (voc_dir / "Annotations" / f"{img_id}.xml").write_bytes(
            make_fake_annotation(img_id)
        )

    return str(tmp_path)


@pytest.fixture
def fake_voc_difficult(tmp_path):
    voc_dir = tmp_path / "VOCdevkit" / "VOC2007"
    (voc_dir / "ImageSets" / "Main").mkdir(parents=True)
    (voc_dir / "JPEGImages").mkdir(parents=True)
    (voc_dir / "Annotations").mkdir(parents=True)

    (voc_dir / "ImageSets" / "Main" / "trainval.txt").write_text("000001")

    import cv2
    img = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
    cv2.imwrite(str(voc_dir / "JPEGImages" / "000001.jpg"), img)
    (voc_dir / "Annotations" / "000001.xml").write_bytes(
        make_fake_difficult_annotation()
    )

    return str(tmp_path)


def test_dataset_len(fake_voc_root):
    ds = VOCDataset(fake_voc_root, year="2007", split="trainval")
    assert len(ds) == 2


def test_dataset_getitem_shapes(fake_voc_root):
    ds = VOCDataset(fake_voc_root, year="2007", split="trainval")
    img, target = ds[0]
    assert img.shape == (3, 448, 448)
    assert target.shape == (7, 7, 30)


def test_dataset_image_normalized(fake_voc_root):
    ds = VOCDataset(fake_voc_root, year="2007", split="trainval")
    img, _ = ds[0]
    assert img.max() <= 1.0
    assert img.min() >= 0.0


def test_dataset_target_encoding(fake_voc_root):
    ds = VOCDataset(fake_voc_root, year="2007", split="trainval")
    _, target = ds[0]
    assert target.sum() > 0


def test_dataset_difficult_skipped(fake_voc_difficult):
    ds = VOCDataset(fake_voc_difficult, year="2007", split="trainval")
    _, target = ds[0]
    assert target[..., 20].sum() == 0


def test_dataset_with_transform(fake_voc_root):
    from yolov1.data.augmentations import TrainTransform
    ds = VOCDataset(fake_voc_root, year="2007", split="trainval", transform=TrainTransform())
    img, target = ds[0]
    assert img.shape == (3, 448, 448)


def test_voc_classes_length():
    assert len(VOC_CLASSES) == 20


def test_class_to_idx_mapping():
    assert CLASS_TO_IDX["cat"] == 7
    assert CLASS_TO_IDX["person"] == 14


def test_encode_boundary_box(fake_voc_root):
    ds = VOCDataset(fake_voc_root, year="2007", split="trainval")
    boxes = np.array([[0.999, 0.999, 0.1, 0.1]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    target = ds._encode(boxes, labels)
    assert target.shape == (7, 7, 30)
