import numpy as np
import pytest
import torch
import xml.etree.ElementTree as ET
import cv2
from yolov1.engine.trainer import YOLOv1Module


@pytest.fixture
def fake_voc_root(tmp_path):
    voc_dir = tmp_path / "VOCdevkit" / "VOC2007"
    (voc_dir / "ImageSets" / "Main").mkdir(parents=True)
    (voc_dir / "JPEGImages").mkdir(parents=True)
    (voc_dir / "Annotations").mkdir(parents=True)

    ids = ["000001", "000002", "000003", "000004"]
    (voc_dir / "ImageSets" / "Main" / "trainval.txt").write_text("\n".join(ids))
    (voc_dir / "ImageSets" / "Main" / "test.txt").write_text("\n".join(ids))

    for img_id in ids:
        img = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
        cv2.imwrite(str(voc_dir / "JPEGImages" / f"{img_id}.jpg"), img)

        root = ET.Element("annotation")
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "448"
        ET.SubElement(size, "height").text = "448"
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "cat"
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = "100"
        ET.SubElement(bb, "ymin").text = "100"
        ET.SubElement(bb, "xmax").text = "200"
        ET.SubElement(bb, "ymax").text = "200"
        (voc_dir / "Annotations" / f"{img_id}.xml").write_bytes(ET.tostring(root))

    return str(tmp_path)


def test_train_dataloader(fake_voc_root):
    module = YOLOv1Module(data_root=fake_voc_root, batch_size=2, num_workers=0)
    dl = module.train_dataloader()
    imgs, targets = next(iter(dl))
    assert imgs.shape == (2, 3, 448, 448)
    assert targets.shape == (2, 7, 7, 30)


def test_val_dataloader(fake_voc_root):
    module = YOLOv1Module(data_root=fake_voc_root, batch_size=2, num_workers=0)
    dl = module.val_dataloader()
    imgs, targets = next(iter(dl))
    assert imgs.shape == (2, 3, 448, 448)
    assert targets.shape == (2, 7, 7, 30)
