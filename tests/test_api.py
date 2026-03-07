import io
import numpy as np
import cv2
import torch
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from yolov1.serve.api import app, load_model
import yolov1.serve.api as api_module


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, 7, 7, 30)
    api_module._model = mock_model
    return TestClient(app)


def make_image_bytes():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_detect_endpoint(client):
    img_bytes = make_image_bytes()
    response = client.post("/detect", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "latency_ms" in data


def test_detect_returns_list(client):
    img_bytes = make_image_bytes()
    response = client.post("/detect", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert isinstance(response.json()["detections"], list)


def test_load_model_sets_global(tmp_path):
    from yolov1.model.yolov1 import YOLOv1
    import yolov1.serve.api as api_module
    model = YOLOv1()
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), str(ckpt))
    load_model(str(ckpt))
    assert api_module._model is not None
