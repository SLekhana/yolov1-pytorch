from __future__ import annotations
import time
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel
from ..model.yolov1 import YOLOv1
from ..eval.nms import decode_predictions

app = FastAPI(title="YOLOv1 Detection API", version="1.0.0")
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

REQUEST_COUNT = Counter("yolo_requests_total", "Total inference requests")
LATENCY = Histogram("yolo_latency_seconds", "Inference latency")

_model: YOLOv1 | None = None


def load_model(ckpt_path: str):
    global _model
    _model = YOLOv1()
    _model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    _model.eval()


class Detection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class InferenceResponse(BaseModel):
    detections: list[Detection]
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/detect", response_model=InferenceResponse)
async def detect(file: UploadFile = File(...)):
    REQUEST_COUNT.inc()
    start = time.perf_counter()
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (448, 448))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        pred = _model(tensor)
    dets = decode_predictions(pred)[0]
    latency = (time.perf_counter() - start) * 1000
    LATENCY.observe(latency / 1000)
    return InferenceResponse(
        detections=[Detection(x1=d[0], y1=d[1], x2=d[2], y2=d[3], confidence=d[4], class_id=d[5]) for d in dets],
        latency_ms=round(latency, 2),
    )
