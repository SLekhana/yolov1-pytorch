import typer
import torch
import cv2
from yolov1.model.yolov1 import YOLOv1
from yolov1.eval.nms import decode_predictions
from yolov1.data.voc_dataset import VOC_CLASSES

app = typer.Typer()


@app.command()
def detect(
    image: str = typer.Argument(...),
    checkpoint: str = typer.Option("checkpoints/best.pt"),
    conf_thresh: float = 0.3,
    iou_thresh: float = 0.5,
    output: str = "output.jpg",
):
    model = YOLOv1()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    img = cv2.imread(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (448, 448))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        pred = model(tensor)
    dets = decode_predictions(pred, conf_thresh=conf_thresh, iou_thresh=iou_thresh)[0]
    h, w = img.shape[:2]
    for det in dets:
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(img, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 255, 0), 2)
        cv2.putText(img, f"{VOC_CLASSES[int(cls)]} {conf:.2f}", (int(x1*w), int(y1*h)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(output, img)
    typer.echo(f"Saved to {output} — {len(dets)} detections")


if __name__ == "__main__":
    app()
