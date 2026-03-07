from __future__ import annotations
import torch
from torchvision.ops import nms


def decode_predictions(
    pred: torch.Tensor,
    S: int = 7,
    B: int = 2,
    C: int = 20,
    conf_thresh: float = 0.3,
    iou_thresh: float = 0.5,
) -> list[list[list[float]]]:
    results: list[list[list[float]]] = []
    N = pred.shape[0]
    for n in range(N):
        boxes: list[list[float]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for i in range(S):
            for j in range(S):
                cell = pred[n, i, j]
                cls_probs = cell[:C]
                cls_id = int(cls_probs.argmax().item())
                cls_prob = cls_probs[cls_id].item()
                for b in range(B):
                    off = C + b * 5
                    conf = cell[off].item()
                    score = conf * cls_prob
                    if score < conf_thresh:
                        continue
                    cx = (cell[off + 1].item() + j) / S
                    cy = (cell[off + 2].item() + i) / S
                    bw = cell[off + 3].item()
                    bh = cell[off + 4].item()
                    x1 = max(0.0, cx - bw / 2)
                    y1 = max(0.0, cy - bh / 2)
                    x2 = min(1.0, cx + bw / 2)
                    y2 = min(1.0, cy + bh / 2)
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    class_ids.append(cls_id)
        if not boxes:
            results.append([])
            continue
        boxes_t = torch.tensor(boxes)
        scores_t = torch.tensor(scores)
        keep = nms(boxes_t, scores_t, iou_thresh)
        results.append([[*boxes_t[k].tolist(), scores_t[k].item(), class_ids[k]] for k in keep])
    return results
