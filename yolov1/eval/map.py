from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from .iou import box_iou_xywh

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
    "chair","cow","diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor",
]


def compute_ap(recall, precision):
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map(all_preds, all_targets, iou_thresh=0.5, n_classes=20):
    class_preds = defaultdict(list)
    class_gts = defaultdict(int)
    for img_id, (preds, tgt) in enumerate(zip(all_preds, all_targets)):
        for gt_cls in tgt["labels"]:
            class_gts[gt_cls] += 1
        for det in preds:
            x1, y1, x2, y2, conf, cls = det
            class_preds[cls].append((img_id, conf, x1, y1, x2, y2))
    aps = {}
    for cls in range(n_classes):
        dets = sorted(class_preds[cls], key=lambda x: -x[1])
        n_gt = class_gts[cls]
        if n_gt == 0:
            aps[VOC_CLASSES[cls]] = 0.0
            continue
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        matched = defaultdict(set)
        for d_idx, (img_id, conf, *box) in enumerate(dets):
            tgt = all_targets[img_id]
            best_iou, best_j = 0.0, -1
            for j, (gt_box, gt_cls) in enumerate(zip(tgt["boxes"], tgt["labels"])):
                if gt_cls != cls or j in matched[img_id]:
                    continue
                iou = box_iou_xywh(
                    torch.tensor(box).unsqueeze(0),
                    torch.tensor(gt_box).unsqueeze(0),
                ).item()
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh:
                tp[d_idx] = 1
                matched[img_id].add(best_j)
            else:
                fp[d_idx] = 1
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / n_gt
        prec = tp_cum / (tp_cum + fp_cum + 1e-6)
        aps[VOC_CLASSES[cls]] = compute_ap(rec, prec)
    aps["mAP"] = float(np.mean(list(aps.values())))
    return aps
