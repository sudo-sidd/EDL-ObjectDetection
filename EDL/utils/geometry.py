 import torch


def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = xywh.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)
    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def iou_xywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax1y1x2y2 = xywh_to_xyxy(a)
    bx1y1x2y2 = xywh_to_xyxy(b)
    return iou_xyxy(ax1y1x2y2, bx1y1x2y2)


def xywhn_to_xyxy_pixels(xywhn: torch.Tensor, img_size: int) -> torch.Tensor:
    cx, cy, w, h = xywhn.unbind(-1)
    x1 = (cx - w / 2.0) * img_size
    y1 = (cy - h / 2.0) * img_size
    x2 = (cx + w / 2.0) * img_size
    y2 = (cy + h / 2.0) * img_size
    return torch.stack([x1, y1, x2, y2], dim=-1)
