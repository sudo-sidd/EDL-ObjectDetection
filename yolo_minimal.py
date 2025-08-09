#!/usr/bin/env python3
# minimal, modular YOLO-style detector (anchor-free, single-scale stride=16)
# comment instructions only

# This monolithic script is superseded by the modular package in ./EDL
# Prefer using: train.py, predict.py, predict_split.py

import os
import sys
import cv2
import yaml
import math
import time
import json
import random
import argparse
import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import shutil

# ---------------------------
# utils
# ---------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_str():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def make_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def parse_device(arg: str) -> torch.device:
    if arg.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def list_images(path: str) -> List[str]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == '.txt':
        with open(p, 'r') as f:
            files = [l.strip() for l in f if l.strip()]
        return files
    if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
        return [str(p)]
    if p.is_dir():
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        return [str(x) for x in sorted(p.rglob('*')) if x.suffix.lower() in exts]
    raise FileNotFoundError(f"Invalid images path: {path}")


def derive_label_path(img_path: str, base_labels: Optional[str] = None) -> str:
    imgp = Path(img_path)
    if base_labels is not None:
        return str(Path(base_labels) / (imgp.stem + '.txt'))
    parts = list(imgp.parts)
    if 'images' in parts:
        i = parts.index('images')
        parts[i] = 'labels'
        labelp = Path(*parts).with_suffix('.txt')
        return str(labelp)
    return str(imgp.with_suffix('.txt'))


def read_label_file(label_path: str) -> np.ndarray:
    if not Path(label_path).exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            c, cx, cy, w, h = parts
            rows.append([int(float(c)), float(cx), float(cy), float(w), float(h)])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def draw_detections(img: np.ndarray, dets: List[Tuple[float, float, float, float, float, int]], class_names: List[str]):
    for x1, y1, x2, y2, conf, cls in dets:
        color = (0, 255, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_names[cls] if 0 <= cls < len(class_names) else cls}:{conf:.2f}"
        cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = (iou <= iou_thres).nonzero(as_tuple=False).squeeze(1)
        order = order[inds + 1]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def xywhn_to_xyxy_pixels(xywhn: torch.Tensor, img_size: int) -> torch.Tensor:
    cx, cy, w, h = xywhn.unbind(-1)
    x1 = (cx - w / 2.0) * img_size
    y1 = (cy - h / 2.0) * img_size
    x2 = (cx + w / 2.0) * img_size
    y2 = (cy + h / 2.0) * img_size
    return torch.stack([x1, y1, x2, y2], dim=-1)


def iou_xywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax1y1x2y2 = xywh_to_xyxy(a)
    bx1y1x2y2 = xywh_to_xyxy(b)
    return iou_xyxy(ax1y1x2y2, bx1y1x2y2)


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

# ---------------------------
# dataset
# ---------------------------

class YOLOTxtDataset(Dataset):
    # images resized to square, labels assumed normalized to original image size
    def __init__(self, data_yaml: str, split: str = 'train', img_size: int = 640):
        super().__init__()
        y = load_yaml(data_yaml)
        base = y.get('path', None)
        names = y.get('names', None)
        self.names = names if isinstance(names, list) else []
        assert split in ('train', 'val', 'test'), 'split must be train|val|test'
        key = split
        if key not in y:
            raise ValueError(f"Missing '{key}' in dataset yaml")
        p = y[key]
        if base is not None and not os.path.isabs(p):
            p = str(Path(base) / p)
        self.images = list_images(p)
        if len(self.images) == 0:
            raise ValueError(f"No images found for split {split} at {p}")
        # labels root inference
        labels_root = None
        if 'labels' in y:
            lr = y['labels']
            if base is not None and not os.path.isabs(lr):
                lr = str(Path(base) / lr)
            labels_root = lr
        self.labels_root = labels_root
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        label_path = derive_label_path(img_path, self.labels_root)
        img0 = cv2.imread(img_path)
        if img0 is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img0, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        labels = read_label_file(label_path)
        # labels: [N, 5] cls cx cy w h normalized
        sample = {
            'image': torch.from_numpy(img),
            'labels': torch.from_numpy(labels),
            'im_file': img_path
        }
        return sample

    @property
    def num_classes(self) -> int:
        return len(self.names)


def collate_fn(batch):
    images = torch.stack([b['image'] for b in batch], dim=0)
    labels = [b['labels'] for b in batch]
    paths = [b['im_file'] for b in batch]
    return {'images': images, 'labels': labels, 'paths': paths}

# ---------------------------
# model
# ---------------------------

class ConvBNAct(nn.Module):
    # Conv-Norm-Act
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TinyBackbone(nn.Module):
    # stride 16 output
    def __init__(self, c1=3, c2=128):
        super().__init__()
        self.stem = ConvBNAct(c1, 32, 3, 2)   # 1/2
        self.c2 = ConvBNAct(32, 64, 3, 2)     # 1/4
        self.c3 = nn.Sequential(
            ConvBNAct(64, 64, 3, 1),
            ConvBNAct(64, 64, 3, 1),
            ConvBNAct(64, 96, 1, 1)
        )
        self.c4 = ConvBNAct(96, 128, 3, 2)    # 1/8
        self.c5 = nn.Sequential(
            ConvBNAct(128, 128, 3, 1),
            ConvBNAct(128, 128, 3, 1),
            ConvBNAct(128, c2, 1, 1)
        )
        self.c6 = ConvBNAct(c2, c2, 3, 2)     # 1/16

    def forward(self, x):
        x = self.stem(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        return x  # stride 16


class SimpleNeck(nn.Module):
    # lightweight neck
    def __init__(self, c=128):
        super().__init__()
        self.m = nn.Sequential(
            ConvBNAct(c, c, 3, 1),
            ConvBNAct(c, c, 3, 1)
        )

    def forward(self, x):
        return self.m(x)


class Head(nn.Module):
    # direct prediction head
    def __init__(self, c=128, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.out_ch = 1 + num_classes + 4
        self.pred = nn.Conv2d(c, self.out_ch, 1, 1, 0)

    def forward(self, x):
        return self.pred(x)


class MiniYOLO(nn.Module):
    # conv backbone -> neck -> head
    def __init__(self, num_classes: int, channels: int = 128):
        super().__init__()
        self.backbone = TinyBackbone(3, channels)
        self.neck = SimpleNeck(channels)
        self.head = Head(channels, num_classes)
        self.stride = 16
        self.num_classes = num_classes

    def forward(self, x):
        f = self.backbone(x)
        f = self.neck(f)
        p = self.head(f)
        return p

# ---------------------------
# loss and assignment
# ---------------------------

def assign_targets(labels_list: List[torch.Tensor], S: int, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # returns obj_t [B,1,S,S], cls_t [B,C,S,S], box_t [B,4,S,S], mask [B,1,S,S]
    B = len(labels_list)
    obj_t = torch.zeros((B, 1, S, S), dtype=torch.float32)
    cls_t = torch.zeros((B, num_classes, S, S), dtype=torch.float32)
    box_t = torch.zeros((B, 4, S, S), dtype=torch.float32)
    mask = torch.zeros((B, 1, S, S), dtype=torch.bool)
    for b, labels in enumerate(labels_list):
        if labels.numel() == 0:
            continue
        # resolve collisions by keeping largest box per cell
        cell_best_area: Dict[Tuple[int, int], Tuple[float, int]] = {}
        for k in range(labels.shape[0]):
            c, cx, cy, w, h = labels[k].tolist()
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            i = min(max(int(cx * S), 0), S - 1)
            j = min(max(int(cy * S), 0), S - 1)
            area = w * h
            key = (j, i)
            if key not in cell_best_area or area > cell_best_area[key][0]:
                cell_best_area[key] = (area, k)
        for (j, i), (_, k) in cell_best_area.items():
            c, cx, cy, w, h = labels[k].tolist()
            gcx = cx * S
            gcy = cy * S
            dx = gcx - i
            dy = gcy - j
            obj_t[b, 0, j, i] = 1.0
            cls_index = int(c)
            if 0 <= cls_index < num_classes:
                cls_t[b, cls_index, j, i] = 1.0
            box_t[b, :, j, i] = torch.tensor([cx, cy, w, h], dtype=torch.float32)
            mask[b, 0, j, i] = True
    return obj_t, cls_t, box_t, mask


def compute_loss(pred: torch.Tensor, obj_t: torch.Tensor, cls_t: torch.Tensor, box_t: torch.Tensor, mask: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    B, C, S, _ = pred.shape
    obj_logits = pred[:, 0:1]
    cls_logits = pred[:, 1:1+num_classes]
    box_logits = pred[:, 1+num_classes:1+num_classes+4]

    obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_t, reduction='mean')

    # classes only on positives
    pos = mask.expand_as(cls_logits)
    if pos.any():
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits[pos], cls_t[pos], reduction='mean')
    else:
        cls_loss = torch.tensor(0.0, device=pred.device)

    # decode predicted boxes to normalized [0,1]
    # predicted center per cell via sigmoid offset dx,dy relative to cell index
    # predicted w,h via sigmoid
    grid_y, grid_x = torch.meshgrid(torch.arange(S, device=pred.device), torch.arange(S, device=pred.device), indexing='ij')
    grid_x = grid_x.view(1, 1, S, S).float()
    grid_y = grid_y.view(1, 1, S, S).float()

    dx = torch.sigmoid(box_logits[:, 0:1])
    dy = torch.sigmoid(box_logits[:, 1:2])
    pw = torch.sigmoid(box_logits[:, 2:3])
    ph = torch.sigmoid(box_logits[:, 3:4])

    cx_pred = (grid_x + dx) / S
    cy_pred = (grid_y + dy) / S
    w_pred = pw
    h_pred = ph

    pred_box = torch.cat([cx_pred, cy_pred, w_pred, h_pred], dim=1)  # [B,4,S,S]

    # IoU loss only on positives
    if mask.any():
        pb = pred_box.permute(0, 2, 3, 1)[mask.squeeze(1)]  # [P,4]
        tb = box_t.permute(0, 2, 3, 1)[mask.squeeze(1)]     # [P,4]
        iou = iou_xywh(pb, tb)
        box_loss = (1.0 - iou).mean()
    else:
        box_loss = torch.tensor(0.0, device=pred.device)

    total = obj_loss + cls_loss + box_loss
    logs = {
        'loss': float(total.detach().item()),
        'obj_loss': float(obj_loss.detach().item()),
        'cls_loss': float(cls_loss.detach().item()),
        'box_loss': float(box_loss.detach().item()),
    }
    return total, logs

# ---------------------------
# inference
# ---------------------------

def decode_predictions(pred: torch.Tensor, num_classes: int, conf_thres: float, iou_thres: float, img_size: int, max_det: int = 300) -> List[torch.Tensor]:
    B, C, S, _ = pred.shape
    obj = torch.sigmoid(pred[:, 0:1])
    cls = torch.sigmoid(pred[:, 1:1+num_classes])
    box = pred[:, 1+num_classes:1+num_classes+4]

    grid_y, grid_x = torch.meshgrid(torch.arange(S, device=pred.device), torch.arange(S, device=pred.device), indexing='ij')
    grid_x = grid_x.view(1, 1, S, S).float()
    grid_y = grid_y.view(1, 1, S, S).float()

    dx = torch.sigmoid(box[:, 0:1])
    dy = torch.sigmoid(box[:, 1:2])
    pw = torch.sigmoid(box[:, 2:3])
    ph = torch.sigmoid(box[:, 3:4])

    cx = (grid_x + dx) / S
    cy = (grid_y + dy) / S
    w = pw
    h = ph

    xywhn = torch.cat([cx, cy, w, h], dim=1)  # [B,4,S,S]
    xyxyp = xywhn_to_xyxy_pixels(xywhn.permute(0, 2, 3, 1).reshape(B, S*S, 4), img_size)

    obj_flat = obj.view(B, 1, S*S)
    cls_flat = cls.view(B, num_classes, S*S)

    outputs = []
    for b in range(B):
        scores = (obj_flat[b] * cls_flat[b]).squeeze(0)  # [C,S*S]
        scores_max, labels = scores.max(dim=0)           # [S*S]
        mask_conf = scores_max > conf_thres
        if mask_conf.sum() == 0:
            outputs.append(torch.zeros((0, 6), device=pred.device))
            continue
        boxes_b = xyxyp[b][mask_conf]
        scores_b = scores_max[mask_conf]
        labels_b = labels[mask_conf]
        keep = nms(boxes_b, scores_b, iou_thres)
        if keep.numel() > max_det:
            keep = keep[:max_det]
        det = torch.cat([boxes_b[keep], scores_b[keep].unsqueeze(1), labels_b[keep].unsqueeze(1).float()], dim=1)
        outputs.append(det)
    return outputs

# ---------------------------
# training and evaluation
# ---------------------------

def train(args):
    seed_everything(args.seed)
    device = parse_device(args.device)
    ds = YOLOTxtDataset(args.data, split='train', img_size=args.imgsz)
    names = ds.names
    nc = ds.num_classes if args.num_classes is None else args.num_classes
    if nc != len(names) and len(names) > 0:
        print(f"[warn] nc({nc}) != len(names)({len(names)}), using nc={nc}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    val_dl = None
    try:
        _ = YOLOTxtDataset(args.data, split='val', img_size=args.imgsz)
        val_dl = DataLoader(YOLOTxtDataset(args.data, split='val', img_size=args.imgsz), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    except Exception as e:
        print(f"[info] val split unavailable: {e}")

    model = MiniYOLO(num_classes=nc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    out_dir = args.out or f"runs/train/exp-{now_str()}"
    weights_dir = str(Path(out_dir) / 'weights')
    make_dir(weights_dir)
    make_dir('runs/train')
    meta = {
        'imgsz': args.imgsz,
        'stride': 16,
        'names': names,
        'num_classes': nc,
    }
    with open(Path(out_dir) / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    model.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        t0 = time.time()
        for batch in dl:
            imgs = batch['images'].to(device)
            labels = [l.to(device) for l in batch['labels']]
            B, _, H, W = imgs.shape
            assert H == W and H % 16 == 0, 'image must be square and divisible by 16'
            S = H // 16

            pred = model(imgs)
            obj_t, cls_t, box_t, mask = assign_targets(labels, S, nc)
            obj_t = obj_t.to(device)
            cls_t = cls_t.to(device)
            box_t = box_t.to(device)
            mask = mask.to(device)

            loss, logs = compute_loss(pred, obj_t, cls_t, box_t, mask, nc)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()

            epoch_loss += logs['loss']
            global_step += 1

            if global_step % args.log_interval == 0:
                print(f"epoch {epoch+1}/{args.epochs} step {global_step} loss {logs['loss']:.4f} obj {logs['obj_loss']:.4f} cls {logs['cls_loss']:.4f} box {logs['box_loss']:.4f}")

        dt = time.time() - t0
        avg_loss = epoch_loss / max(1, len(dl))
        print(f"[epoch {epoch+1}] avg_loss {avg_loss:.4f} time {dt:.1f}s")

        last_w = Path(weights_dir) / 'last.pt'
        torch.save({'model': model.state_dict(), 'meta': meta}, last_w)
        try:
            shutil.copyfile(last_w, Path('runs/train/last.pt'))
        except Exception as _:
            pass
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_w = Path(weights_dir) / 'best.pt'
            torch.save({'model': model.state_dict(), 'meta': meta}, best_w)
            try:
                shutil.copyfile(best_w, Path('runs/train/best.pt'))
            except Exception as _:
                pass

    print(f"[done] weights saved to {weights_dir}")


def load_model(weights: str, device: torch.device, num_classes: Optional[int] = None) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(weights, map_location=device)
    meta = ckpt.get('meta', {})
    nc_meta = meta.get('num_classes', None)
    imgsz = meta.get('imgsz', 640)
    nc = num_classes if num_classes is not None else (nc_meta if nc_meta is not None else 1)
    model = MiniYOLO(num_classes=nc).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, meta


def predict_on_images(args, paths: List[str], save_dir: str):
    device = parse_device(args.device)
    model, meta = load_model(args.weights, device, args.num_classes)
    names = meta.get('names', [])
    imgsz = meta.get('imgsz', args.imgsz)
    make_dir(save_dir)
    for p in paths:
        img0 = cv2.imread(p)
        if img0 is None:
            print(f"[warn] skip unreadable {p}")
            continue
        img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_t)
            dets = decode_predictions(pred, model.num_classes, args.conf, args.iou, imgsz, max_det=args.max_det)[0]
        det_list = []
        for x1, y1, x2, y2, conf, cls in dets.detach().cpu().numpy().tolist():
            det_list.append((x1, y1, x2, y2, conf, int(cls)))
        img_draw = img.copy()
        img_draw = draw_detections(img_draw, det_list, names)
        out_path = str(Path(save_dir) / (Path(p).stem + '_pred.jpg'))
        cv2.imwrite(out_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
        print(f"saved {out_path}")


def predict_image(args):
    save_dir = args.save_dir or f"runs/predict-image/exp-{now_str()}"
    predict_on_images(args, [args.source], save_dir)


def predict_batch(args):
    save_dir = args.save_dir or f"runs/predict-batch/exp-{now_str()}"
    paths = list_images(args.source)
    predict_on_images(args, paths, save_dir)


def predict_video(args):
    device = parse_device(args.device)
    model, meta = load_model(args.weights, device, args.num_classes)
    names = meta.get('names', [])
    imgsz = meta.get('imgsz', args.imgsz)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    save_dir = args.save_dir or f"runs/predict-video/exp-{now_str()}"
    make_dir(save_dir)
    out_path = str(Path(save_dir) / (Path(args.source).stem + '_pred.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (imgsz, imgsz))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_t)
            dets = decode_predictions(pred, model.num_classes, args.conf, args.iou, imgsz, max_det=args.max_det)[0]
        det_list = []
        for x1, y1, x2, y2, conf, cls in dets.detach().cpu().numpy().tolist():
            det_list.append((x1, y1, x2, y2, conf, int(cls)))
        img_draw = img.copy()
        img_draw = draw_detections(img_draw, det_list, names)
        out_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
        writer.write(out_bgr)

    cap.release()
    writer.release()
    print(f"saved {out_path}")


def predict_split(args):
    ds = YOLOTxtDataset(args.data, split=args.split, img_size=args.imgsz)
    save_dir = args.save_dir or f"runs/predict-{args.split}/exp-{now_str()}"
    predict_on_images(args, ds.images, save_dir)

# ---------------------------
# cli
# ---------------------------

def build_argparser():
    p = argparse.ArgumentParser(description='Minimal YOLO-style object detector (anchor-free, stride=16)')
    sub = p.add_subparsers(dest='cmd', required=True)

    # train
    pt = sub.add_parser('train', help='train model')
    pt.add_argument('--data', type=str, default='data.yaml')
    pt.add_argument('--epochs', type=int, default=100)
    pt.add_argument('--batch', type=int, default=16)
    pt.add_argument('--imgsz', type=int, default=640)
    pt.add_argument('--device', type=str, default='cuda')
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--workers', type=int, default=4)
    pt.add_argument('--out', type=str, default='')
    pt.add_argument('--seed', type=int, default=42)
    pt.add_argument('--num-classes', dest='num_classes', type=int, default=None)
    pt.add_argument('--log-interval', type=int, default=50)

    # predict image
    pi = sub.add_parser('predict-image', help='predict a single image')
    pi.add_argument('--weights', type=str, default='runs/train/last.pt')
    pi.add_argument('--source', type=str, required=True)
    pi.add_argument('--save-dir', type=str, default='')
    pi.add_argument('--conf', type=float, default=0.25)
    pi.add_argument('--iou', type=float, default=0.45)
    pi.add_argument('--max-det', type=int, default=300)
    pi.add_argument('--device', type=str, default='cuda')
    pi.add_argument('--imgsz', type=int, default=640)
    pi.add_argument('--num-classes', dest='num_classes', type=int, default=None)

    # predict batch (directory or txt)
    pb = sub.add_parser('predict-batch', help='predict images in directory or txt file')
    pb.add_argument('--weights', type=str, default='runs/train/last.pt')
    pb.add_argument('--source', type=str, required=True)
    pb.add_argument('--save-dir', type=str, default='')
    pb.add_argument('--conf', type=float, default=0.25)
    pb.add_argument('--iou', type=float, default=0.45)
    pb.add_argument('--max-det', type=int, default=300)
    pb.add_argument('--device', type=str, default='cuda')
    pb.add_argument('--imgsz', type=int, default=640)
    pb.add_argument('--num-classes', dest='num_classes', type=int, default=None)

    # predict video
    pv = sub.add_parser('predict-video', help='predict a video file')
    pv.add_argument('--weights', type=str, default='runs/train/last.pt')
    pv.add_argument('--source', type=str, required=True)
    pv.add_argument('--save-dir', type=str, default='')
    pv.add_argument('--conf', type=float, default=0.25)
    pv.add_argument('--iou', type=float, default=0.45)
    pv.add_argument('--max-det', type=int, default=300)
    pv.add_argument('--device', type=str, default='cuda')
    pv.add_argument('--imgsz', type=int, default=640)
    pv.add_argument('--num-classes', dest='num_classes', type=int, default=None)

    # predict dataset split
    ps = sub.add_parser('predict-split', help='predict a dataset split from yaml (train/val/test)')
    ps.add_argument('--weights', type=str, default='runs/train/last.pt')
    ps.add_argument('--data', type=str, default='data.yaml')
    ps.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    ps.add_argument('--save-dir', type=str, default='')
    ps.add_argument('--conf', type=float, default=0.25)
    ps.add_argument('--iou', type=float, default=0.45)
    ps.add_argument('--max-det', type=int, default=300)
    ps.add_argument('--device', type=str, default='cuda')
    ps.add_argument('--imgsz', type=int, default=640)
    ps.add_argument('--num-classes', dest='num_classes', type=int, default=None)

    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'predict-image':
        predict_image(args)
    elif args.cmd == 'predict-batch':
        predict_batch(args)
    elif args.cmd == 'predict-video':
        predict_video(args)
    elif args.cmd == 'predict-split':
        predict_split(args)
    else:
        raise ValueError(f"unknown cmd {args.cmd}")


if __name__ == '__main__':
    main()
