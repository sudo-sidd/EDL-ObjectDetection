import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add tqdm with a safe fallback
try:
    from tqdm.auto import tqdm
except Exception:  # minimal fallback if tqdm isn't available
    class _SimpleTQDM:
        def __init__(self, iterable=None, total=None, desc=None, leave=True):
            self.iterable = iterable
            self.total = total if total is not None else (len(iterable) if iterable is not None else None)
            self.desc = desc or ''
            self.count = 0
        def __iter__(self):
            for x in self.iterable:
                self.count += 1
                if self.total:
                    print(f"{self.desc} [{self.count}/{self.total}]", end='\r')
                yield x
            print()
        def set_postfix(self, **kwargs):
            pass
        def close(self):
            pass
    def tqdm(iterable=None, total=None, desc=None, leave=True):
        return _SimpleTQDM(iterable=iterable, total=total, desc=desc, leave=leave)

from .model import MiniYOLO
from .data import YOLOTxtDataset, collate_fn
from .utils import xywhn_to_xyxy_pixels, iou_xywh, nms, draw_detections

# assign targets

def assign_targets(labels_list: List[torch.Tensor], S: int, num_classes: int):
    B = len(labels_list)
    obj_t = torch.zeros((B, 1, S, S), dtype=torch.float32)
    cls_t = torch.zeros((B, num_classes, S, S), dtype=torch.float32)
    box_t = torch.zeros((B, 4, S, S), dtype=torch.float32)
    mask = torch.zeros((B, 1, S, S), dtype=torch.bool)
    for b, labels in enumerate(labels_list):
        if labels.numel() == 0:
            continue
        cell_best = {}
        for k in range(labels.shape[0]):
            c, cx, cy, w, h = labels[k].tolist()
            i = min(max(int(cx * S), 0), S - 1)
            j = min(max(int(cy * S), 0), S - 1)
            area = w * h
            key = (j, i)
            if key not in cell_best or area > cell_best[key][0]:
                cell_best[key] = (area, k)
        for (j, i), (_, k) in cell_best.items():
            c, cx, cy, w, h = labels[k].tolist()
            obj_t[b, 0, j, i] = 1.0
            ci = int(c)
            if 0 <= ci < num_classes:
                cls_t[b, ci, j, i] = 1.0
            box_t[b, :, j, i] = torch.tensor([cx, cy, w, h], dtype=torch.float32)
            mask[b, 0, j, i] = True
    return obj_t, cls_t, box_t, mask

# loss

def compute_loss(pred: torch.Tensor, obj_t: torch.Tensor, cls_t: torch.Tensor, box_t: torch.Tensor, mask: torch.Tensor, num_classes: int):
    B, C, S, _ = pred.shape
    obj_logits = pred[:, 0:1]
    cls_logits = pred[:, 1:1+num_classes]
    box_logits = pred[:, 1+num_classes:1+num_classes+4]

    obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_t, reduction='mean')

    pos = mask.expand_as(cls_logits)
    if pos.any():
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits[pos], cls_t[pos], reduction='mean')
    else:
        cls_loss = torch.tensor(0.0, device=pred.device)

    gy, gx = torch.meshgrid(torch.arange(S, device=pred.device), torch.arange(S, device=pred.device), indexing='ij')
    gx = gx.view(1, 1, S, S).float()
    gy = gy.view(1, 1, S, S).float()

    dx = torch.sigmoid(box_logits[:, 0:1])
    dy = torch.sigmoid(box_logits[:, 1:2])
    pw = torch.sigmoid(box_logits[:, 2:3])
    ph = torch.sigmoid(box_logits[:, 3:4])

    cx_pred = (gx + dx) / S
    cy_pred = (gy + dy) / S
    w_pred = pw
    h_pred = ph

    pred_box = torch.cat([cx_pred, cy_pred, w_pred, h_pred], dim=1)

    if mask.any():
        pb = pred_box.permute(0, 2, 3, 1)[mask.squeeze(1)]
        tb = box_t.permute(0, 2, 3, 1)[mask.squeeze(1)]
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

# decode

def decode_predictions(pred: torch.Tensor, num_classes: int, conf_thres: float, iou_thres: float, img_size: int, max_det: int = 300):
    B, C, S, _ = pred.shape
    obj = torch.sigmoid(pred[:, 0:1])
    cls = torch.sigmoid(pred[:, 1:1+num_classes])
    box = pred[:, 1+num_classes:1+num_classes+4]

    gy, gx = torch.meshgrid(torch.arange(S, device=pred.device), torch.arange(S, device=pred.device), indexing='ij')
    gx = gx.view(1, 1, S, S).float()
    gy = gy.view(1, 1, S, S).float()

    dx = torch.sigmoid(box[:, 0:1])
    dy = torch.sigmoid(box[:, 1:2])
    pw = torch.sigmoid(box[:, 2:3])
    ph = torch.sigmoid(box[:, 3:4])

    cx = (gx + dx) / S
    cy = (gy + dy) / S
    w = pw
    h = ph

    xywhn = torch.cat([cx, cy, w, h], dim=1)
    xyxyp = xywhn_to_xyxy_pixels(xywhn.permute(0, 2, 3, 1).reshape(B, S*S, 4), img_size)

    obj_flat = obj.view(B, S*S)  # Remove extra dimension
    cls_flat = cls.view(B, num_classes, S*S)

    out = []
    for b in range(B):
        scores = (obj_flat[b:b+1].unsqueeze(0) * cls_flat[b]).squeeze(0)  # Fix broadcasting
        scores_max, labels = scores.max(dim=0)
        m = scores_max > conf_thres
        if m.sum() == 0:
            out.append(torch.zeros((0, 6), device=pred.device))
            continue
        boxes_b = xyxyp[b][m]
        scores_b = scores_max[m]
        labels_b = labels[m]
        keep = nms(boxes_b, scores_b, iou_thres)
        if keep.numel() > max_det:
            keep = keep[:max_det]
        det = torch.cat([boxes_b[keep], scores_b[keep].unsqueeze(1), labels_b[keep].unsqueeze(1).float()], dim=1)
        out.append(det)
    return out

# train loop

def train_loop(args):
    from .utils import seed_everything, parse_device, now_str, make_dir
    seed_everything(args.seed)
    device = parse_device(args.device)

    ds = YOLOTxtDataset(args.data, split='train', img_size=args.imgsz)
    names = ds.names
    nc = ds.num_classes if args.num_classes is None else args.num_classes
    if nc != len(names) and len(names) > 0:
        print(f"[warn] nc({nc}) != len(names)({len(names)}), using nc={nc}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

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

    # Training configuration banner
    try:
        gpu_name = torch.cuda.get_device_name(0) if (hasattr(device, 'type') and device.type == 'cuda') and torch.cuda.is_available() else 'CPU'
    except Exception:
        gpu_name = 'CPU'
    n_params = sum(p.numel() for p in model.parameters())
    print("\n🚀 EDL Training Configuration")
    print("=" * 60)
    print(f"Data YAML     : {args.data}")
    print(f"Images        : {len(ds)} train")
    print(f"Classes       : {nc} -> {names}")
    print(f"Image size    : {args.imgsz}")
    print(f"Batch size    : {args.batch}")
    print(f"Epochs        : {args.epochs}")
    print(f"Learning rate : {args.lr}")
    print(f"Workers       : {args.workers}")
    print(f"Device        : {device} ({gpu_name})")
    print(f"Model params  : {n_params:,}")
    print(f"Output dir    : {out_dir}")
    print("=" * 60)

    model.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        t0 = time.time()
        pbar = tqdm(dl, total=len(dl), desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for batch in pbar:
            imgs = batch['images'].to(device)
            labels = [l.to(device) for l in batch['labels']]
            B, _, H, W = imgs.shape
            assert H == W and H % 16 == 0
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

            # Update progress bar postfix with current metrics
            pbar.set_postfix({
                'loss': f"{logs['loss']:.4f}",
                'obj': f"{logs['obj_loss']:.4f}",
                'cls': f"{logs['cls_loss']:.4f}",
                'box': f"{logs['box_loss']:.4f}",
            })
        pbar.close()

        dt = time.time() - t0
        avg_loss = epoch_loss / max(1, len(dl))
        print(f"[epoch {epoch+1}] avg_loss {avg_loss:.4f} time {dt:.1f}s")

        last_w = Path(weights_dir) / 'last.pt'
        torch.save({'model': model.state_dict(), 'meta': meta}, last_w)
        try:
            import shutil
            shutil.copyfile(last_w, Path('runs/train/last.pt'))
        except Exception:
            pass
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_w = Path(weights_dir) / 'best.pt'
            torch.save({'model': model.state_dict(), 'meta': meta}, best_w)
            try:
                import shutil
                shutil.copyfile(best_w, Path('runs/train/best.pt'))
            except Exception:
                pass

    print(f"[done] weights saved to {weights_dir}")

# predict utils

def predict_on_images(args, paths: List[str], save_dir: str):
    from .utils import parse_device, make_dir, draw_detections
    import cv2
    import numpy as np

    device = parse_device(args.device)
    model, meta = load_model(args.weights, device, args.num_classes)
    names = meta.get('names', [])
    imgsz = meta.get('imgsz', args.imgsz)

    save_outputs = bool(save_dir)
    if save_outputs:
        make_dir(save_dir)

    results = []  # collect (path, annotated_image, boxes_dict)

    for p in paths:
        t0 = time.time()
        out_path = ''
        img0 = cv2.imread(p)
        if img0 is None:
            print(f"[warn] skip unreadable {p}")
            continue
        h0, w0 = img0.shape[:2]
        img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_t)
            dets = decode_predictions(
                pred,
                model.num_classes,
                conf_thres=float(args.conf),
                iou_thres=args.iou,
                img_size=imgsz,
                max_det=args.max_det,
            )[0]

        # Build boxes dict with ids like apple1, apple2, ...
        det_np = dets.detach().cpu().numpy()
        # sort by confidence desc for stable numbering
        if det_np.size:
            det_np = det_np[(-det_np[:, 4]).argsort()]
        # scale back to original resolution
        scale_x = w0 / float(imgsz)
        scale_y = h0 / float(imgsz)
        counts: Dict[str, int] = {}
        boxes_dict: Dict[str, Dict[str, object]] = {}
        det_list = []
        for x1, y1, x2, y2, conf, cls in det_np.tolist():
            x1o = x1 * scale_x; y1o = y1 * scale_y; x2o = x2 * scale_x; y2o = y2 * scale_y
            cls_int = int(cls)
            label = names[cls_int] if 0 <= cls_int < len(names) and names else str(cls_int)
            counts[label] = counts.get(label, 0) + 1
            obj_id = f"{label}{counts[label]}"
            boxes_dict[obj_id] = {
                'bbox': [float(x1o), float(y1o), float(x2o), float(y2o)],
                'conf': float(conf),
                'cls': label,
            }
            det_list.append((x1o, y1o, x2o, y2o, conf, cls_int))

        # draw on original resolution
        img_draw = draw_detections(img_rgb.copy(), det_list, names, thickness=max(1, int(imgsz/640)))

        if save_outputs:
            out_path = str(Path(save_dir) / (Path(p).stem + '_pred.jpg'))
            cv2.imwrite(out_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))

        # concise per-image debug line
        dt_ms = (time.time() - t0) * 1000.0
        print(f"{Path(p).name} det={len(det_list)} conf>={float(args.conf):.2f} time={dt_ms:.1f}ms out={out_path if out_path else '-'}")

        results.append({
            'path': p,
            'image': img_draw,  # RGB numpy array at original resolution
            'boxes': boxes_dict,
        })

    return results

# Simple video prediction (optional saving, does not return frames to avoid high memory)

def predict_on_video(args, source: str | int, save_dir: str):
    from .utils import parse_device, make_dir, draw_detections
    import cv2
    import numpy as np

    device = parse_device(args.device)
    model, meta = load_model(args.weights, device, args.num_classes)
    names = meta.get('names', [])
    imgsz = meta.get('imgsz', args.imgsz)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    save_outputs = bool(save_dir)
    writer = None
    out_path = None
    if save_outputs:
        make_dir(save_dir)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = str(Path(save_dir) / (f"{Path(str(source)).stem if isinstance(source, str) else 'webcam'}_pred.mp4"))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    total_boxes = 0
    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            t0 = time.time()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h0, w0 = frame_rgb.shape[:2]
            resized = cv2.resize(frame_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            img_t = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)
            pred = model(img_t)
            dets = decode_predictions(pred, model.num_classes, float(args.conf), args.iou, imgsz, args.max_det)[0]
            # prepare detections scaled to original resolution
            det_np = dets.detach().cpu().numpy()
            if det_np.size:
                det_np = det_np[(-det_np[:, 4]).argsort()]
            scale_x = w0 / float(imgsz)
            scale_y = h0 / float(imgsz)
            det_list = []
            for x1, y1, x2, y2, conf, cls in det_np.tolist():
                x1o = x1 * scale_x; y1o = y1 * scale_y; x2o = x2 * scale_x; y2o = y2 * scale_y
                det_list.append((x1o, y1o, x2o, y2o, float(conf), int(cls)))
            annotated = draw_detections(frame_rgb.copy(), det_list, names, thickness=max(1, int(imgsz/640)))
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            if writer is not None:
                writer.write(annotated_bgr)
            frame_idx += 1
            total_boxes += len(det_list)
            # concise per-frame debug
            dt_ms = (time.time() - t0) * 1000.0
            print(f"frame={frame_idx} det={len(det_list)} conf>={float(args.conf):.2f} time={dt_ms:.1f}ms out={out_path if out_path else '-'}")

    cap.release()
    if writer is not None:
        writer.release()

    return {
        'saved_path': out_path,
        'frames': frame_idx,
        'total_boxes': total_boxes,
    }
# load weights

def load_model(weights: str, device: torch.device, num_classes: Optional[int] = None):
    model, meta = None, {}
    ckpt = torch.load(weights, map_location=device)
    if not isinstance(ckpt, dict) or 'model' not in ckpt:
        raise RuntimeError(f"Checkpoint missing 'model' key: {weights}")
    meta = ckpt.get('meta', {}) or {}
    names = meta.get('names', [])
    nc_meta = meta.get('num_classes', None)
    imgsz = meta.get('imgsz', 640)

    # Prefer explicit num_classes, else use from meta, else None (set later)
    nc = num_classes if num_classes is not None else (nc_meta if nc_meta is not None else None)

    # Quick safety defaults to avoid downstream shape/index errors
    if not names:
        print("[warn] checkpoint missing class names; defaulting to ['class0']")
        meta['names'] = ["class0"]
        if num_classes is None:  # only override if user didn't explicitly set it
            nc = 1
            meta['num_classes'] = 1

    if nc is None:
        print("[warn] checkpoint missing num_classes; defaulting to 1")
        nc = 1
        meta['num_classes'] = 1

    if isinstance(meta.get('names'), list) and len(meta['names']) != nc:
        print(f"[warn] num_classes({nc}) != len(names)({len(meta['names'])}); proceeding")

    model = MiniYOLO(num_classes=nc).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, meta
