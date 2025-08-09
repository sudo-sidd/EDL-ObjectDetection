# comment instructions only
import argparse
from pathlib import Path

from .engine import train_loop, predict_on_images, load_model, decode_predictions
from .data import YOLOTxtDataset
from .utils import list_images, now_str, make_dir, parse_device

import cv2
import numpy as np
import torch


def build_argparser():
    p = argparse.ArgumentParser(description='Minimal YOLO-style object detector (anchor-free, stride=16)')
    sub = p.add_subparsers(dest='cmd', required=True)

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


def predict_image(args):
    from .engine import predict_on_images
    save_dir = args.save_dir or f"runs/predict-image/exp-{now_str()}"
    predict_on_images(args, [args.source], save_dir)


def predict_batch(args):
    from .engine import predict_on_images
    save_dir = args.save_dir or f"runs/predict-batch/exp-{now_str()}"
    paths = list_images(args.source)
    predict_on_images(args, paths, save_dir)


def predict_video(args):
    from .engine import load_model, decode_predictions
    device = parse_device(args.device)
    model, meta = load_model(args.weights, device, args.num_classes)
    names = meta.get('names', [])
    imgsz = meta.get('imgsz', args.imgsz)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.source}")
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
        writer.write(cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))

    cap.release()
    writer.release()
    print(f"saved {out_path}")
