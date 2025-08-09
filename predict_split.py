#!/usr/bin/env python3
# comment instructions only
from EDL.cli import build_argparser
from EDL.engine import predict_on_images
from EDL.data import YOLOTxtDataset
from EDL.utils import now_str

if __name__ == '__main__':
    args = build_argparser().parse_args()
    assert args.cmd == 'predict-split'
    ds = YOLOTxtDataset(args.data, split=args.split, img_size=args.imgsz)
    save_dir = args.save_dir or f"runs/predict-{args.split}/exp-{now_str()}"
    predict_on_images(args, ds.images, save_dir)
