#!/usr/bin/env python3
# comment instructions only
from miniyolo.cli import build_argparser, predict_image, predict_batch, predict_video

if __name__ == '__main__':
    args = build_argparser().parse_args()
    if args.cmd == 'predict-image':
        predict_image(args)
    elif args.cmd == 'predict-batch':
        predict_batch(args)
    elif args.cmd == 'predict-video':
        predict_video(args)
    else:
        raise SystemExit('use predict-image|predict-batch|predict-video')
