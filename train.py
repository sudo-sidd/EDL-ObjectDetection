#!/usr/bin/env python3
# comment instructions only
from miniyolo.cli import build_argparser
from miniyolo.engine import train_loop

if __name__ == '__main__':
    args = build_argparser().parse_args()
    assert args.cmd == 'train'
    train_loop(args)
