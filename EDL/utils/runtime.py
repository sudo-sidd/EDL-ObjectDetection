# comment instructions only
import os
import random
import datetime
from pathlib import Path
import numpy as np
import torch


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
    a = (arg or '').strip().lower()
    if a == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.startswith('cuda'):
        # allows 'cuda' or 'cuda:0', will raise if invalid index
        if a == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device(a)
    return torch.device('cpu')
