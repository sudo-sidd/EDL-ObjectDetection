# comment instructions only
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.io import load_yaml, list_images, derive_label_path, read_label_file

class YOLOTxtDataset(Dataset):
    def __init__(self, data_yaml: str, split: str = 'train', img_size: int = 640):
        super().__init__()
        y = load_yaml(data_yaml)
        yaml_dir = Path(data_yaml).parent  # Get YAML file directory
        base = y.get('path', None)
        names = y.get('names', None)
        self.names = names if isinstance(names, list) else []
        assert split in ('train', 'val', 'test'), 'split must be train|val|test'
        key = split
        if key not in y:
            raise ValueError(f"Missing '{key}' in dataset yaml")
        p = y[key]
        
        # Handle path resolution
        if base is not None and not os.path.isabs(p):
            # If 'path' key exists, use it as base
            p = str(Path(base) / p)
        elif not os.path.isabs(p):
            # If no 'path' key, resolve relative to YAML file location
            p = str(yaml_dir / p)
            
        self.images = list_images(p)
        if len(self.images) == 0:
            raise ValueError(f"No images found for split {split} at {p}")
        labels_root = None
        if 'labels' in y:
            lr = y['labels']
            if base is not None and not os.path.isabs(lr):
                lr = str(Path(base) / lr)
            elif not os.path.isabs(lr):
                lr = str(yaml_dir / lr)
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
    import torch
    images = torch.stack([b['image'] for b in batch], dim=0)
    labels = [b['labels'] for b in batch]
    paths = [b['im_file'] for b in batch]
    return {'images': images, 'labels': labels, 'paths': paths}
