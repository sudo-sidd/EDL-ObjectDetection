 from pathlib import Path
from typing import List, Optional
import yaml
import numpy as np


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
