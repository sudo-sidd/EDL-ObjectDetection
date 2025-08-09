# API and CLI Usage

This guide covers how to use EDL programmatically and via the command line.

## Programmatic API

### Detector

High-level API for inference.

```python
from EDL import Detector

# Initialize with defaults
model = Detector(weights='runs/train/best.pt', device='auto', imgsz=640, conf=0.25, iou=0.45)

# Predict on images (list/dir/glob/single)
results = model.pred(source='images/*.jpg', save_results=None)
# -> List of dicts with keys: 'path', 'image' (RGB np.ndarray at original resolution), 'boxes' dict

# Predict on video or webcam
stats = model.pred(source='video.mp4', save_results='runs/predict')
# -> Dict with keys: 'saved_path', 'frames', 'total_boxes'
```

Arguments (constructor):

- weights: str, path to .pt weights
- device: 'auto' | 'cpu' | 'cuda' | 'cuda:N'
- imgsz: int, inference size
- conf: float, confidence threshold
- iou: float, NMS IoU threshold
- max_det: int, max detections per image/frame

predict()/pred() parameters:

- source: str | int | Sequence[str] (image path, directory, glob, video path, 'webcam', or 0 for webcam)
- save_results: None/False (no save), True (save to runs/predict), or str (directory)
- conf/iou/imgsz/max_det: optional per-call overrides

Return types:

- Image mode: List[Dict]
  - 'path': str
  - 'image': numpy.ndarray (RGB, HxWx3, original resolution)
  - 'boxes': Dict[str, { 'bbox': [x1,y1,x2,y2], 'conf': float, 'cls': str }]
- Video/Webcam: Dict
  - 'saved_path': Optional[str]
  - 'frames': int
  - 'total_boxes': int

## CLI

Single entry point: `python main.py`

### Train

```bash
python main.py train --data data.yaml --epochs 100 --batch 16 --imgsz 640 --device auto
```

Key flags:

- --data, --epochs, --batch, --imgsz, --device, --lr, --workers, --out, --seed

### Predict

```bash
# Images (no save)
python main.py predict --weights runs/train/best.pt --source images/

# Images (save to default)
python main.py predict --weights runs/train/best.pt --source images/ --save_results

# Video
python main.py predict --weights runs/train/best.pt --source video.mp4 --save_results results

# Webcam
python main.py predict --weights runs/train/best.pt --source webcam --save_results
```

Key flags:

- --weights, --source, --conf, --iou, --max-det, --device, --imgsz, --save_results [DIR]

Notes:

- Saving is opt-in; when provided without a directory, outputs go to `runs/predict`.
- Per-image/per-frame concise logs are printed.
