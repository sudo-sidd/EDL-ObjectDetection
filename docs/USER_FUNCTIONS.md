# User-Facing Functions

This guide documents the functions you will typically use, along with parameters and return values. Helper/low-level internals are not listed here.

## High-Level Inference API

### Detector (class)

Constructor:

- weights: str — path to weights .pt
- device: str — 'auto' | 'cpu' | 'cuda' | 'cuda:N'
- imgsz: int — inference size (default 640)
- conf: float — confidence threshold (default 0.25)
- iou: float — IoU threshold for NMS (default 0.45)
- max_det: int — max detections per image/frame (default 100)

Methods:

- pred(source, save_results=None, conf=None, iou=None, imgsz=None, max_det=None)
  - Args:
    - source: str | int | Sequence[str] (image path, directory, glob, video path, 'webcam', or 0)
    - save_results: None/False (no save), True (save to runs/predict), or str (directory)
    - conf/iou/imgsz/max_det: optional overrides for this call
  - Returns:
    - If images: List[Dict] (for each image)
      - 'path': str
      - 'image': numpy.ndarray (RGB, HxWx3 at original image resolution)
      - 'boxes': Dict[str, { 'bbox':[x1,y1,x2,y2], 'conf': float, 'cls': str }]
    - If video/webcam: Dict
      - 'saved_path': Optional[str]
      - 'frames': int
      - 'total_boxes': int

## Engine Utilities (direct usage)

These functions power the CLI and API; you can call them directly if preferred.

### predict_on_images(args, paths, save_dir)

- args: SimpleNamespace with fields: weights, device, imgsz, conf, iou, max_det, num_classes
- paths: List[str] — image paths
- save_dir: str — '' (no save) or directory path to save annotated images
- Returns: List[Dict] (same schema as Detector image results)

### predict_on_video(args, source, save_dir)

- args: SimpleNamespace like above
- source: str | int — path to video or 0/webcam
- save_dir: str — '' (no save) or directory path to save annotated mp4
- Returns: Dict with keys { 'saved_path', 'frames', 'total_boxes' }

## CLI Commands (overview)

Use `python main.py help` or see docs/API_AND_CLI.md for details.
