# EDL: Minimal Object Detection in PyTorch

EDL stands for Enna Da Looku 

A modular, from-scratch object detection framework inspired by YOLO. Includes a tiny model, training loop, decoding + NMS, a clean CLI, and simple programmatic APIs.

## Features

- Clean, modular package layout (`EDL/`)
- Single command center (`main.py`) for training and prediction
- Standard YOLO dataset format (images + labels with TXT) and YAML config
- Prediction returns in-memory results; saving is opt-in with `--save_results`
- Progress bars and configuration banner during training
- Concise per-image/per-frame prediction logs and auto device selection (`--device auto`)
- Boxes are scaled back to original resolution, clamped to image bounds; drawing thickness scales with resolution

## Requirements

- Python 3.8+
- PyTorch (GPU optional)
- OpenCV (opencv-python)
- NumPy
- tqdm
- PyYAML (for dataset yaml)

Example installation:

- pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
- pip install opencv-python numpy tqdm pyyaml

## Project Structure

```text
Custom-ObjectDetection-Model/
├── main.py                      # Command center (train/predict)
├── EDL/
│   ├── engine.py               # Train loop, predict, decode, load weights
│   ├── model.py                # MiniYOLO model definition
│   ├── data/
│   │   └── dataset.py          # YOLO TXT dataset loader, YAML path resolution
│   └── utils/                  # Helpers: NMS, drawing, geometry, runtime, etc.
│       ├── vis.py              # draw_detections (clamped boxes, thickness scaling)
│       ├── nms.py              # NMS implementation
│       ├── geometry.py         # xywh/xyxy conversions
│       ├── runtime.py          # device parsing, seeding, dirs
│       └── io.py               # image listing (if present)
└── README.md                   # This document
```

## Documentation

- docs/API_AND_CLI.md — API and CLI usage, arguments, return types.
- docs/USER_FUNCTIONS.md — User-facing functions, parameters, outputs.
- docs/MODEL_ARCHITECTURE.md — Model design and math details.
- docs/DEVELOPER_GUIDE.md — Package modules, roles, file-by-file guide.
- docs/TECHNICAL.md — Technical overview and internals.

## Quickstart

1. Prepare a dataset in YOLO format and a data YAML (see Dataset section).
2. Train:
   - python main.py train --data path/to/data.yaml --epochs 100 --batch 16
3. Predict on images (no saving by default):
   - python main.py predict --weights runs/train/best.pt --source path/to/images/
4. Save predictions:
   - python main.py predict --weights runs/train/best.pt --source image_or_folder --save_results
   - python main.py predict --weights runs/train/best.pt --source video.mp4 --save_results results

Notes:

- When `--save_results` is provided without a value, outputs go to `runs/predict`.
- When `--save_results <dir>` is provided, outputs go to `<dir>` (it is created if missing).

## Dataset Format (YOLO)

- data.yaml example:

```yaml
path: /abs/or/relative/base  # optional; if omitted, splits are resolved relative to this yaml's dir
train: images/train
val: images/val
names: ["apple"]
```

- Directory layout:

```text
dataset/
├── images/
│   ├── train/  (*.jpg, *.png, ...)
│   └── val/
└── labels/
    ├── train/  (*.txt, same basename as images)
    └── val/
```

- Label TXT lines: `class x_center y_center width height` in normalized [0,1].
- Path resolution:
  - If `path` exists in YAML, it is used as base for `train`/`val`.
  - If `path` is omitted, paths are resolved relative to the folder containing the YAML file.

## Training

Example:

- python main.py train --data data.yaml --epochs 100 --batch 16 --imgsz 640 --device auto

What happens:

- Loads `YOLOTxtDataset` with YAML semantics above
- Builds `MiniYOLO` with `num_classes` inferred from the dataset
- Shows a configuration banner (device, classes, imgsz, etc.)
- Runs a simple training loop with progress bars
- Saves `last.pt` and `best.pt` under `runs/train/.../weights` and copies convenience links to `runs/train/`

Useful flags:

- --epochs N, --batch B, --imgsz S, --device cpu|cuda|auto, --lr, --workers

## Prediction (CLI)

Base command:

- python main.py predict --weights runs/train/best.pt --source <image|folder|glob|video|webcam>

Key flags:

- --conf 0.25           Confidence threshold
- --iou 0.45            IoU for NMS
- --imgsz 640           Inference image size
- --max-det 100         Max detections per image
- --save_results [DIR]  Save annotated outputs; defaults to `runs/predict` when DIR omitted

Sources:

- Single image file, a directory of images, a glob (e.g. "images/*.jpg"), a video file (.mp4, .avi, ...), or `webcam`.

Saving behavior:

- By default, nothing is saved. Results are returned in memory from the engine.
- When `--save_results` is specified, images are saved as `<stem>_pred.jpg`, and video as `<stem>_pred.mp4` to the chosen directory (created if missing).

## Programmatic API

Use the engine functions directly from Python/Notebooks.

Example:

```python
from types import SimpleNamespace
from EDL.engine import predict_on_images

args = SimpleNamespace(
    weights='runs/train/best.pt',
    conf=0.25,
    iou=0.45,
    max_det=100,
    device='auto',
    imgsz=640,
)
results = predict_on_images(args, ['test.jpg'], save_dir='')

# Each result:
# {
#   'path': 'test.jpg',
#   'image': <H x W x 3 RGB numpy array at original resolution>,
#   'boxes': {
#       'apple1': {'bbox': [x1,y1,x2,y2], 'conf': 0.87, 'cls': 'apple'},
#       'apple2': {...},
#    }
# }
annotated = results[0]['image']
boxes = results[0]['boxes']
```

Notes:

- Returned coordinates and annotated image are at the original image resolution. Boxes are clamped to image bounds; drawing thickness scales with resolution.

## Internals Overview

- Model (`EDL/model.py`): Tiny CNN backbone/neck/head producing per-cell predictions with stride 16.
- Target Assignment (`assign_targets` in `engine.py`): One object per best (area) cell.
- Loss (`compute_loss` in `engine.py`): BCE for objectness, BCE for per-class, (1 - IoU) for boxes.
- Decoding (`decode_predictions` in `engine.py`): Sigmoid offsets + grid to get normalized xywh; converts to xyxy pixels; multiplies obj × class scores; filters by `conf`; applies NMS; returns `[x1,y1,x2,y2,conf,cls]` per image.
- Visualization (`EDL/utils/vis.py`): `draw_detections(image_rgb, det_list, names)` draws boxes and `class:conf` labels with clamping and thickness scaling.

## CLI Reference

Training:

- python main.py train --data DATA.YAML [--epochs N] [--batch B] [--imgsz S] [--device auto] [--lr 1e-3] [--workers 4] [--out runs/train]

Prediction:

- python main.py predict --weights WEIGHTS.PT --source SRC [--conf 0.25] [--iou 0.45] [--imgsz 640] [--max-det 100] [--save_results [DIR]]

## Troubleshooting

- Invalid images path:
  - Ensure your YAML follows the format above. Relative paths are resolved relative to the YAML file directory when `path` is not provided.
- CUDA out of memory:
  - Reduce `--imgsz`, `--batch`; close other GPU apps; consider `--device cpu`; clear CUDA cache between runs.
- No boxes saved:
  - Check `--conf` threshold; ensure `--save_results` is used when you want files written.

## Notes

- This project is minimal and intended for learning and small experiments, not SOTA accuracy.
- Extend as needed: add augmentations, multi-scale, anchor variants, better loss, etc.
