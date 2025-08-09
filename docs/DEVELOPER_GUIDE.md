# Developer Guide

This guide explains how the EDL package is organized, what each module does, and how to extend or modify it.

## Repository Layout

```text
Custom-ObjectDetection-Model/
├── main.py                 # CLI command center (train/predict)
├── EDL/
│   ├── engine.py           # Training loop, decode, predict helpers, load weights
│   ├── model.py            # MiniYOLO model
│   ├── data/
│   │   └── dataset.py      # YOLO TXT dataset + YAML path resolution
│   └── utils/
│       ├── vis.py          # draw_detections (clamp, thickness scaling)
│       ├── nms.py          # NMS implementation
│       ├── geometry.py     # Box conversions
│       ├── runtime.py      # Device parsing, seeding, dirs
│       └── io.py           # I/O helpers (if present)
└── docs/                   # Documentation
```

## Module Responsibilities

- main.py
  - Parses CLI args, routes to train/predict functions.
  - Handles source discovery (images dir/glob, video, webcam) and save directory selection.

- EDL/engine.py
  - assign_targets: grid target building (one object per best cell)
  - compute_loss: obj/cls/box losses
  - decode_predictions: tensor -> boxes, scores, labels at inference resolution
  - train_loop: training coordinator (saves last.pt/best.pt, meta.json)
  - predict_on_images: preprocessing, inference, decode, scale-back, clamp, draw, save/return
  - predict_on_video: streaming version for video/webcam with per-frame logs
  - load_model: loads ckpt, validates keys, sets safe defaults for meta

- EDL/model.py
  - MiniYOLO: simple CNN with a single detection head (stride 16)

- EDL/data/dataset.py
  - YOLOTxtDataset: resolves train/val/test splits relative to YAML, loads images/labels
  - collate_fn: batches variable-length labels

- EDL/utils/
  - vis.py: draws boxes/labels (clamping, thickness scaling)
  - nms.py: NMS
  - geometry.py: coordinate conversions
  - runtime.py: device parsing (auto/cuda:N), seeding, directory utils
  - io.py: optional helpers for listing images

## Contributing/Extending

- Add features behind minimal, clear APIs and update docs.
- Keep engine helpers stateless where possible; prefer pure functions.
- Validate changes with small scripts or notebooks; add concise logs.

### Ideas to Extend

- Mixed precision training (autocast/GradScaler)
- Multi-scale inference and training
- Improved losses (GIoU/CIoU), Focal loss for obj/cls
- EMA for model weights
- More robust data augmentations

## Checkpoints & Meta

- Saved as dict with keys: 'model' (state_dict) and 'meta' (json-able dict)
- Meta should include: imgsz, stride, names, num_classes
- load_model is tolerant (strict=False) and warns on missing/mismatched fields

## Coding Standards

- Python 3.8+ compatible typing (use Optional[...] instead of |)
- Keep logs concise, avoid noisy prints
- Clamp boxes before drawing to avoid OpenCV errors
- Scale drawing thickness with resolution for consistent visuals

