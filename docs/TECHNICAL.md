# EDL Technical Documentation

This document provides a deeper, implementation-focused view of the EDL object detection package, including module responsibilities, data flows, core algorithms, and extension points.

## Architecture Overview

- Entry point: `main.py`
  - CLI that delegates into the `EDL.engine` APIs for training and prediction.
- Core module: `EDL/engine.py`
  - Training loop, target assignment, loss computation, decoding, prediction utilities, and weight loading.
- Model: `EDL/model.py`
  - Defines the minimal YOLO-style CNN (`MiniYOLO`).
- Data: `EDL/data/dataset.py`
  - Dataset loader (`YOLOTxtDataset`), collate function, and YAML path resolution semantics.
- Utilities: `EDL/utils/*`
  - `vis.py` (drawing with clamped boxes and thickness scaling), `nms.py` (NMS), `geometry.py` (box conversions), `runtime.py` (device, seeds, dirs), `io.py` (optional helpers).

Data flow (high-level):

1. Training: DataLoader -> MiniYOLO forward -> assign_targets -> compute_loss -> optimizer step -> save weights.
2. Prediction: Image/Video -> preprocess -> MiniYOLO forward -> decode_predictions -> scale to original resolution -> clamp -> draw_detections -> optional save -> return in-memory results.

## Core Components

### Model: `MiniYOLO` (`EDL/model.py`)

- Purpose: Minimal detector producing a single feature map (stride 16) with channels: `[1 obj, C classes, 4 box]`.
- Output tensor: shape `[B, 1 + C + 4, S, S]`, where `S = imgsz / 16`.
- Box parameterization (anchor-free):
  - For each cell (j,i), predicts offsets `dx, dy` (relative to cell) and sizes `pw, ph` in [0,1] after sigmoid.

### Target Assignment: `assign_targets`

- Input: A batch of label tensors (per image) in YOLO TXT format: `[class, cx, cy, w, h]` (all normalized).
- Strategy: One object per best (by area) cell. For each object, pick the grid cell `(j,i)` containing its center, but if multiple objects fall into the same cell, keep the larger area object (avoids conflicting supervision).
- Outputs:
  - `obj_t`: `[B, 1, S, S]` binary mask for objectness.
  - `cls_t`: `[B, C, S, S]` one-hot class targets where `obj_t==1`.
  - `box_t`: `[B, 4, S, S]` (cx, cy, w, h) normalized targets.
  - `mask`: `[B, 1, S, S]` boolean selecting positives.

### Loss: `compute_loss`

- Objectness: BCEWithLogits between predicted objectness and `obj_t`.
- Classification: BCEWithLogits over active locations (masked by `mask`). Uses multi-label BCE for simplicity (equivalent to softmax with one-hot in this setting).
- Box regression: Converts predicted offsets to normalized `xywh`, computes IoU with target boxes at positive locations, applies `1 - IoU` loss.
- Total loss: `obj + cls + box` (unit weights). Extension point: reweight or add GIoU/CIoU.

### Decoding: `decode_predictions`

- Inputs: `pred` logits `[B, 1+C+4, S, S]`, `num_classes`, `conf_thres`, `iou_thres`, `img_size`, `max_det`.
- Steps:
  1. Apply sigmoid to obj, cls, and the 4 box channels.
  2. Reconstruct normalized centers: `cx=(i+dx)/S`, `cy=(j+dy)/S`, sizes `w=pw`, `h=ph`.
  3. Convert `xywhn` to pixel `xyxy` with `img_size` using `xywhn_to_xyxy_pixels`.
  4. Compute per-class scores as `obj * cls` and take max over classes to get a single `(score, label)` per location.
  5. Filter by `conf_thres`, apply NMS (`iou_thres`), clamp to `max_det`.
  6. Return list (len B) of tensors `[N, 6]` with columns `[x1,y1,x2,y2,conf,cls]` at the inference resolution.

At prediction time, these detections are scaled back to the original image/frame resolution before visualization and returned values.

### Visualization: `utils/vis.py`

- `draw_detections(image_rgb, det_list, class_names, thickness=None)`:
  - Clamps boxes to image bounds, ensures non-zero area, scales line/text thickness with resolution.
  - Draws rectangles and labels `class:conf` using OpenCV.

## Data & YAML Semantics: `data/dataset.py`

- YAML rules:
  - If `path` exists -> join with `train`, `val`.
  - If `path` missing -> resolve `train`, `val` relative to the YAML file directory (matches YOLO conventions).
- Images discovered by extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`).
- Labels loaded from `labels/<split>/*.txt` with the same basenames as images.
- Collate function returns:
  - `images`: Tensor `[B,3,S,S]` (preprocessed to `imgsz` and normalized to 0..1).
  - `labels`: List of `[K_i,5]` tensors per image with YOLO rows.

## Prediction APIs

### Images: `engine.predict_on_images(args, paths, save_dir)`

- Preprocessing: BGR->RGB, resize to `imgsz`, `[H,W,3]` -> `[1,3,H,W]`, normalize to 0..1.
- Inference: `model.eval()`; `torch.no_grad()`.
- Decoding: `decode_predictions(...)[0]` for single image, then scaled back to original resolution and clamped.
- ID/box mapping:
  - Sort detections by confidence desc for stable numbering.
  - Generate IDs as `<classLabel><index>` (e.g., `apple1`, `apple2`).
  - Return per-image dict `{ id: { bbox:[x1,y1,x2,y2], conf, cls } }`.
- Return value: list of results, each `{'path', 'image' (annotated RGB at original resolution), 'boxes'}`.
- Saving policy: controlled by non-empty `save_dir`; images saved as `<stem>_pred.jpg` and directory auto-created.

### Video: `engine.predict_on_video(args, source, save_dir)`

- Source: path or `0` (webcam).
- Per-frame flow similar to images but writes frames to disk if `save_dir` is provided, with concise per-frame logs.
- Writer: MP4 (`mp4v`), FPS inferred or default 25; output `<stem>_pred.mp4` in `save_dir` (created if needed).
- Return value: stats dict `{ saved_path, frames, total_boxes }`.

## Coordinate Systems

- Training labels: normalized (0..1) `xywh`.
- Decoded predictions: `xyxy` in pixels at the inference resolution, then scaled back to original resolution in the prediction helpers.
- Final outputs and drawings are in the original image/frame resolution and clamped to bounds.
