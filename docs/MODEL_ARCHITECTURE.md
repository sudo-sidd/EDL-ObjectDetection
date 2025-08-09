# Model Architecture (MiniYOLO)

This document explains the model used in EDL: a minimalist, YOLO-style detector built from scratch for clarity.

## High-Level Design

- Single-scale detection head at stride 16 (i.e., output grid S = imgsz/16)
- Output channels per grid cell: `[1 obj, C classes, 4 box]`
- Anchor-free parameterization with sigmoid activations

## Components

1. Backbone
   - A few convolutional blocks with downsampling to reduce spatial size and increase channels.
   - Emphasis on simplicity over accuracy.

2. Neck
   - Lightweight conv layers to mix features.

3. Head
   - Final 1x1 conv to produce `1 + C + 4` channels per spatial location.

## Output Representation

- Objectness (1 channel): probability of an object in the cell.
- Classes (C channels): per-class probabilities (sigmoid), max used for label.
- Box (4 channels): `(dx, dy, pw, ph)` after sigmoid.
  - `dx, dy`: offsets within the grid cell (0..1)
  - `pw, ph`: width and height (0..1)

## Coordinate Reconstruction

For a grid cell `(j, i)` and stride `S`:

- `cx = (i + dx) / S`
- `cy = (j + dy) / S`
- `w = pw`
- `h = ph`

These normalized `xywh` are converted to pixel `xyxy` at `imgsz` using utility conversions.

## Losses

- Objectness: BCEWithLogits
- Classification: BCEWithLogits on positives (multi-label but equivalent to one-hot here)
- Boxes: `1 - IoU` between predicted and target boxes at positive cells

## Target Assignment

- One object per best (by area) cell.
- If multiple objects fall into the same cell, the largest area object is kept.

## Decoding and NMS

- Multiply objectness by class probabilities to get per-class scores.
- Take the best class per location.
- Apply confidence threshold and greedy NMS.

## Notes

- This design aims for readability and minimal dependencies.
- Extend with multi-scale heads, better losses, and augmentations for improved accuracy.
