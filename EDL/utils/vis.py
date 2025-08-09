from typing import List, Tuple, Optional
import numpy as np
import cv2


def draw_detections(img: np.ndarray, dets: List[Tuple[float, float, float, float, float, int]], class_names: List[str], thickness: Optional[int] = None):
    # Determine line/text thickness
    if thickness is None:
        # Scale with image size if not provided
        thickness = max(1, int(round(min(img.shape[0], img.shape[1]) / 640)))
    text_thickness = max(1, thickness - 1)
    font_scale = 0.5 * (thickness / 2)

    h, w = img.shape[:2]
    w1, h1 = w - 1, h - 1

    for x1, y1, x2, y2, conf, cls in dets:
        # Clamp to image bounds
        x1 = max(0.0, min(float(x1), w1))
        y1 = max(0.0, min(float(y1), h1))
        x2 = max(0.0, min(float(x2), w1))
        y2 = max(0.0, min(float(y2), h1))
        # Ensure non-zero area for visibility
        if x2 <= x1:
            x2 = min(w1, x1 + 1)
        if y2 <= y1:
            y2 = min(h1, y1 + 1)

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        color = (0, 255, 0)

        cv2.rectangle(img, p1, p2, color, thickness)

        # Label
        name = class_names[cls] if 0 <= int(cls) < len(class_names) else str(int(cls))
        label = f"{name}:{conf:.2f}"
        tx, ty = p1[0], max(0, p1[1] - 5)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)

    return img
