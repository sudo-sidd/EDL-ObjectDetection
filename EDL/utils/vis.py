# comment instructions only
from typing import List, Tuple
import numpy as np
import cv2

def draw_detections(img: np.ndarray, dets: List[Tuple[float, float, float, float, float, int]], class_names: List[str]):
    for x1, y1, x2, y2, conf, cls in dets:
        color = (0, 255, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_names[cls] if 0 <= cls < len(class_names) else cls}:{conf:.2f}"
        cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img
