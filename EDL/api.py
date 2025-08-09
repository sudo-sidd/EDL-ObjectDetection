from types import SimpleNamespace
from pathlib import Path
import os
import glob
from typing import Any, Dict, List, Sequence, Union

from .engine import predict_on_images, predict_on_video


class Detector:
    """High-level EDL detector API (YOLO-like).

    Usage:
        from EDL import Detector
        det = Detector(weights='path/to/best.pt', device='auto', imgsz=640, conf=0.25, iou=0.45)
        results = det.pred(source='images/', save_results='runs/predict')
    """

    def __init__(
        self,
        weights: str,
        device: str = 'auto',
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 100,
    ) -> None:
        self.weights = weights
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.num_classes = None  # auto from weights

    def _build_args(self, **overrides: Any) -> SimpleNamespace:
        p = {
            'weights': self.weights,
            'device': self.device,
            'imgsz': self.imgsz,
            'conf': self.conf,
            'iou': self.iou,
            'max_det': self.max_det,
            'num_classes': self.num_classes,
        }
        p.update({k: v for k, v in overrides.items() if v is not None})
        return SimpleNamespace(**p)

    def predict(
        self,
        source: Union[str, int, Sequence[str]],
        save_results: Union[bool, str, None] = None,
        *,
        conf: float = None,
        iou: float = None,
        imgsz: int = None,
        max_det: int = None,
    ):
        """Run prediction on images, directory, glob, video file, or webcam.

        - source: path to image/dir/glob/video, 'webcam', 0, or list of image paths
        - save_results: True to save to runs/predict, a string path to save there, or None/False to not save
        - conf/iou/imgsz/max_det: optional overrides
        Returns list of image results or video stats dict.
        """
        args = self._build_args(conf=conf, iou=iou, imgsz=imgsz, max_det=max_det)

        # Determine save directory
        if isinstance(save_results, str) and save_results:
            save_dir = save_results
        elif save_results is True:
            save_dir = 'runs/predict'
        else:
            save_dir = ''

        # Sequence of image paths
        if isinstance(source, (list, tuple)):
            paths = list(source)
            if not paths:
                raise FileNotFoundError('Empty source path list')
            return predict_on_images(args, paths, save_dir)

        # String or int source
        if isinstance(source, int):
            return predict_on_video(args, source, save_dir)
        if not isinstance(source, str):
            raise TypeError(f'Unsupported source type: {type(source)}')

        s = source
        sl = s.lower()
        if sl == 'webcam':
            return predict_on_video(args, 0, save_dir)

        # Video file
        if sl.endswith(('.mp4', '.avi', '.mov', '.mkv')) and os.path.isfile(s):
            return predict_on_video(args, s, save_dir)

        # Images: directory / glob / single file
        paths: List[str] = []
        if os.path.isdir(s):
            patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for pat in patterns:
                paths.extend(sorted(glob.glob(os.path.join(s, pat))))
        elif any(ch in s for ch in ['*', '?', '[']):
            paths = sorted(glob.glob(s))
        elif os.path.isfile(s):
            paths = [s]
        else:
            raise FileNotFoundError(f'Source not found: {source}')

        if not paths:
            raise FileNotFoundError(f'No images found for source: {source}')

        return predict_on_images(args, paths, save_dir)

    # alias
    def pred(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
