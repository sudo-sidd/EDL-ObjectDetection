import time
import json
from pathlib import Path
from typing import Optional

import torch
import cv2
import numpy as np

from .model import MiniYOLO
from .utils import make_dir, now_str, parse_device


class ModelDebugger:
    """Built-in debugging utilities for EDL models"""
    
    @staticmethod
    def test_model_forward(model, device, input_shape=(1, 3, 640, 640)):
        """Test model forward pass with dummy data"""
        dummy_input = torch.randn(*input_shape).to(device)
        model.eval()
        with torch.no_grad():
            try:
                output = model(dummy_input)
                return True, output.shape
            except Exception as e:
                return False, str(e)
    
    @staticmethod
    def benchmark_model(model, device, num_runs=50, input_shape=(1, 3, 640, 640)):
        """Benchmark model inference speed"""
        model.eval()
        dummy_input = torch.randn(*input_shape).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Benchmark
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        avg_time = total_time / num_runs
        fps = 1.0 / avg_time
        
        return avg_time, fps


class TestDataGenerator:
    """Built-in test data generation utilities"""
    
    @staticmethod
    def create_sample_image(size=(640, 640), add_objects=True):
        """Create a test image with optional geometric objects"""
        img = np.random.randint(50, 200, (*size, 3), dtype=np.uint8)
        
        if add_objects:
            h, w = size
            # Add rectangles
            for _ in range(np.random.randint(1, 4)):
                x1 = np.random.randint(50, w//2)
                y1 = np.random.randint(50, h//2)
                x2 = min(x1 + np.random.randint(50, 200), w-50)
                y2 = min(y1 + np.random.randint(50, 200), h-50)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
            # Add circles
            for _ in range(np.random.randint(0, 3)):
                center = (np.random.randint(100, w-100), np.random.randint(100, h-100))
                radius = np.random.randint(30, 80)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.circle(img, center, radius, color, -1)
        
        return img
    
    @staticmethod
    def create_test_dataset(output_dir, num_train=20, num_val=10, img_size=640):
        """Create a complete test dataset with images and labels"""
        base_path = Path(output_dir)
        
        # Create directory structure
        for split in ['train', 'val']:
            make_dir(str(base_path / 'images' / split))
            make_dir(str(base_path / 'labels' / split))
        
        # Generate data
        for split, num_samples in [('train', num_train), ('val', num_val)]:
            for i in range(num_samples):
                # Create image
                img = TestDataGenerator.create_sample_image((img_size, img_size))
                img_path = base_path / 'images' / split / f'sample_{i:03d}.jpg'
                cv2.imwrite(str(img_path), img)
                
                # Create label
                num_objects = np.random.randint(1, 6)
                labels = []
                for _ in range(num_objects):
                    cls = 0  # single class
                    cx = np.random.uniform(0.1, 0.9)
                    cy = np.random.uniform(0.1, 0.9)
                    w = np.random.uniform(0.05, 0.3)
                    h = np.random.uniform(0.05, 0.3)
                    labels.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                
                label_path = base_path / 'labels' / split / f'sample_{i:03d}.txt'
                with open(label_path, 'w') as f:
                    f.write('\n'.join(labels))
        
        # Create dataset YAML
        yaml_config = {
            'path': str(base_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': ['person']
        }
        
        yaml_path = base_path / 'data.yaml'
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        return str(yaml_path)


class ModelManager:
    """Built-in model management utilities"""
    
    @staticmethod
    def create_dummy_model(save_path, num_classes=1, img_size=640):
        """Create and save a dummy model for testing"""
        model = MiniYOLO(num_classes=num_classes)
        meta = {
            'imgsz': img_size,
            'stride': 16,
            'names': ['person'] * num_classes,
            'num_classes': num_classes,
        }
        
        make_dir(str(Path(save_path).parent))
        torch.save({'model': model.state_dict(), 'meta': meta}, save_path)
        return save_path
    
    @staticmethod
    def find_best_weights(search_paths=None):
        """Find the best available model weights"""
        if search_paths is None:
            search_paths = [
                "runs/train/best.pt",
                "runs/train/last.pt",
                "runs/test_train/test_model.pt",
                "runs/test_full_train/weights/best.pt",
                "runs/test_full_train/weights/last.pt"
            ]
        
        for path in search_paths:
            if Path(path).exists():
                return path
        
        # Create dummy if none found
        dummy_path = "temp_weights/dummy_model.pt"
        return ModelManager.create_dummy_model(dummy_path)



