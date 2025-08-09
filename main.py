#!/usr/bin/env python3
"""
EDL Object Detection - Command Center
Single entry point for all EDL operations
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path for EDL imports
sys.path.insert(0, str(Path(__file__).parent))

from EDL.cli import parse_train_args, parse_predict_args
from EDL.engine import train_loop, predict_on_images, predict_on_video


def create_parser():
    """Create the main argument parser with subcommands"""
    parser = argparse.ArgumentParser(
        prog='python main.py',
        description='EDL Object Detection - Command Center',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --data data.yaml --epochs 100
  python main.py predict --weights runs/train/best.pt --source image.jpg
  python main.py predict --weights runs/train/best.pt --source video.mp4
  python main.py predict --weights runs/train/best.pt --source webcam
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    train_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    train_parser.add_argument('--device', type=str, default='auto', help='Device (cpu, cuda, auto)')
    train_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    train_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    train_parser.add_argument('--out', type=str, default='runs/train', help='Output directory')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    train_parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Run predictions')
    predict_parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    predict_parser.add_argument('--source', type=str, required=True, help='Source (image, video, webcam)')
    predict_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    predict_parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    predict_parser.add_argument('--max-det', type=int, default=100, help='Maximum detections')
    predict_parser.add_argument('--device', type=str, default='auto', help='Device (cpu, cuda, auto)')
    predict_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    predict_parser.add_argument('--save-dir', type=str, default='runs/predict', help='Save directory')
    
    # Help command
    help_parser = subparsers.add_parser('help', help='Show detailed help')
    
    return parser


def show_help():
    """Show detailed help information"""
    print("""
🚀 EDL Object Detection Framework
=================================

COMMANDS:
  train     Train a new model
  predict   Run predictions on images/videos
  help      Show this help

TRAINING:
  python main.py train --data data.yaml --epochs 100 --batch 16
  
  Required:
    --data        Dataset YAML file path
  
  Optional:
    --epochs      Number of training epochs (default: 100)
    --batch       Batch size (default: 16)
    --imgsz       Image size (default: 640)
    --device      Device to use: cpu, cuda, auto (default: auto)
    --lr          Learning rate (default: 1e-3)
    --workers     Number of data loading workers (default: 4)
    --out         Output directory (default: runs/train)
    --seed        Random seed (default: 42)
    --log-interval Training log interval (default: 10)

PREDICTION:
  python main.py predict --weights model.pt --source image.jpg
  python main.py predict --weights model.pt --source video.mp4
  python main.py predict --weights model.pt --source webcam
  
  Required:
    --weights     Path to model weights (.pt file)
    --source      Source: image file, video file, or 'webcam'
  
  Optional:
    --conf        Confidence threshold (default: 0.25)
    --iou         IoU threshold for NMS (default: 0.45)
    --max-det     Maximum detections per image (default: 100)
    --device      Device to use: cpu, cuda, auto (default: auto)
    --imgsz       Image size (default: 640)
    --save-dir    Save directory (default: runs/predict)

EXAMPLES:
  # Train on custom dataset
  python main.py train --data my_dataset.yaml --epochs 50 --batch 8
  
  # Predict on single image
  python main.py predict --weights runs/train/best.pt --source test.jpg
  
  # Predict on video with custom settings
  python main.py predict --weights model.pt --source video.mp4 --conf 0.5 --iou 0.4
  
  # Real-time webcam detection
  python main.py predict --weights model.pt --source webcam

DATASET FORMAT:
  Your data.yaml should contain:
    path: /path/to/dataset
    train: images/train
    val: images/val
    names: ['class1', 'class2', ...]
    
  Directory structure:
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
""")


def main():
    """Main command center entry point"""
    parser = create_parser()
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        print("🚀 EDL Object Detection Framework")
        print("Use 'python main.py help' for detailed information")
        print("Use 'python main.py -h' for quick help")
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    # Route commands
    if args.command == 'train':
        print("🚀 Starting EDL Training...")
        args.num_classes = None  # Will be auto-detected from dataset
        train_loop(args)
        
    elif args.command == 'predict':
        print("🚀 Starting EDL Prediction...")
        args.num_classes = None  # Will be loaded from weights
        
        # Handle different source types
        source = args.source.lower()
        if source == 'webcam':
            predict_on_video(args, 0, args.save_dir)  # 0 for webcam
        elif source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            predict_on_video(args, args.source, args.save_dir)
        else:
            # Assume it's an image or directory of images
            predict_on_images(args, [args.source], args.save_dir)
            
    elif args.command == 'help':
        show_help()
        
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()
