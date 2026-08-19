# src/meter_reader/train.py
import argparse
from ultralytics import YOLO

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the YOLOv8 meter detector.")
    p.add_argument("--model", default="yolov8n.pt", help="Base YOLO model to start training from")
    p.add_argument("--data", default="dataset.yaml", help="Path to the dataset configuration file")
    p.add_argument("--epochs", type=int, default=120, help="Maximum number of training epochs")
    p.add_argument("--batch", type=int, default=16, help="Training batch size")
    p.add_argument("--project", default="runs/detect", help="Project directory to save results")
    p.add_argument("--name", default="meter_detector_prod", help="Experiment name for this training run")
    p.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    
    print(f"Initializing YOLO training with {args.model}...")
    model = YOLO(args.model)
    
    print(f"Starting training for {args.epochs} epochs on {args.data}...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=args.patience,
        device="cpu" # Remove or set to 0 if running on a GPU machine later
    )
    
    print("\n✅ Training complete!")