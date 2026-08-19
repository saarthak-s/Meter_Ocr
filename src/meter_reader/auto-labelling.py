# src/meter_reader/auto-labelling.py
import argparse
from pathlib import Path
from ultralytics import YOLO

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto-label images using a draft YOLO model.")
    p.add_argument("--draft-model", default="runs/detect/meter_detector-2/weights/best.pt", help="Path to the draft YOLO weights")
    p.add_argument("--images-dir", default="dataset/images", help="Directory containing unlabeled images")
    p.add_argument("--labels-dir", default="dataset/labels", help="Directory to save generated YOLO labels")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for auto-labeling")
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    
    img_dir = Path(args.images_dir)
    lbl_dir = Path(args.labels_dir)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading draft model from {args.draft_model}...")
    try:
        model = YOLO(args.draft_model)
    except FileNotFoundError:
        print(f"❌ Error: Draft model not found at {args.draft_model}")
        print("Please train a draft model first or specify the correct path using --draft-model")
        exit(1)

    images = list(img_dir.glob("*.jpg"))
    if not images:
        print(f"No images found in {img_dir.absolute()}")
        exit(0)

    print(f"Starting auto-labeling for {len(images)} images...")
    processed = 0
    skipped = 0

    for img_path in images:
        label_file = lbl_dir / f"{img_path.stem}.txt"
        
        # Skip if the human label already exists
        if label_file.exists():
            skipped += 1
            continue
            
        results = model(str(img_path), conf=args.conf, verbose=False)[0]
        
        if len(results.boxes) > 0:
            with open(label_file, "w") as f:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    # Get normalized xywh
                    x_c, y_c, w, h = box.xywhn[0].tolist() 
                    f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            processed += 1

    print(f"\n✅ Auto-labeling complete!")
    print(f"   Generated labels: {processed}")
    print(f"   Skipped (already labeled): {skipped}")