# src/meter_reader/auto_label.py
from ultralytics import YOLO
from pathlib import Path

def main():
    # Load the draft model you just finished training
    model = YOLO('runs/detect/meter_detector-2/weights/best.pt')

    # Point directly to your dataset folders
    image_dir = Path("dataset/images")
    label_dir = Path("dataset/labels")
    
    # Ensure the labels directory exists
    label_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    # Loop through all jpg and png files
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"

        # Skip images you already manually labeled (protects your 50 images!)
        if label_path.exists():
            continue

        print(f"Predicting boxes for {img_path.name}...")
        results = model(img_path)[0]
        
        # Write coordinates to YOLO format
        with open(label_path, "w") as f:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                x_c, y_c, w, h = box.xywhn[0].tolist() 
                f.write(f"{cls_id} {x_c} {y_c} {w} {h}\n")
        
        count += 1

    print(f"\nAuto-labeling complete! Generated boxes for {count} remaining images.")

if __name__ == "__main__":
    main()