# scripts/visualize_results.py
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

def generate_visuals(model_path: str, image_paths: list, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading YOLO model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"Skipping {img_path.name} - file not found.")
            continue
            
        print(f"Processing {img_path.name}...")
        img = cv2.imread(str(img_path))
        annotated_img = img.copy()
        
        results = model(img, conf=0.25, verbose=False)[0]
        
        for idx, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            label = "Reading" if cls_id == 0 else "Serial"
            color = (0, 255, 0) if cls_id == 0 else (255, 165, 0) # Green for reading, Orange for serial
            
            # Draw on full image
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Extract and preprocess crop (matching pipeline logic)
            h, w, _ = img.shape
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            x1_pad, y1_pad = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2_pad, y2_pad = min(w, x2 + pad_x), min(h, y2 + pad_y)
            
            crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            contrast_crop = cv2.convertScaleAbs(gray_crop, alpha=1.3, beta=0)
            padded_crop = cv2.copyMakeBorder(contrast_crop, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
            
            # Save the preprocessed crop
            crop_out = out_dir / f"{img_path.stem}_crop_{label.lower()}.jpg"
            cv2.imwrite(str(crop_out), padded_crop)
            
        # Save the fully annotated image
        full_out = out_dir / f"{img_path.stem}_annotated.jpg"
        cv2.imwrite(str(full_out), annotated_img)
        print(f"Saved visuals for {img_path.name} to {out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visual examples for the README.")
    parser.add_argument("--model", default="models/best.pt", help="Path to YOLO weights")
    parser.add_argument("--out-dir", default="docs/examples", help="Output directory for visuals")
    # You can pass specific test images via CLI, or use these defaults
    parser.add_argument("--images", nargs="+", default=[
        "dataset/processed/images/val/000103049033.jpg", # Known success case
        "dataset/processed/images/val/000103154739.jpg"  # Known glare failure case
    ], help="List of images to process")
    
    args = parser.parse_args()
    generate_visuals(args.model, args.images, args.out_dir)