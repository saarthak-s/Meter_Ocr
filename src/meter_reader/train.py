# src/meter_reader/train.py
from ultralytics import YOLO

def main():
    print("Loading lightweight YOLOv8 nano model...")
    model = YOLO('yolov8n.pt')

    print("Starting production training on the complete dataset...")
    results = model.train(
        data='dataset.yaml', 
        epochs=120,        
        imgsz=640,       
        batch=16,         
        name='meter_detector_prod', 
        device='cpu',
        patience=10     # Stop if no improvement after 10 epochs
    )
    
    print("\n--- Production Training Complete! ---")
    print("Your production model is saved at: runs/detect/meter_detector_prod/weights/best.pt")

if __name__ == "__main__":
    main()