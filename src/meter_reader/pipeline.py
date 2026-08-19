# src/meter_reader/pipeline.py
import cv2
import json
import logging
import argparse
from pathlib import Path
from ultralytics import YOLO
from src.meter_reader.ocr_engine import MeterOCREngine

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MeterPipeline:
    def __init__(self, yolo_model_path: str):
        logger.info(f"Loading YOLO detector from {yolo_model_path}...")
        if not Path(yolo_model_path).exists():
            logger.error(f"Model weights not found at {yolo_model_path}")
            raise FileNotFoundError(f"Model weights not found at {yolo_model_path}")
            
        self.detector = YOLO(yolo_model_path)
        
        logger.info("Loading OCR Engine...")
        self.ocr_engine = MeterOCREngine()

    def process_image(self, image_path: str | Path) -> dict:
        img_path = str(image_path)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

        try:
            results = self.detector(img, conf=0.25, verbose=False)[0]
        except Exception as e:
            logger.error(f"YOLO inference failed on {img_path}: {e}")
            raise

        if len(results.boxes) == 0:
            logger.warning(f"No detections found in {img_path}")

        extracted_data = {
            "meter_reading": None,
            "raw_meter_reading": None,
            "serial_number": None,
            "raw_serial_number": None,
            "detections": {
                "meter_reading_conf": None,
                "serial_number_conf": None
            }
        }

        for box in results.boxes:
            try:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                h, w, _ = img.shape
                
                # Pad by 5%
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.05)
                x1_pad, y1_pad = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2_pad, y2_pad = min(w, x2 + pad_x), min(h, y2 + pad_y)

                crop = img[y1_pad:y2_pad, x1_pad:x2_pad]

                # OpenCV Preprocessing for OCR
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                contrast_crop = cv2.convertScaleAbs(gray_crop, alpha=1.3, beta=0)
                padded_crop = cv2.copyMakeBorder(
                    contrast_crop, 15, 15, 15, 15, 
                    cv2.BORDER_CONSTANT, value=255
                )

                # Ensure temp crop names are unique to prevent collisions in multi-threading or loops
                temp_crop_path = Path(f"temp_crop_{Path(img_path).name}.png")
                cv2.imwrite(str(temp_crop_path), padded_crop)

                if cls_id == 0:
                    raw_text = self.ocr_engine.extract_text(temp_crop_path)
                    extracted_data["raw_meter_reading"] = raw_text
                    extracted_data["meter_reading"] = self.ocr_engine.validate_reading(raw_text)
                    extracted_data["detections"]["meter_reading_conf"] = round(conf, 4)

                elif cls_id == 1:
                    raw_text = self.ocr_engine.extract_text(temp_crop_path)
                    extracted_data["raw_serial_number"] = raw_text
                    extracted_data["serial_number"] = self.ocr_engine.validate_serial(raw_text)
                    extracted_data["detections"]["serial_number_conf"] = round(conf, 4)

                # Clean up the temp file
                if temp_crop_path.exists():
                    temp_crop_path.unlink()
                    
            except Exception as e:
                logger.error(f"Failed to process bounding box for {img_path}: {e}")
                # We continue to the next box instead of crashing the whole image
                continue

        return extracted_data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Meter OCR pipeline on a directory of images.")
    p.add_argument("--model", default="models/best.pt", help="Path to the trained YOLO weights")
    p.add_argument("--input-dir", default="dataset/processed/images/val", help="Directory containing images to process")
    p.add_argument("--output", default="batch_results.json", help="Path to save the output JSON results")
    p.add_argument("--failures", default="failures.json", help="Path to save the failed filenames")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    
    try:
        pipeline = MeterPipeline(yolo_model_path=args.model)
    except Exception as e:
        logger.critical(f"Pipeline initialization failed: {e}")
        exit(1)

    test_image_dir = Path(args.input_dir)
    sample_images = list(test_image_dir.glob("*.jpg"))

    if not sample_images:
        logger.warning(f"No test images found in {test_image_dir.absolute()}")
    else:
        logger.info(f"Found {len(sample_images)} images. Starting Batch Processing...")
        
        all_results = []
        failures = []
        
        for img_path in sample_images:
            logger.info(f"Processing {img_path.name}...")
            try:
                result = pipeline.process_image(img_path)
                result["filename"] = img_path.name 
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}", exc_info=False)
                failures.append({
                    "filename": img_path.name,
                    "error": str(e)
                })

        # Save successful results
        output_file = Path(args.output)
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=4)
            
        # Save failures if any occurred
        failures_file = Path(args.failures)
        if failures:
            with open(failures_file, "w") as f:
                json.dump(failures, f, indent=4)
            logger.warning(f"Batch completed with {len(failures)} failures. Details saved to {failures_file.absolute()}")
        else:
            logger.info(f"Batch complete! 0 failures. All results saved to {output_file.absolute()}")