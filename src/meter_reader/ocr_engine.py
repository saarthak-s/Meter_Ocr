import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel


class MeterOCREngine:
    def __init__(self):
        print("Initializing TBOCR (TrOCR) Engine...")

        # Automatically use Kaggle's GPU if available, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running OCR on device: {self.device}")

        model_id = "microsoft/trocr-base-printed"

        # 1. Image Processor (handles the pixels)
        self.image_processor = AutoImageProcessor.from_pretrained(model_id)

        # 2. Tokenizer (handles the text decoding, explicitly pulling from roberta-base)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # 3. Core Model weights
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device)

    def extract_text(self, image_data: str | Path | np.ndarray) -> str:
        """Runs TrOCR on a cropped image and returns the raw text."""
        # 1. Load the image using OpenCV
        if isinstance(image_data, (str, Path)):
            img_cv = cv2.imread(str(image_data))
        else:
            img_cv = image_data

        if img_cv is None:
            return ""

        # 2. Transformers prefer PIL images in RGB format, not OpenCV's BGR
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # 3. Preprocess the image for the Vision Transformer
        pixel_values = self.image_processor(
            images=pil_img, return_tensors="pt"
        ).pixel_values.to(self.device)

        # 4. Generate the text without calculating gradients (faster inference)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        # 5. Decode the token IDs using the actual Tokenizer, not the Image Processor
        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return text.strip()

    def extract_unit(self, raw_text: str) -> str | None:
        if not raw_text:
            return None
        no_spaces = raw_text.replace(" ", "")
        match = re.search(r"(?i)(kwh|kvah)", no_spaces)
        if match:
            extracted = match.group(1).lower()
            if extracted == "kwh":
                return "kWh"
            elif extracted == "kvah":
                return "kVAh"
        return None

    def validate_reading(self, raw_text: str) -> float | None:
        no_spaces = raw_text.replace(" ", "")
        match = re.search(r"(\d+\.?\d*)", no_spaces)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def validate_serial(self, raw_text: str) -> str | None:
        numbers = re.findall(r"\d+", raw_text)
        for num in numbers:
            if len(num) == 8:
                return num
        if numbers:
            longest_num = max(numbers, key=len)
            if len(longest_num) >= 5:
                return longest_num
        return None
