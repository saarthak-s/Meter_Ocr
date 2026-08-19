# src/meter_reader/ocr_engine.py
import re
import numpy as np
from paddleocr import PaddleOCR
from pathlib import Path

class MeterOCREngine:
    def __init__(self):
        print("Initializing PaddleOCR Engine...")
        # FIX 1: Turn off the angle classifier so it doesn't rotate our crops
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en', enable_mkldnn=False)

    def extract_text(self, image_data: str | Path | np.ndarray) -> str:
        """Runs PaddleOCR on a cropped image and returns the raw text."""
        img_input = str(image_data) if isinstance(image_data, (str, Path)) else image_data
        
        results = self.ocr.ocr(img_input)
        
        if not results or not results[0]:
            return ""
            
        res_dict = results[0]
        
        # FIX 2: Safely extract text from the new dictionary structure
        if isinstance(res_dict, dict) and 'rec_texts' in res_dict and res_dict['rec_texts']:
            # Join all detected text blocks into a single string
            return " ".join(res_dict['rec_texts'])
            
        return ""
    
    def extract_unit(self, raw_text: str) -> str | None:
        """Extracts standard utility units (kWh, kVAh) from the raw OCR text."""
        if not raw_text:
            return None
            
        # 1. Remove all spaces to catch OCR errors like "kw h" or "k v a h"
        no_spaces = raw_text.replace(" ", "")
        
        # 2. Look exactly for kwh or kvah (case-insensitive)
        unit_pattern = r"(?i)(kwh|kvah)"
        match = re.search(unit_pattern, no_spaces)
        
        if match:
            # 3. Standardize the capitalization for the final JSON
            extracted = match.group(1).lower()
            if extracted == "kwh":
                return "kWh"
            elif extracted == "kvah":
                return "kVAh"
                
        return None
    
    def validate_reading(self, raw_text: str) -> float | None:
        """Cleans the raw meter reading text into a strict float."""
        no_spaces = raw_text.replace(" ", "")
        match = re.search(r'(\d+\.?\d*)', no_spaces)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def validate_serial(self, raw_text: str) -> str | None:
        """Extracts the serial number by finding the most likely digit sequence."""
        # Find all isolated blocks of numbers in the raw text
        # For "CAT-C3 46260789", this creates a list: ['3', '46260789']
        numbers = re.findall(r'\d+', raw_text)
        
        # 1. Look for a block that is exactly 8 digits long
        for num in numbers:
            if len(num) == 8:
                return num
                
        # 2. Fallback: Grab the longest number block found
        if numbers:
            longest_num = max(numbers, key=len)
            if len(longest_num) >= 5:
                return longest_num
                
        return None