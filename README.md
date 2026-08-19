# Meter OCR

Automated pipeline for reading utility meters from photos. A YOLOv8 model
locates the **meter reading** and **serial number** regions on a meter
image, each region is cropped and preprocessed, and
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) extracts the text,
which is then parsed into a clean numeric reading and serial number.

## How it works

1. **Detect** — a YOLOv8 model (`ultralytics`) finds two classes of
   bounding boxes on the input image:
   - `0` — `meter_reading`
   - `1` — `serial_number`
2. **Crop & preprocess** — each detected box is padded, converted to
   grayscale, contrast-boosted, and given a white border so PaddleOCR can
   read faded or edge-cut digits more reliably.
3. **Recognize** — PaddleOCR runs on each crop to produce raw text.
4. **Validate** — regex-based post-processing turns the raw OCR text into
   a clean `float` meter reading and a clean serial-number string (the
   engine prefers an 8-digit block, falling back to the longest digit
   run of 5+ characters).
5. **Output** — results (reading, serial number, raw OCR text, and
   detection confidences) are collected into a JSON file.

## Project structure

```
Meter_Ocr/
├── dataset.yaml                    # YOLO dataset config (classes, paths)
├── pyproject.toml                  # Project metadata & dependencies
├── requirements.txt                # Full environment dependency freeze
└── src/meter_reader/
    ├── __init__.py
    ├── split_data.py                # Train/val split for the labeled dataset
    ├── auto-labelling.py            # Semi-automated labeling with a draft model
    ├── train.py                     # YOLOv8 training script
    ├── ocr_engine.py                # PaddleOCR wrapper + text validation
    └── pipeline.py                  # End-to-end detect -> crop -> OCR pipeline
```

## Requirements

- Python 3.10
- Core dependencies (from `pyproject.toml`):
  - `ultralytics` (YOLOv8)
  - `paddleocr` + `paddlepaddle`
  - `opencv-python`
  - `label-studio` (for manual annotation)

> **Note:** `requirements.txt` is a full environment freeze rather than a
> minimal dependency list. For a clean install, prefer `pyproject.toml`.

### Install with `uv` (recommended)

```bash
uv sync
```

### Install with `pip`

```bash
pip install -e .
```

## Dataset preparation

The pipeline expects a YOLO-format dataset described by `dataset.yaml`:

```yaml
path: ./dataset/processed
train: images/train
val: images/val
names:
  0: meter_reading
  1: serial_number
```

The dataset was built using an AI-assisted labeling workflow:

1. **Manually annotate a seed set** — ~50 images were labeled by hand in
   [Label Studio](https://labelstud.io/) and exported in YOLO format into
   `dataset/images/` and `dataset/labels/`.
2. **Train a draft model** on just that seed set (via `train.py`,
   pointed at the small labeled subset) to get a rough detector.
3. **Auto-label the rest of the dataset** with the draft model. This
   skips any image that already has a label file, so the manually
   labeled seed images are never overwritten:

   ```bash
   python src/meter_reader/auto-labelling.py
   ```

   By default this loads the draft model from
   `runs/detect/meter_detector-2/weights/best.pt` — update the path in
   the script if your draft model was saved elsewhere.

4. **Review and correct in Label Studio** — the auto-generated labels
   were re-imported into Label Studio and manually reviewed/corrected
   for accuracy.
5. **Discard the draft model** — once the full dataset was labeled and
   corrected, the draft model was deleted; it was only a labeling aid,
   not the production model.
6. **Split into train/val sets** (80/20 by default, seeded for
   reproducibility):

   ```bash
   python src/meter_reader/split_data.py
   ```

   This reads from `dataset/` and writes the split into
   `dataset/processed/{images,labels}/{train,val}`, matching the layout
   `dataset.yaml` expects.

## Training

`train.py` trains a YOLOv8n detector and is used twice in this project's
workflow: once on the small hand-labeled seed set to produce the
temporary **draft model** used for auto-labeling, and once on the
**full, reviewed dataset** to produce the final production model:

```bash
python src/meter_reader/train.py
```

This trains for up to 120 epochs (with early stopping after 10 epochs of
no improvement) on CPU and saves the best weights to:

```
runs/detect/meter_detector_prod/weights/best.pt
```

## Running the pipeline

Once you have trained weights, run the full detect → crop → OCR pipeline
on a set of images:

```bash
python src/meter_reader/pipeline.py
```

By default this loads the model from `models/best.pt` and processes every
image in `dataset/processed/images/val`, writing combined results to
`batch_results.json`:

```json
{
    "meter_reading": 12345.6,
    "raw_meter_reading": "12345.6",
    "serial_number": "46260789",
    "raw_serial_number": "CAT-C3 46260789",
    "detections": {
        "meter_reading_conf": 0.94,
        "serial_number_conf": 0.88
    },
    "filename": "example.jpg"
}
```

To use `MeterPipeline` on a single image programmatically:

```python
from src.meter_reader.pipeline import MeterPipeline

pipeline = MeterPipeline(yolo_model_path="models/best.pt")
result = pipeline.process_image("path/to/image.jpg")
print(result)
```

## Status / known limitations

- `auto-labelling.py` references the draft model's path
  (`runs/detect/meter_detector-2/weights/best.pt`); since the draft
  model is deleted after labeling is complete, this script is only
  needed again if you extend the dataset with new unlabeled images.
- `pipeline.py` loads the final production model from `models/best.pt`
  by default — update this path to wherever your trained
  `meter_detector_prod/weights/best.pt` is copied.
- The `meter-reader` console script declared in `pyproject.toml` expects
  a `main()` function in `src/meter_reader/__init__.py`, which is not
  yet implemented.
- `requirements.txt` reflects a full local environment rather than a
  minimal set of runtime dependencies — use `pyproject.toml` for
  installs.

