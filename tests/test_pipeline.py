# tests/test_pipeline.py
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from meter_reader.pipeline import MeterPipeline


@pytest.fixture
def dummy_weights_path(tmp_path):
    """Creates a temporary dummy weights file to satisfy the existence check."""
    weights_file = tmp_path / "dummy_weights.pt"
    weights_file.touch()
    return str(weights_file)


@pytest.fixture
def dummy_image_path(tmp_path):
    """Creates a blank dummy image file for testing."""
    img_path = tmp_path / "test_meter.jpg"
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), blank_img)
    return img_path


@patch("src.meter_reader.pipeline.MeterOCREngine")
@patch("src.meter_reader.pipeline.YOLO")
def test_pipeline_no_detections(mock_yolo, mock_ocr, dummy_weights_path, dummy_image_path):
    """Verifies pipeline returns a clean null dict when no boxes are detected."""
    mock_yolo_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = []
    mock_yolo_instance.return_value = [mock_result]
    mock_yolo.return_value = mock_yolo_instance

    pipeline = MeterPipeline(yolo_model_path=dummy_weights_path)
    result = pipeline.process_image(dummy_image_path)

    assert result["meter_reading"] is None
    assert result["serial_number"] is None
    assert result["detections"]["meter_reading_conf"] is None
    assert result["detections"]["serial_number_conf"] is None


def test_pipeline_missing_file_raises_error(dummy_weights_path):
    """Verifies that passing a non-existent image path raises FileNotFoundError."""
    with patch("src.meter_reader.pipeline.YOLO"), patch("src.meter_reader.pipeline.MeterOCREngine"):
        pipeline = MeterPipeline(yolo_model_path=dummy_weights_path)
        with pytest.raises(FileNotFoundError):
            pipeline.process_image("non_existent_image.jpg")