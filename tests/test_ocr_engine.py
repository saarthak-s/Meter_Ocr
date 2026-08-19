# tests/test_ocr_engine.py
import pytest
from meter_reader.ocr_engine import MeterOCREngine


@pytest.fixture
def engine():
    # Instantiates the class without invoking __init__ (skips PaddleOCR download/load)
    return MeterOCREngine.__new__(MeterOCREngine)


# --- Meter Reading Validation Tests ---

def test_validate_reading_basic(engine):
    assert engine.validate_reading("0006455") == 6455.0


def test_validate_reading_decimal(engine):
    assert engine.validate_reading("123.45 kWh") == 123.45


def test_validate_reading_with_text_prefix(engine):
    assert engine.validate_reading("kW h 075669") == 75669.0


def test_validate_reading_no_digits(engine):
    assert engine.validate_reading("garbage text") is None


def test_validate_reading_empty_string(engine):
    assert engine.validate_reading("") is None


# --- Serial Number Validation Tests ---

def test_validate_serial_prefers_8_digit(engine):
    assert engine.validate_serial("ULMa Meter. CAT-C3 46260789") == "46260789"


def test_validate_serial_standalone(engine):
    assert engine.validate_serial("26223997") == "26223997"


def test_validate_serial_fallback_longest(engine):
    # Falls back to the longest block if 8 digits aren't present (min length 5)
    assert engine.validate_serial("ABC 123 456789") == "456789"


def test_validate_serial_ignores_short_numbers(engine):
    # Ignore numbers with fewer than 5 digits
    assert engine.validate_serial("Ph 1 Type 23") is None


def test_validate_serial_empty(engine):
    assert engine.validate_serial("") is None