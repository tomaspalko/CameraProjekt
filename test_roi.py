"""Unit tests for src/core/roi.py."""
import numpy as np
import pytest
from src.core.roi import ROI


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def basic_roi():
    return ROI(x0=10, y0=20, x1=100, y1=150)

IMAGE_SHAPE = (256, 256)


# ── is_valid ─────────────────────────────────────────────────────────────────

def test_valid_roi(basic_roi):
    assert basic_roi.is_valid()

def test_valid_roi_within_shape(basic_roi):
    assert basic_roi.is_valid(shape=IMAGE_SHAPE)

def test_invalid_roi_zero_width():
    assert not ROI(10, 10, 10, 50).is_valid()

def test_invalid_roi_zero_height():
    assert not ROI(10, 10, 50, 10).is_valid()

def test_invalid_roi_negative_width():
    assert not ROI(50, 10, 10, 50).is_valid()

def test_invalid_roi_negative_height():
    assert not ROI(10, 50, 50, 10).is_valid()

def test_invalid_roi_out_of_bounds_right():
    assert not ROI(0, 0, 300, 100).is_valid(shape=IMAGE_SHAPE)

def test_invalid_roi_out_of_bounds_bottom():
    assert not ROI(0, 0, 100, 300).is_valid(shape=IMAGE_SHAPE)

def test_invalid_roi_negative_origin():
    assert not ROI(-1, 0, 100, 100).is_valid(shape=IMAGE_SHAPE)

def test_valid_roi_exact_shape_boundary():
    assert ROI(0, 0, 256, 256).is_valid(shape=IMAGE_SHAPE)


# ── create_mask ───────────────────────────────────────────────────────────────

def test_mask_shape(basic_roi):
    mask = basic_roi.create_mask(IMAGE_SHAPE)
    assert mask.shape == IMAGE_SHAPE

def test_mask_dtype(basic_roi):
    mask = basic_roi.create_mask(IMAGE_SHAPE)
    assert mask.dtype == np.uint8

def test_mask_inside_pixels(basic_roi):
    mask = basic_roi.create_mask(IMAGE_SHAPE)
    assert mask[basic_roi.y0, basic_roi.x0] == 255
    assert mask[basic_roi.y1 - 1, basic_roi.x1 - 1] == 255

def test_mask_outside_pixels(basic_roi):
    mask = basic_roi.create_mask(IMAGE_SHAPE)
    assert mask[0, 0] == 0
    assert mask[255, 255] == 0

def test_mask_count(basic_roi):
    mask = basic_roi.create_mask(IMAGE_SHAPE)
    expected = basic_roi.width * basic_roi.height
    assert np.sum(mask == 255) == expected

def test_mask_full_image():
    roi = ROI(0, 0, 256, 256)
    mask = roi.create_mask(IMAGE_SHAPE)
    assert np.all(mask == 255)

def test_mask_clamps_to_bounds():
    roi = ROI(-10, -10, 300, 300)   # oversized
    mask = roi.create_mask(IMAGE_SHAPE)
    assert mask.shape == IMAGE_SHAPE
    assert np.all(mask == 255)       # entire image covered after clamping

def test_mask_degenerate_after_clamp():
    roi = ROI(300, 300, 400, 400)    # entirely outside
    mask = roi.create_mask(IMAGE_SHAPE)
    assert np.all(mask == 0)


# ── Properties ────────────────────────────────────────────────────────────────

def test_width(basic_roi):
    assert basic_roi.width == basic_roi.x1 - basic_roi.x0

def test_height(basic_roi):
    assert basic_roi.height == basic_roi.y1 - basic_roi.y0


# ── Serialisation ─────────────────────────────────────────────────────────────

def test_to_dict(basic_roi):
    d = basic_roi.to_dict()
    assert d == {"x0": 10, "y0": 20, "x1": 100, "y1": 150}

def test_from_dict_round_trip(basic_roi):
    restored = ROI.from_dict(basic_roi.to_dict())
    assert restored == basic_roi

def test_from_dict_string_values():
    d = {"x0": "5", "y0": "10", "x1": "50", "y1": "100"}
    roi = ROI.from_dict(d)
    assert roi.x0 == 5 and roi.y1 == 100
