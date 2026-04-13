"""Unit tests for preprocessor.py — run after every change."""
import numpy as np
import pytest
from src.core.preprocessor import preprocess


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def gray_image():
    rng = np.random.default_rng(42)
    return (rng.integers(0, 256, (1024, 1280), dtype=np.uint8))

@pytest.fixture
def color_image():
    rng = np.random.default_rng(42)
    return (rng.integers(0, 256, (1024, 1280, 3), dtype=np.uint8))


# ── Output shape & dtype ─────────────────────────────────────────────────────

def test_output_is_grayscale_from_color(color_image):
    result = preprocess(color_image)
    assert result.ndim == 2, "Output must be 2D grayscale"

def test_output_is_grayscale_from_gray(gray_image):
    result = preprocess(gray_image)
    assert result.ndim == 2

def test_output_dtype_uint8(gray_image):
    result = preprocess(gray_image)
    assert result.dtype == np.uint8

def test_output_shape_preserved(gray_image):
    result = preprocess(gray_image)
    assert result.shape == gray_image.shape

def test_output_shape_preserved_from_color(color_image):
    result = preprocess(color_image)
    assert result.shape == color_image.shape[:2]


# ── Value range ──────────────────────────────────────────────────────────────

def test_output_values_in_valid_range(gray_image):
    result = preprocess(gray_image)
    assert result.min() >= 0
    assert result.max() <= 255

def test_clahe_increases_contrast():
    """CLAHE should increase std deviation of a low-contrast image."""
    flat = np.full((256, 256), 128, dtype=np.uint8)
    flat[100:150, 100:150] = 130  # tiny contrast
    result = preprocess(flat)
    assert result.std() >= flat.std()


# ── Edge cases ───────────────────────────────────────────────────────────────

def test_all_black_image_does_not_crash():
    black = np.zeros((256, 256), dtype=np.uint8)
    result = preprocess(black)
    assert result.shape == black.shape
    assert result.dtype == np.uint8

def test_all_white_image_does_not_crash():
    white = np.full((256, 256), 255, dtype=np.uint8)
    result = preprocess(white)
    assert result.shape == white.shape
    assert result.dtype == np.uint8

def test_single_pixel_image_does_not_crash():
    img = np.array([[128]], dtype=np.uint8)
    result = preprocess(img)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

@pytest.mark.parametrize("clahe_clip", [1.0, 2.0, 4.0, 8.0])
def test_various_clahe_clips(gray_image, clahe_clip):
    result = preprocess(gray_image, clahe_clip=clahe_clip)
    assert result.shape == gray_image.shape

@pytest.mark.parametrize("blur_kernel", [1, 3, 5])
def test_various_blur_kernels(gray_image, blur_kernel):
    result = preprocess(gray_image, blur_kernel=blur_kernel)
    assert result.shape == gray_image.shape


# ── auto_clahe ───────────────────────────────────────────────────────────────

def test_auto_clahe_output_shape(gray_image):
    result = preprocess(gray_image, auto_clahe=True)
    assert result.shape == gray_image.shape

def test_auto_clahe_output_dtype(gray_image):
    result = preprocess(gray_image, auto_clahe=True)
    assert result.dtype == np.uint8

def test_auto_clahe_does_not_crash_black():
    black = np.zeros((256, 256), dtype=np.uint8)
    result = preprocess(black, auto_clahe=True)
    assert result.shape == black.shape
    assert result.dtype == np.uint8

def test_auto_clahe_does_not_crash_white():
    white = np.full((256, 256), 255, dtype=np.uint8)
    result = preprocess(white, auto_clahe=True)
    assert result.shape == white.shape
    assert result.dtype == np.uint8

def test_auto_clahe_uniform_image():
    """Perfectly uniform image (std=0, gradient=0) must not crash or produce invalid output."""
    uniform = np.full((256, 256), 100, dtype=np.uint8)
    result = preprocess(uniform, auto_clahe=True)
    assert result.shape == uniform.shape
    assert result.dtype == np.uint8
    assert 0 <= int(result.min()) <= int(result.max()) <= 255
