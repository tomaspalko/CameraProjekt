"""tests/conftest.py — shared fixtures and tolerance constants for all tests."""
import numpy as np
import pytest

from tests.constants import (
    TRANSLATION_TOL_PX,
    ROTATION_TOL_DEG,
    MM_CONVERSION_TOL,
    MIN_CONFIDENCE,
)
from tests.synthetic.generator import make_textured, transform


def pytest_configure(config):
    config._translation_tol = TRANSLATION_TOL_PX
    config._rotation_tol    = ROTATION_TOL_DEG


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def reference_image():
    """A single reference image reused across the whole test session."""
    return make_textured(seed=0)


@pytest.fixture
def image_pair(reference_image):
    """Factory fixture: call it with (dx, dy, angle) → (ref, transformed)."""
    def _factory(dx=0.0, dy=0.0, angle=0.0, noise=1.0, seed=42):
        rng = np.random.default_rng(seed)
        transformed = transform(reference_image, dx, dy, angle)
        if noise > 0:
            n = rng.normal(0, noise, transformed.shape).astype(np.float32)
            transformed = np.clip(transformed.astype(np.float32) + n, 0, 255).astype(np.uint8)
        return reference_image.copy(), transformed
    return _factory


@pytest.fixture
def tolerance():
    """Expose tolerance constants to tests."""
    return {
        "translation_px": TRANSLATION_TOL_PX,
        "rotation_deg":   ROTATION_TOL_DEG,
        "mm_conversion":  MM_CONVERSION_TOL,
        "min_confidence": MIN_CONFIDENCE,
    }
