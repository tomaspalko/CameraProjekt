"""conftest.py — shared fixtures and tolerance constants for all tests."""
import cv2
import numpy as np
import pytest


# ── Global tolerances (single source of truth) ───────────────────────────────
# Change these here to tighten/loosen ALL tests at once.

TRANSLATION_TOL_PX = 0.10   # max allowed error in dx/dy [px]
ROTATION_TOL_DEG   = 0.05   # max allowed error in angle [°]
MM_CONVERSION_TOL  = 1e-5   # relative error for px→mm
MIN_CONFIDENCE     = 0.60   # minimum acceptable ECC confidence


def pytest_configure(config):
    config._translation_tol = TRANSLATION_TOL_PX
    config._rotation_tol    = ROTATION_TOL_DEG


# ── Image generators ──────────────────────────────────────────────────────────

def _make_textured(size=(256, 256), seed=42):
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 220, size, dtype=np.uint8).astype(np.float32)
    for _ in range(30):
        cx = int(rng.integers(20, size[1] - 20))
        cy = int(rng.integers(20, size[0] - 20))
        r  = int(rng.integers(5, 20))
        c  = float(rng.integers(50, 200))
        cv2.circle(img, (cx, cy), r, c, -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _transform(ref, dx, dy, angle_deg):
    h, w = ref.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(ref, M, (w, h))


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def reference_image():
    """A single reference image reused across the whole test session."""
    return _make_textured(seed=0)


@pytest.fixture
def image_pair(reference_image):
    """Factory fixture: call it with (dx, dy, angle) → (ref, transformed)."""
    def _factory(dx=0.0, dy=0.0, angle=0.0, noise=1.0, seed=42):
        rng = np.random.default_rng(seed)
        transformed = _transform(reference_image, dx, dy, angle)
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
