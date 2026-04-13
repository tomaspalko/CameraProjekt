"""Tests for aligner.py — ECC-based sub-pixel registration."""
import numpy as np
import pytest
from src.core.aligner import align

from tests.constants import TRANSLATION_TOL_PX, ROTATION_TOL_DEG, MIN_CONFIDENCE
from tests.synthetic.generator import make_pair


# ── Tolerances (aliases for readability in this file) ────────────────────────

TRANSLATION_TOL = TRANSLATION_TOL_PX
ROTATION_TOL    = ROTATION_TOL_DEG
GOOD_CONFIDENCE = MIN_CONFIDENCE


# ── Basic convergence ────────────────────────────────────────────────────────

def test_align_returns_dict():
    ref, img, _ = make_pair(3.0, -2.0, 0.5)
    result = align(ref, img)
    for key in ("dx_px", "dy_px", "angle_deg", "confidence"):
        assert key in result, f"Missing key: {key}"

def test_align_confidence_is_float_in_range():
    ref, img, _ = make_pair(3.0, -2.0, 0.5)
    result = align(ref, img)
    assert 0.0 <= result["confidence"] <= 1.0

def test_align_no_motion_gives_zero():
    ref, _, _ = make_pair(0.0, 0.0, 0.0)
    result = align(ref, ref)
    assert abs(result["dx_px"])    < TRANSLATION_TOL
    assert abs(result["dy_px"])    < TRANSLATION_TOL
    assert abs(result["angle_deg"]) < ROTATION_TOL


# ── Accuracy parametric tests ─────────────────────────────────────────────────

ACCURACY_CASES = [
    # (dx,   dy,   angle)
    (  3.0,  0.0,  0.0  ),   # pure translation X
    (  0.0, -4.0,  0.0  ),   # pure translation Y
    (  0.0,  0.0,  1.0  ),   # pure rotation 1°
    (  0.0,  0.0, -1.0  ),   # pure rotation -1°
    (  5.0, -3.0,  0.8  ),   # combined
    ( -2.5,  1.5, -0.5  ),   # combined negative
    (  0.3,  0.1,  0.1  ),   # sub-pixel only
    ( 10.0,  8.0,  1.5  ),   # larger displacement
]

@pytest.mark.parametrize("dx,dy,angle", ACCURACY_CASES)
def test_accuracy(dx, dy, angle):
    ref, img, gt = make_pair(dx, dy, angle)
    result = align(ref, img)

    err_dx    = abs(result["dx_px"]    - gt["dx"])
    err_dy    = abs(result["dy_px"]    - gt["dy"])
    err_angle = abs(result["angle_deg"] - gt["angle_deg"])

    assert err_dx    < TRANSLATION_TOL, f"dx error {err_dx:.4f} px > {TRANSLATION_TOL}"
    assert err_dy    < TRANSLATION_TOL, f"dy error {err_dy:.4f} px > {TRANSLATION_TOL}"
    assert err_angle < ROTATION_TOL,    f"angle error {err_angle:.4f}° > {ROTATION_TOL}"


# ── Noise robustness ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("noise_sigma", [0.0, 2.0, 5.0, 10.0])
def test_noise_robustness(noise_sigma):
    ref, img, gt = make_pair(3.0, -2.0, 0.5, noise_sigma=noise_sigma)
    result = align(ref, img)
    assert abs(result["dx_px"] - gt["dx"]) < TRANSLATION_TOL * (1 + noise_sigma / 5)


# ── Regression: confidence on good image ─────────────────────────────────────

def test_confidence_good_image():
    ref, img, _ = make_pair(2.0, 1.0, 0.3)
    result = align(ref, img)
    assert result["confidence"] > GOOD_CONFIDENCE, \
        f"Confidence {result['confidence']:.3f} too low for good image pair"

def test_confidence_bad_image_is_low():
    """Random noise image should yield low confidence."""
    rng = np.random.default_rng(0)
    ref  = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    img  = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    result = align(ref, img)
    assert result["confidence"] < 0.8, "Expected low confidence for random pair"


# ── Does not crash on edge cases ──────────────────────────────────────────────

def test_align_does_not_crash_on_dark_image():
    dark = np.zeros((256, 256), dtype=np.uint8)
    result = align(dark, dark)
    for key in ("dx_px", "dy_px", "angle_deg", "confidence"):
        assert key in result
    assert 0.0 <= result["confidence"] <= 1.0

def test_align_does_not_crash_on_uniform_image():
    flat = np.full((256, 256), 128, dtype=np.uint8)
    result = align(flat, flat)
    for key in ("dx_px", "dy_px", "angle_deg", "confidence"):
        assert key in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_align_all_zero_mask_does_not_crash():
    """All-zero mask (no valid pixels) must not crash — returns neutral result."""
    ref, img, _ = make_pair(2.0, 1.0, 0.3)
    mask = np.zeros(ref.shape, dtype=np.uint8)
    result = align(ref, img, mask=mask)
    for key in ("dx_px", "dy_px", "angle_deg", "confidence"):
        assert key in result
    assert 0.0 <= result["confidence"] <= 1.0


# ── RMS over all accuracy cases ───────────────────────────────────────────────

def test_rms_accuracy_over_all_cases():
    """Aggregate RMS must be below threshold across all test cases."""
    dx_errs, dy_errs, angle_errs = [], [], []
    for dx, dy, angle in ACCURACY_CASES:
        ref, img, gt = make_pair(dx, dy, angle)
        result = align(ref, img)
        dx_errs.append((result["dx_px"] - gt["dx"]) ** 2)
        dy_errs.append((result["dy_px"] - gt["dy"]) ** 2)
        angle_errs.append((result["angle_deg"] - gt["angle_deg"]) ** 2)

    rms_dx    = np.sqrt(np.mean(dx_errs))
    rms_dy    = np.sqrt(np.mean(dy_errs))
    rms_angle = np.sqrt(np.mean(angle_errs))

    assert rms_dx    < TRANSLATION_TOL, f"RMS dx {rms_dx:.4f} px"
    assert rms_dy    < TRANSLATION_TOL, f"RMS dy {rms_dy:.4f} px"
    assert rms_angle < ROTATION_TOL,    f"RMS angle {rms_angle:.4f}°"
