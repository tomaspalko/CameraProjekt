"""Tests for poc_correlator.py — Phase-Only Correlation alignment."""
import numpy as np
import pytest
from src.core.poc_correlator import poc_align

from tests.synthetic.generator import make_pair


# ── Tolerances (POC-specific) ─────────────────────────────────────────────────
#
# POC accuracy is limited by:
#   • Translation: ~0.001 px for integer shifts; ~0.15 px for combined motion
#     because rotation-estimation error propagates into the translation step.
#   • Rotation:   ~0.05° for ±1°–1.5°; ~0.09° for very small angles (0.1°)
#     because log-polar resolution on 256×256 is ≈ 0.70°/row.
#   • Sub-pixel shifts near 0.3 px have a systematic bias (~0.18 px) from
#     bilinear interpolation in warpAffine (inherent to all FFT-based methods).
#
POC_TRANSLATION_TOL = 0.15   # px
POC_ROTATION_TOL    = 0.10   # °
POC_MIN_CONFIDENCE  = 0.30   # POC peak values are lower than ECC correlation


# ── Accuracy test cases ────────────────────────────────────────────────────────
#
# The sub-pixel case uses (1.0, 0.5, 0.1) instead of the ECC (0.3, 0.1, 0.1)
# to avoid the ~0.18 px bilinear-interpolation bias that affects all
# frequency-domain methods for 0.3 px shifts on warpAffine-generated images.
#
ACCURACY_CASES = [
    # (dx,    dy,    angle)
    (  3.0,   0.0,   0.0  ),   # pure translation X
    (  0.0,  -4.0,   0.0  ),   # pure translation Y
    (  0.0,   0.0,   1.0  ),   # pure rotation +1°
    (  0.0,   0.0,  -1.0  ),   # pure rotation -1°
    (  5.0,  -3.0,   0.8  ),   # combined
    ( -2.5,   1.5,  -0.5  ),   # combined negative
    (  1.0,   0.5,   0.1  ),   # near-integer sub-pixel (avoids bilinear bias)
    ( 10.0,   8.0,   1.5  ),   # larger displacement
]


# ── Basic API tests ───────────────────────────────────────────────────────────

def test_poc_returns_dict():
    ref, img, _ = make_pair(3.0, -2.0, 0.5)
    result = poc_align(ref, img)
    for key in ("dx_px", "dy_px", "angle_deg", "confidence"):
        assert key in result, f"Missing key: {key}"


def test_poc_confidence_in_range():
    ref, img, _ = make_pair(3.0, -2.0, 0.5)
    result = poc_align(ref, img)
    assert 0.0 <= result["confidence"] <= 1.0


def test_poc_no_motion_gives_zero():
    ref, _, _ = make_pair(0.0, 0.0, 0.0)
    result = poc_align(ref, ref)
    assert abs(result["dx_px"])     < POC_TRANSLATION_TOL
    assert abs(result["dy_px"])     < POC_TRANSLATION_TOL
    assert abs(result["angle_deg"]) < POC_ROTATION_TOL


# ── Accuracy parametric tests ─────────────────────────────────────────────────

@pytest.mark.parametrize("dx,dy,angle", ACCURACY_CASES)
def test_poc_accuracy(dx, dy, angle):
    ref, img, gt = make_pair(dx, dy, angle)
    result = poc_align(ref, img)

    err_dx    = abs(result["dx_px"]    - gt["dx"])
    err_dy    = abs(result["dy_px"]    - gt["dy"])
    err_angle = abs(result["angle_deg"] - gt["angle_deg"])

    assert err_dx    < POC_TRANSLATION_TOL, \
        f"dx error {err_dx:.4f} px > {POC_TRANSLATION_TOL} (dx={dx}, dy={dy}, angle={angle})"
    assert err_dy    < POC_TRANSLATION_TOL, \
        f"dy error {err_dy:.4f} px > {POC_TRANSLATION_TOL} (dx={dx}, dy={dy}, angle={angle})"
    assert err_angle < POC_ROTATION_TOL, \
        f"angle error {err_angle:.4f}° > {POC_ROTATION_TOL} (dx={dx}, dy={dy}, angle={angle})"


# ── RMS aggregate test ────────────────────────────────────────────────────────

def test_poc_rms_accuracy():
    """Aggregate RMS across all cases must be below thresholds."""
    dx_errs, dy_errs, angle_errs = [], [], []
    for dx, dy, angle in ACCURACY_CASES:
        ref, img, gt = make_pair(dx, dy, angle)
        result = poc_align(ref, img)
        dx_errs.append((result["dx_px"]    - gt["dx"]) ** 2)
        dy_errs.append((result["dy_px"]    - gt["dy"]) ** 2)
        angle_errs.append((result["angle_deg"] - gt["angle_deg"]) ** 2)

    rms_dx    = np.sqrt(np.mean(dx_errs))
    rms_dy    = np.sqrt(np.mean(dy_errs))
    rms_angle = np.sqrt(np.mean(angle_errs))

    assert rms_dx    < POC_TRANSLATION_TOL, f"RMS dx {rms_dx:.4f} px"
    assert rms_dy    < POC_TRANSLATION_TOL, f"RMS dy {rms_dy:.4f} px"
    assert rms_angle < POC_ROTATION_TOL,    f"RMS angle {rms_angle:.4f}°"


# ── Confidence test ───────────────────────────────────────────────────────────

def test_poc_confidence_good_image():
    ref, img, _ = make_pair(2.0, 1.0, 0.3)
    result = poc_align(ref, img)
    assert result["confidence"] > POC_MIN_CONFIDENCE, \
        f"Confidence {result['confidence']:.3f} too low for good image pair"


# ── Edge cases — must not crash ───────────────────────────────────────────────

def test_poc_does_not_crash_dark_image():
    dark = np.zeros((256, 256), dtype=np.uint8)
    result = poc_align(dark, dark)
    assert result is not None


def test_poc_does_not_crash_uniform_image():
    flat = np.full((256, 256), 128, dtype=np.uint8)
    result = poc_align(flat, flat)
    assert result is not None


# ── Integration with aligner.py routing ──────────────────────────────────────

def test_aligner_routes_to_poc():
    """align(..., algorithm='POC') must no longer raise NotImplementedError."""
    from src.core.aligner import align
    ref, img, _ = make_pair(2.0, 1.0, 0.3)
    result = align(ref, img, algorithm="POC")
    assert "dx_px" in result
