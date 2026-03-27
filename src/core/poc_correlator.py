"""Phase-Only Correlation (POC) with log-polar rotation estimation.

Pipeline:
  1. Log-polar FFT phase correlation  →  rotation angle
  2. Derotate image by estimated angle
  3. POC on derotated pair            →  sub-pixel translation
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from src.core.aligner import _ncc_score


# ── Internal helpers ──────────────────────────────────────────────────────────

def _hann2d(h: int, w: int) -> np.ndarray:
    """2D Hann window — reduces spectral leakage at image borders."""
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    return np.outer(wy, wx)


def _cross_power_spectrum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised cross-power spectrum (POC kernel).

    Given f(x) = g(x - d), the IFFT of this map is a sharp peak at d.
    """
    A = np.fft.fft2(a.astype(np.float32))
    B = np.fft.fft2(b.astype(np.float32))
    R = A * np.conj(B)
    eps = np.finfo(np.float32).eps
    R /= (np.abs(R) + eps)
    return np.real(np.fft.ifft2(R)).astype(np.float32)


def _parabolic_peak(surface: np.ndarray) -> tuple[float, float, float]:
    """Find the peak of *surface* with sub-pixel accuracy.

    Fits a 1-D parabola through the peak and its two neighbours along
    each axis independently.  Uses periodic (modular) indexing so that
    peaks near the array boundary are handled correctly.

    Returns:
        (x_sub, y_sub, peak_value)  — peak_value is the integer-peak height.
    """
    idx = np.unravel_index(int(np.argmax(surface)), surface.shape)
    py, px = idx
    peak_val = float(surface[py, px])

    h, w = surface.shape

    # Y-axis parabolic fit (periodic wrap-around)
    fy_m = float(surface[(py - 1) % h, px])
    fy_0 = float(surface[py,            px])
    fy_p = float(surface[(py + 1) % h, px])
    denom = fy_m - 2.0 * fy_0 + fy_p
    dy = -(fy_p - fy_m) / (2.0 * denom) if abs(denom) > 1e-10 else 0.0
    dy = float(np.clip(dy, -2.0, 2.0))

    # X-axis parabolic fit (periodic wrap-around)
    fx_m = float(surface[py, (px - 1) % w])
    fx_0 = float(surface[py,  px         ])
    fx_p = float(surface[py, (px + 1) % w])
    denom = fx_m - 2.0 * fx_0 + fx_p
    dx = -(fx_p - fx_m) / (2.0 * denom) if abs(denom) > 1e-10 else 0.0
    dx = float(np.clip(dx, -2.0, 2.0))

    return px + dx, py + dy, peak_val


# ── Rotation estimation ───────────────────────────────────────────────────────

def _estimate_rotation(ref: np.ndarray, img: np.ndarray) -> tuple[float, float]:
    """Estimate rotation angle via log-polar FFT phase correlation.

    Exploits the property that rotating an image by α° rotates its Fourier
    magnitude spectrum by the same α°.  After a log-polar transform this
    rotation becomes a translation, which POC detects directly.

    Returns:
        (angle_deg, confidence)  where confidence is the POC peak value.
    """
    h, w = ref.shape[:2]
    center = (w / 2.0, h / 2.0)
    max_radius = min(h, w) / 2.0

    # Log-polar output: 512 rows × log-radius columns.
    # Using only the first half-circle (0–180°) because the FFT magnitude
    # spectrum is symmetric (180° periodicity).
    # half_h = 256 → resolution ≈ 180/256 ≈ 0.70 °/row (before sub-pixel).
    # INTER_CUBIC gives sharper interpolation than INTER_LINEAR for the warp.
    out_h = 512
    out_w = int(max_radius)
    half_h = out_h // 2

    hann = _hann2d(h, w)

    def _log_polar_mag(image: np.ndarray) -> np.ndarray:
        f = np.fft.fft2(image.astype(np.float32) * hann)
        mag = np.abs(np.fft.fftshift(f)).astype(np.float32)
        mag = np.log1p(mag)
        flags = cv2.WARP_POLAR_LOG | cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS
        lp = cv2.warpPolar(mag, (out_w, out_h), center, max_radius, flags)
        return lp[:half_h, :]   # first half = 0..180°

    ref_lp = _log_polar_mag(ref)
    img_lp = _log_polar_mag(img)

    poc_map = _cross_power_spectrum(ref_lp, img_lp)
    _peak_x, peak_y, confidence = _parabolic_peak(poc_map)

    # Map peak row → angle in degrees (range 0..180°)
    angle_raw = peak_y / half_h * 180.0

    # Wrap: rotations > 90° are interpreted as negative rotations
    if angle_raw > 90.0:
        angle_raw -= 180.0

    return float(angle_raw), float(np.clip(confidence, 0.0, 1.0))


# ── Translation estimation ────────────────────────────────────────────────────

def _estimate_translation(
    ref: np.ndarray,
    img: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Estimate translation (dx, dy) via Phase-Only Correlation.

    Returns:
        (dx_px, dy_px, confidence)
    """
    h, w = ref.shape[:2]
    hann = _hann2d(h, w)

    ref_f = ref.astype(np.float32) * hann
    img_f = img.astype(np.float32) * hann

    if mask is not None:
        weight = mask.astype(np.float32) / 255.0
        ref_f *= weight
        img_f *= weight

    # IMG * conj(REF): IFFT peak lands at (dy, dx) — the forward displacement
    poc_map = _cross_power_spectrum(img_f, ref_f)

    # fftshift centres the DC component — zero shift = centre pixel
    poc_shifted = np.fft.fftshift(poc_map)
    peak_x, peak_y, confidence = _parabolic_peak(poc_shifted)

    dx = peak_x - w / 2.0
    dy = peak_y - h / 2.0

    return float(dx), float(dy), float(np.clip(confidence, 0.0, 1.0))


# ── Public API ────────────────────────────────────────────────────────────────

def poc_align(
    reference: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    """Align *image* to *reference* using Phase-Only Correlation (POC).

    Two-stage pipeline:
      1. Log-polar FFT phase correlation estimates rotation.
      2. After derotation, POC estimates sub-pixel translation.

    Assumes small motion (translation ≤ ~15 px, rotation ≤ ~2°).

    Args:
        reference: Grayscale uint8 reference image.
        image:     Grayscale uint8 image to align.
        mask:      Optional uint8 mask (255 = use pixel, 0 = ignore).
                   Applied only to the translation step.

    Returns:
        dict with keys:
            dx_px     – horizontal shift in pixels (+ = right)
            dy_px     – vertical shift in pixels   (+ = down)
            angle_deg – rotation in degrees
            confidence – POC peak value in [0, 1]
    """
    try:
        h, w = reference.shape[:2]

        # Stage 1 — rotation
        angle_deg, rot_conf = _estimate_rotation(reference, image)

        # Stage 2 — derotate image, then estimate translation
        if abs(angle_deg) > 1e-6:
            M_derot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle_deg, 1.0)
            image_derot = cv2.warpAffine(
                image, M_derot, (w, h), flags=cv2.INTER_LINEAR
            )
        else:
            image_derot = image

        dx, dy, trans_conf = _estimate_translation(reference, image_derot, mask)

        # Combined confidence: limited by the weaker stage
        confidence = float(min(rot_conf, trans_conf))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # NCC score: apply estimated transform to image and compare to reference
        M_align = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle_deg, 1.0)
        M_align[0, 2] += dx
        M_align[1, 2] += dy
        aligned = cv2.warpAffine(
            image.astype(np.float32), M_align, (w, h), flags=cv2.INTER_LINEAR
        )
        ncc = _ncc_score(reference.astype(np.float32), aligned)

    except (cv2.error, ValueError, FloatingPointError):
        # Degenerate input (uniform / dark image) — return neutral result
        dx, dy, angle_deg, confidence, ncc = 0.0, 0.0, 0.0, 0.0, 0.0

    return {
        "dx_px":     float(dx),
        "dy_px":     float(dy),
        "angle_deg": float(angle_deg),
        "confidence": confidence,
        "ncc_score": float(ncc),
    }
