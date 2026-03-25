"""tests/synthetic/generator.py — synthetic image generation helpers.

Used by tests/conftest.py fixtures and unit/integration test files.
"""
from __future__ import annotations

import cv2
import numpy as np


def make_textured(size=(256, 256), seed=42) -> np.ndarray:
    """Create a reproducible grayscale image with 30 random circular features."""
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 220, size, dtype=np.uint8).astype(np.float32)
    for _ in range(30):
        cx = int(rng.integers(20, size[1] - 20))
        cy = int(rng.integers(20, size[0] - 20))
        r  = int(rng.integers(5, 20))
        c  = float(rng.integers(50, 200))
        cv2.circle(img, (cx, cy), r, c, -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def transform(ref: np.ndarray, dx: float, dy: float, angle_deg: float) -> np.ndarray:
    """Apply rigid transform (rotation about centre + translation) to *ref*."""
    h, w = ref.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(ref, M, (w, h))


def make_pair(
    dx: float,
    dy: float,
    angle_deg: float,
    size=(256, 256),
    noise_sigma: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (reference, transformed, ground_truth_dict).

    The single rng is consumed in order: image pixels → circle params → noise.
    Do NOT split into make_textured + separate noise — it would change the rng
    state and alter test images, potentially breaking accuracy thresholds.
    """
    rng = np.random.default_rng(seed)
    ref = rng.integers(40, 220, size, dtype=np.uint8).astype(np.float32)
    for _ in range(30):
        cx = int(rng.integers(20, size[1] - 20))
        cy = int(rng.integers(20, size[0] - 20))
        cv2.circle(ref, (cx, cy), int(rng.integers(5, 20)),
                   float(rng.integers(50, 200)), -1)
    ref = np.clip(ref, 0, 255).astype(np.uint8)

    h, w = size
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    transformed = cv2.warpAffine(ref, M, (w, h))

    noise = rng.normal(0, noise_sigma, size).astype(np.float32)
    transformed = np.clip(transformed.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    gt = {"dx": dx, "dy": dy, "angle_deg": angle_deg}
    return ref, transformed, gt


def make_image_pair(
    dx: float,
    dy: float,
    angle: float,
    size=(256, 256),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (reference, transformed) without noise or ground truth dict."""
    rng = np.random.default_rng(seed)
    ref = rng.integers(40, 220, size, dtype=np.uint8).astype(np.float32)
    for _ in range(30):
        cx = int(rng.integers(20, size[1] - 20))
        cy = int(rng.integers(20, size[0] - 20))
        cv2.circle(ref, (cx, cy), int(rng.integers(5, 20)),
                   float(rng.integers(50, 200)), -1)
    ref = np.clip(ref, 0, 255).astype(np.uint8)

    h, w = size
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    img = cv2.warpAffine(ref, M, (w, h))
    return ref, img
