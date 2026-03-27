"""Edge detection algorithms for weld inspection.

Supported methods:
    Canny           — classic gradient + hysteresis (opencv)
    Scharr          — Scharr gradient magnitude, normalised threshold
    LoG             — Laplacian of Gaussian (scipy), zero-crossing via threshold
    PhaseCongruency — Kovesi log-Gabor filter bank; contrast-invariant,
                      well suited for monochromatic industrial cameras
    DexiNed         — deep learning model (Dense Extreme Inception Network);
                      requires PyTorch + pre-trained weights file
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage

METHODS: list[str] = ["Canny", "Scharr", "LoG", "PhaseCongruency", "DexiNed"]

# Module-level cache for loaded DexiNed model (lazy-loaded once per session)
_dexined_cache: dict = {}          # {"model": ..., "path": ..., "device": ...}

# Default weights path (project-relative)
DEXINED_DEFAULT_WEIGHTS = str(
    Path(__file__).parent.parent.parent / "models" / "dexined.onnx"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_edges(
    gray: np.ndarray,
    method: str,
    **params,
) -> np.ndarray:
    """Detect edges in a uint8 grayscale image.

    Args:
        gray:   uint8 grayscale image (2-D).
        method: one of METHODS.
        **params: method-specific keyword arguments (see individual functions).

    Returns:
        uint8 binary mask — 255 = edge, 0 = background; same shape as *gray*.
    """
    if gray.ndim != 2:
        raise ValueError("gray must be a 2-D uint8 array")

    dispatch = {
        "Canny":           detect_canny,
        "Scharr":          detect_scharr,
        "LoG":             detect_log,
        "PhaseCongruency": detect_phase_congruency,
        "DexiNed":         detect_dexined,
    }
    fn = dispatch.get(method)
    if fn is None:
        raise ValueError(f"Unknown edge detection method '{method}'. Choose from {METHODS}.")
    return fn(gray, **params)


# ---------------------------------------------------------------------------
# Canny
# ---------------------------------------------------------------------------

def detect_canny(
    gray: np.ndarray,
    t1: int = 50,
    t2: int = 150,
    blur: int = 3,
) -> np.ndarray:
    """Canny edge detector.

    Args:
        t1:   lower hysteresis threshold (0–255).
        t2:   upper hysteresis threshold (0–255).
        blur: Gaussian blur kernel size before Canny (1 = disabled, must be odd).
    """
    g = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray.copy()
    return cv2.Canny(g, float(t1), float(t2))


# ---------------------------------------------------------------------------
# Scharr
# ---------------------------------------------------------------------------

def detect_scharr(
    gray: np.ndarray,
    threshold: int = 30,
    blur: int = 3,
) -> np.ndarray:
    """Scharr gradient-magnitude edge detector.

    Args:
        threshold: edge strength threshold in 0–255 (applied to normalised magnitude).
        blur:      Gaussian blur kernel size (1 = disabled, must be odd).
    """
    g = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray.copy()
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    max_val = float(mag.max())
    if max_val < 1e-6:
        return np.zeros_like(gray)
    mag_norm = (mag / max_val * 255.0)
    return np.where(mag_norm > threshold, np.uint8(255), np.uint8(0)).astype(np.uint8)


# ---------------------------------------------------------------------------
# LoG — Laplacian of Gaussian
# ---------------------------------------------------------------------------

def detect_log(
    gray: np.ndarray,
    sigma: float = 1.5,
    threshold: int = 10,
    blur: int = 3,
) -> np.ndarray:
    """Laplacian-of-Gaussian edge detector.

    Computes |LoG| response, normalises to 0–255 and applies a threshold.

    Args:
        sigma:     standard deviation for the Gaussian kernel.
        threshold: edge response threshold in 0–255 (normalised).
        blur:      additional pre-blur kernel size (1 = disabled, must be odd).
    """
    g = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray.copy()
    log_response = ndimage.gaussian_laplace(g.astype(np.float32), sigma=sigma)
    log_abs = np.abs(log_response)
    max_val = float(log_abs.max())
    if max_val < 1e-6:
        return np.zeros_like(gray)
    log_norm = log_abs / max_val * 255.0
    return np.where(log_norm > threshold, np.uint8(255), np.uint8(0)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Phase Congruency — Kovesi algorithm (1-D log-Gabor, isotropic)
# ---------------------------------------------------------------------------

def detect_phase_congruency(
    gray: np.ndarray,
    nscale: int = 4,
    min_wavelength: int = 6,
    mult: float = 2.1,
    sigma_on_f: float = 0.55,
    k: float = 2.0,
) -> np.ndarray:
    """Phase Congruency edge detector (Kovesi, isotropic 1-D log-Gabor filter bank).

    Contrast- and brightness-invariant — ideal for monochromatic cameras with
    metallic surfaces where illumination varies across the field.

    Args:
        nscale:         number of filter scales (1–8, more → finer detail).
        min_wavelength: wavelength of the smallest-scale filter in pixels (2–40).
        mult:           scaling factor between successive filters (> 1).
        sigma_on_f:     bandwidth of each log-Gabor filter (0.2–0.9, larger → wider).
        k:              noise threshold multiplier (higher → fewer edges).

    Returns:
        uint8 binary mask where Phase Congruency exceeds the noise estimate.

    Algorithm reference:
        Kovesi, P. (1999). Image features from phase congruency.
        Videre: Journal of Computer Vision Research, 1(3).
    """
    rows, cols = gray.shape
    f_img = gray.astype(np.float32) / 255.0

    # 2-D FFT of image
    IM = np.fft.fft2(f_img)

    # Build frequency grids (normalised: [-0.5, 0.5])
    u = np.fft.fftfreq(cols).astype(np.float32)   # horizontal
    v = np.fft.fftfreq(rows).astype(np.float32)   # vertical
    u2d, v2d = np.meshgrid(u, v)
    radius = np.sqrt(u2d ** 2 + v2d ** 2)
    radius[0, 0] = 1.0  # avoid log(0) at DC

    # Accumulate PC components across scales
    sum_amplitude = np.zeros((rows, cols), dtype=np.float32)
    sum_e         = np.zeros((rows, cols), dtype=np.float32)  # even (Re)
    sum_o         = np.zeros((rows, cols), dtype=np.float32)  # odd  (Im)

    eps = 1e-5

    # Store first-scale response for noise estimation
    first_scale_resp: np.ndarray | None = None

    for s in range(nscale):
        wavelength = min_wavelength * (mult ** s)
        fo = 1.0 / wavelength                   # centre frequency
        rfo = fo / 0.5                           # normalise to [0, 1]

        # Log-Gabor filter (isotropic)
        log_gabor = np.exp(
            -(np.log(radius / fo + eps) ** 2) / (2.0 * np.log(sigma_on_f + eps) ** 2)
        )
        log_gabor[0, 0] = 0.0  # zero DC

        # Filter response
        EO = np.fft.ifft2(IM * log_gabor)
        An    = np.abs(EO).astype(np.float32)
        E     = EO.real.astype(np.float32)
        O     = EO.imag.astype(np.float32)

        sum_amplitude += An
        sum_e         += E
        sum_o         += O

        if s == 0:
            first_scale_resp = An

    # Phase congruency value
    energy = np.sqrt(sum_e ** 2 + sum_o ** 2)

    # Noise estimate from first scale (Gaussian noise → median / 0.6745)
    if first_scale_resp is not None:
        noise_est = float(np.median(first_scale_resp)) / 0.6745
    else:
        noise_est = 0.0

    T = k * noise_est
    PC = np.maximum(energy - T, 0.0) / (sum_amplitude + eps)

    # Threshold: k * std of PC map
    thresh = float(PC.mean()) + k * float(PC.std()) * 0.5
    mask = (PC > max(thresh, 1e-4)).astype(np.uint8) * 255

    # Thin edges with morphological operation for cleaner result
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


# ---------------------------------------------------------------------------
# DexiNed — Deep Learning edge detection
# ---------------------------------------------------------------------------

def detect_dexined(
    gray: np.ndarray,
    weights_path: str = "",
    threshold: float = 0.5,
    device: str = "cpu",
) -> np.ndarray:
    """DexiNed edge detector (Dense Extreme Inception Network).

    Uses OpenCV DNN backend with an ONNX model — no PyTorch required.

    Args:
        gray:         uint8 grayscale image (2-D).
        weights_path: Path to the .onnx model file.  If empty, uses the default
                      location ``models/dexined.onnx`` inside the project root.
        threshold:    Sigmoid threshold for binarising the edge probability map (0–1).
        device:       "cpu" (default) or "cuda" — selects cv2.dnn backend.

    Returns:
        uint8 binary edge mask (255 = edge, 0 = background).

    Model download:
        See ``models/download_dexined.py`` or run it directly:
            py -3.12 models/download_dexined.py
    """
    # Resolve model path
    path = weights_path.strip() or DEXINED_DEFAULT_WEIGHTS
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"DexiNed model not found: {path}\n"
            "Download by running:  py -3.12 models/download_dexined.py"
        )

    # Lazy-load cv2.dnn network (re-load only if path/device changes)
    global _dexined_cache
    cache_key = (path, device)
    if _dexined_cache.get("key") != cache_key:
        net = cv2.dnn.readNetFromONNX(path)
        if device == "cuda" and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _dexined_cache = {"key": cache_key, "net": net}

    net = _dexined_cache["net"]

    h_orig, w_orig = gray.shape

    # Pre-process: grayscale → BGR float32, mean subtraction, pad to ×32
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    bgr -= np.array([103.939, 116.779, 123.68], dtype=np.float32)

    # Pad to multiple of 32 (ONNX model requirement)
    h_pad = ((h_orig + 31) // 32) * 32
    w_pad = ((w_orig + 31) // 32) * 32
    if h_pad != h_orig or w_pad != w_orig:
        bgr = cv2.copyMakeBorder(bgr, 0, h_pad - h_orig, 0, w_pad - w_orig,
                                  cv2.BORDER_REFLECT)

    # Create blob: HWC → NCHW
    blob = cv2.dnn.blobFromImage(bgr, scalefactor=1.0, swapRB=False)
    net.setInput(blob)

    # Forward pass — output shape (1, 1, H_pad, W_pad)
    out = net.forward()
    fused = out[0, 0, :h_orig, :w_orig]   # remove padding, squeeze batch+channel

    # Sigmoid + threshold
    prob = 1.0 / (1.0 + np.exp(-fused.astype(np.float32)))
    mask = (prob > threshold).astype(np.uint8) * 255

    # Light morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask
