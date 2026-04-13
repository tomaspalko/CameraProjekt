"""Preprocessing: grayscale conversion, CLAHE contrast enhancement, Gaussian blur."""
import cv2
import numpy as np


def preprocess(
    image: np.ndarray,
    clahe_clip: float = 2.0,
    blur_kernel: int = 5,
    auto_clahe: bool = False,
) -> np.ndarray:
    """Preprocess an image for ECC alignment.

    Args:
        image:      Input image (grayscale 2D or color 3-channel uint8).
        clahe_clip: CLAHE clip limit (contrast enhancement strength).
                    Ignored when *auto_clahe* is True.
        blur_kernel: Gaussian blur kernel size (must be odd); 1 = no blur.
        auto_clahe: When True, derive the clip limit from the image's gradient
                    statistics instead of using the fixed *clahe_clip* value.
                    Adapts contrast enhancement to the local image contrast:
                    low-contrast images receive a higher clip limit and
                    high-contrast images receive a lower one.
                    Formula: clip = clamp(2.0 * 50.0 / grad_std, 1.0, 4.0)

    Returns:
        Preprocessed uint8 2D grayscale array, same spatial dimensions.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if auto_clahe:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_std = float(cv2.magnitude(gx, gy).std()) + 1e-8
        clahe_clip = float(np.clip(2.0 * 50.0 / grad_std, 1.0, 4.0))

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    if blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        result = cv2.GaussianBlur(enhanced, (k, k), 0)
    else:
        result = enhanced

    return result
