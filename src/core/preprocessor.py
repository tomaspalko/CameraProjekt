"""Preprocessing: grayscale conversion, CLAHE contrast enhancement, Gaussian blur."""
import cv2
import numpy as np


def preprocess(image: np.ndarray, clahe_clip: float = 2.0, blur_kernel: int = 5) -> np.ndarray:
    """Preprocess an image for ECC alignment.

    Args:
        image: Input image (grayscale 2D or color 3-channel uint8).
        clahe_clip: CLAHE clip limit (contrast enhancement strength).
        blur_kernel: Gaussian blur kernel size (must be odd); 1 = no blur.

    Returns:
        Preprocessed uint8 2D grayscale array, same spatial dimensions.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    if blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        result = cv2.GaussianBlur(enhanced, (k, k), 0)
    else:
        result = enhanced

    return result
