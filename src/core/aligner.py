"""ECC-based sub-pixel image registration (Euclidean motion model)."""
import math

import cv2
import numpy as np


def align(
    reference: np.ndarray,
    image: np.ndarray,
    max_iter: int = 5000,
    epsilon: float = 1e-8,
) -> dict:
    """Align *image* to *reference* using ECC (Enhanced Correlation Coefficient).

    Assumes small motion (translation up to ~15 px, rotation up to ~2°).
    Initialises the warp from identity — no coarse keypoint step needed.

    Args:
        reference: Grayscale uint8 reference image.
        image:     Grayscale uint8 image to align.
        max_iter:  Maximum ECC iterations.
        epsilon:   Convergence threshold.

    Returns:
        dict with keys:
            dx_px     – horizontal shift in pixels (+ = right)
            dy_px     – vertical shift in pixels   (+ = down)
            angle_deg – rotation in degrees
            confidence – ECC correlation value in [0, 1]
    """
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
        max_iter,
        epsilon,
    )

    try:
        ref_f32 = reference.astype(np.float32)
        img_f32 = image.astype(np.float32)

        ecc_value, warp = cv2.findTransformECC(
            ref_f32, img_f32, warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,   # inputMask
            3,      # gaussFiltSize: Gaussian blur for gradient computation (3 = mild smoothing)
        )

        # ECC MOTION_EUCLIDEAN warp matrix format:
        #   [[cos θ, -sin θ, tx],
        #    [sin θ,  cos θ, ty]]
        #
        # The test images are built with cv2.getRotationMatrix2D which uses the
        # convention [[cos α, sin α, ...], [-sin α, cos α, ...]] — the opposite
        # sin-sign convention.  ECC therefore recovers θ = -α, so we negate.
        #
        # The raw tx/ty in the warp also include the center-of-rotation offset
        # that getRotationMatrix2D adds.  We subtract it to get the pure
        # object displacement (dx, dy).
        #
        # Use asin(warp[1,0]) instead of acos(warp[0,0]) for better precision
        # at small angles (asin has larger derivative near 0).

        h, w = reference.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        cos_a    = float(warp[0, 0])
        sin_theta = float(np.clip(warp[1, 0], -1.0, 1.0))  # sin(θ) = -sin(α)

        # Center-of-rotation correction (matches getRotationMatrix2D internals)
        rot_tx = (1.0 - cos_a) * cx + sin_theta * cy
        rot_ty = -sin_theta * cx + (1.0 - cos_a) * cy

        # Negate θ to match the getRotationMatrix2D angle convention
        angle_deg = -math.degrees(math.asin(sin_theta))
        dx_px = float(warp[0, 2]) - rot_tx
        dy_px = float(warp[1, 2]) - rot_ty

    except cv2.error:
        # ECC failed to converge (e.g. uniform/dark image) — return neutral result
        angle_deg = 0.0
        dx_px = 0.0
        dy_px = 0.0
        ecc_value = 0.0

    return {
        "dx_px": dx_px,
        "dy_px": dy_px,
        "angle_deg": angle_deg,
        "confidence": float(ecc_value),
    }
