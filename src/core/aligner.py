"""ECC-based sub-pixel image registration (Euclidean motion model)."""
import math
import time

import cv2
import numpy as np


def _ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation coefficient in [-1, 1].

    Algorithm-independent quality metric: compare reference to the aligned
    (warped) image over the full image area.  A score near 1.0 means near-
    perfect alignment; near 0 means no correlation; negative means inverted.
    """
    fa = a.astype(np.float32)
    fb = b.astype(np.float32)
    fa -= fa.mean()
    fb -= fb.mean()
    denom = float(np.sqrt(np.sum(fa ** 2) * np.sum(fb ** 2))) + 1e-10
    return float(np.clip(np.sum(fa * fb) / denom, -1.0, 1.0))


def align(
    reference: np.ndarray,
    image: np.ndarray,
    max_iter: int = 5000,
    epsilon: float = 1e-8,
    mask: np.ndarray | None = None,
    algorithm: str = "ECC",
    two_pass: bool = False,
    gauss_filt_size: int = 3,
) -> dict:
    """Align *image* to *reference* using ECC (Enhanced Correlation Coefficient).

    Assumes small motion (translation up to ~15 px, rotation up to ~2°).
    Initialises the warp from identity — no coarse keypoint step needed.

    Args:
        reference: Grayscale uint8 reference image.
        image:     Grayscale uint8 image to align.
        max_iter:  Maximum ECC iterations.
        epsilon:   Convergence threshold.
        mask:      Optional uint8 mask (255 = use pixel, 0 = ignore).
        algorithm: "ECC" (default) or "POC".
        two_pass:  If True, run a second ECC pass from the first-pass result
                   with gaussFiltSize=1 and tighter epsilon (1e-10).
                   Improves sub-pixel accuracy at ~15% extra runtime cost.

    Returns:
        dict with keys:
            dx_px     – horizontal shift in pixels (+ = right)
            dy_px     – vertical shift in pixels   (+ = down)
            angle_deg – rotation in degrees
            confidence – ECC correlation value in [0, 1]
    """
    t0 = time.perf_counter()
    if algorithm == "POC":
        from src.core.poc_correlator import poc_align
        result = poc_align(reference, image, mask=mask)
        result["elapsed_s"] = time.perf_counter() - t0
        return result
    if algorithm != "ECC":
        raise ValueError(f"Unknown algorithm '{algorithm}'. Expected 'ECC' or 'POC'.")

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
        max_iter,
        epsilon,
    )

    try:
        ref_f32 = reference.astype(np.float32)
        img_f32 = image.astype(np.float32)

        input_mask = mask if mask is not None else None
        # gaussFiltSize must be odd and >= 1; enforce that
        gfs = max(1, int(gauss_filt_size))
        if gfs % 2 == 0:
            gfs += 1
        ecc_value, warp = cv2.findTransformECC(
            ref_f32, img_f32, warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            input_mask,
            gfs,
        )

        # Optional second pass: use first-pass result as init, remove gradient
        # smoothing (gaussFiltSize=1) and tighten epsilon for sub-pixel refinement.
        if two_pass:
            criteria2 = (
                cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                500,
                1e-10,
            )
            ecc_value, warp = cv2.findTransformECC(
                ref_f32, img_f32, warp,
                cv2.MOTION_EUCLIDEAN,
                criteria2,
                input_mask,
                1,  # gaussFiltSize=1: no gradient smoothing → finer sub-pixel convergence
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

        # NCC score: warp the image using the final warp and compare to reference
        aligned = cv2.warpAffine(img_f32, warp, (reference.shape[1], reference.shape[0]))
        ncc = _ncc_score(ref_f32, aligned)

    except cv2.error:
        # ECC failed to converge (e.g. uniform/dark image) — return neutral result
        angle_deg = 0.0
        dx_px = 0.0
        dy_px = 0.0
        ecc_value = 0.0
        ncc = 0.0

    return {
        "dx_px": dx_px,
        "dy_px": dy_px,
        "angle_deg": angle_deg,
        "confidence": float(ecc_value),
        "ncc_score": float(ncc),
        "elapsed_s": time.perf_counter() - t0,
    }
