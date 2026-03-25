"""Batch processor: align a folder of images against a reference."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.core.preprocessor import preprocess
from src.core.aligner import align


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class BatchStats:
    """Aggregate statistics computed over all successful rows in a batch."""
    count_total: int
    count_ok: int
    count_error: int
    # dx_px
    dx_px_mean: Optional[float]
    dx_px_std:  Optional[float]
    dx_px_min:  Optional[float]
    dx_px_max:  Optional[float]
    # dy_px
    dy_px_mean: Optional[float]
    dy_px_std:  Optional[float]
    dy_px_min:  Optional[float]
    dy_px_max:  Optional[float]
    # dx_mm
    dx_mm_mean: Optional[float]
    dx_mm_std:  Optional[float]
    dx_mm_min:  Optional[float]
    dx_mm_max:  Optional[float]
    # dy_mm
    dy_mm_mean: Optional[float]
    dy_mm_std:  Optional[float]
    dy_mm_min:  Optional[float]
    dy_mm_max:  Optional[float]
    # angle_deg
    angle_deg_mean: Optional[float]
    angle_deg_std:  Optional[float]
    angle_deg_min:  Optional[float]
    angle_deg_max:  Optional[float]
    # confidence
    confidence_mean: Optional[float]
    confidence_std:  Optional[float]
    confidence_min:  Optional[float]
    confidence_max:  Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchResult:
    """Container returned by process_batch_profile."""
    rows: list
    stats: BatchStats


# ── Private helpers ───────────────────────────────────────────────────────────

def _run_batch(
    ref_raw,
    image_folder,
    mask,
    algorithm: str,
    max_iter: int,
    epsilon: float,
    mm_per_px: float,
) -> list[dict]:
    """Core loop: align every image in *image_folder* against pre-loaded *ref_raw*.

    Returns a list of result dicts. Each dict always contains:
        filename, status ("OK" | "ERROR"),
        dx_px, dy_px, dx_mm, dy_mm, angle_deg, confidence  (None on ERROR)
    """
    image_folder = Path(image_folder)
    ref_pre = preprocess(ref_raw)

    image_paths = sorted(
        p for p in image_folder.glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    )

    results = []
    for img_path in image_paths:
        entry: dict = {"filename": img_path.name}
        try:
            img_raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_raw is None:
                raise ValueError("cv2.imread returned None")
            img_pre = preprocess(img_raw)
            r = align(ref_pre, img_pre, mask=mask, algorithm=algorithm,
                      max_iter=max_iter, epsilon=epsilon)
            entry.update({
                "status":     "OK",
                "dx_px":      r["dx_px"],
                "dy_px":      r["dy_px"],
                "dx_mm":      r["dx_px"] * mm_per_px,
                "dy_mm":      r["dy_px"] * mm_per_px,
                "angle_deg":  r["angle_deg"],
                "confidence": r["confidence"],
            })
        except Exception:
            entry.update({
                "status":     "ERROR",
                "dx_px":      None,
                "dy_px":      None,
                "dx_mm":      None,
                "dy_mm":      None,
                "angle_deg":  None,
                "confidence": None,
            })
        results.append(entry)

    return results


def _compute_stats(rows: list[dict]) -> BatchStats:
    """Compute aggregate statistics over OK rows."""
    ok = [r for r in rows if r.get("status") == "OK"]
    n_total = len(rows)
    n_ok    = len(ok)
    n_error = n_total - n_ok

    if n_ok == 0:
        none4 = (None, None, None, None)
        return BatchStats(
            count_total=n_total, count_ok=0, count_error=n_error,
            dx_px_mean=None,      dx_px_std=None,      dx_px_min=None,      dx_px_max=None,
            dy_px_mean=None,      dy_px_std=None,      dy_px_min=None,      dy_px_max=None,
            dx_mm_mean=None,      dx_mm_std=None,      dx_mm_min=None,      dx_mm_max=None,
            dy_mm_mean=None,      dy_mm_std=None,      dy_mm_min=None,      dy_mm_max=None,
            angle_deg_mean=None,  angle_deg_std=None,  angle_deg_min=None,  angle_deg_max=None,
            confidence_mean=None, confidence_std=None, confidence_min=None, confidence_max=None,
        )

    def _stats(key):
        arr = np.array([r[key] for r in ok], dtype=float)
        return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))

    dx_px_mean,      dx_px_std,      dx_px_min,      dx_px_max      = _stats("dx_px")
    dy_px_mean,      dy_px_std,      dy_px_min,      dy_px_max      = _stats("dy_px")
    dx_mm_mean,      dx_mm_std,      dx_mm_min,      dx_mm_max      = _stats("dx_mm")
    dy_mm_mean,      dy_mm_std,      dy_mm_min,      dy_mm_max      = _stats("dy_mm")
    angle_deg_mean,  angle_deg_std,  angle_deg_min,  angle_deg_max  = _stats("angle_deg")
    confidence_mean, confidence_std, confidence_min, confidence_max = _stats("confidence")

    return BatchStats(
        count_total=n_total, count_ok=n_ok, count_error=n_error,
        dx_px_mean=dx_px_mean,           dx_px_std=dx_px_std,
        dx_px_min=dx_px_min,             dx_px_max=dx_px_max,
        dy_px_mean=dy_px_mean,           dy_px_std=dy_px_std,
        dy_px_min=dy_px_min,             dy_px_max=dy_px_max,
        dx_mm_mean=dx_mm_mean,           dx_mm_std=dx_mm_std,
        dx_mm_min=dx_mm_min,             dx_mm_max=dx_mm_max,
        dy_mm_mean=dy_mm_mean,           dy_mm_std=dy_mm_std,
        dy_mm_min=dy_mm_min,             dy_mm_max=dy_mm_max,
        angle_deg_mean=angle_deg_mean,   angle_deg_std=angle_deg_std,
        angle_deg_min=angle_deg_min,     angle_deg_max=angle_deg_max,
        confidence_mean=confidence_mean, confidence_std=confidence_std,
        confidence_min=confidence_min,   confidence_max=confidence_max,
    )


def _write_csv(results: list[dict], path: Path) -> None:
    fieldnames = [
        "filename", "dx_px", "dy_px", "dx_mm", "dy_mm",
        "angle_deg", "confidence", "status",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})


def _write_json(results: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _write_json_batch(batch_result: BatchResult, path: Path) -> None:
    data = {
        "results": batch_result.rows,
        "stats":   batch_result.stats.to_dict(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def process_batch(
    reference_path,
    image_folder,
    export_csv=None,
    export_json=None,
) -> list[dict]:
    """Align all images in *image_folder* against *reference_path*.

    Args:
        reference_path: Path to the reference image file.
        image_folder:   Directory containing images to process.
        export_csv:     Optional path to write a CSV results file.
        export_json:    Optional path to write a JSON results file.

    Returns:
        List of result dicts. Each dict contains:
            filename, status ("OK" | "ERROR"),
            dx_px, dy_px, dx_mm, dy_mm, angle_deg, confidence  (None on ERROR)

        With default mm_per_px=1.0 the mm values equal px values numerically.
    """
    reference_path = Path(reference_path)

    ref_raw = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    if ref_raw is None:
        raise ValueError(f"Cannot load reference image: {reference_path}")

    results = _run_batch(
        ref_raw, image_folder,
        mask=None, algorithm="ECC",
        max_iter=5000, epsilon=1e-8, mm_per_px=1.0,
    )

    # exclude the reference file itself if it ended up in the folder
    results = [r for r in results if r["filename"] != reference_path.name]

    if export_csv is not None:
        _write_csv(results, Path(export_csv))
    if export_json is not None:
        _write_json(results, Path(export_json))

    return results


def process_batch_profile(
    profile,
    image_folder,
    export_csv=None,
    export_json=None,
) -> BatchResult:
    """Align all images using a saved Profile (ROI, algorithm, calibration).

    Args:
        profile:      A validated Profile object.
        image_folder: Directory containing images to process.
        export_csv:   Optional path to write a CSV results file.
        export_json:  Optional path to write a JSON results file (includes stats).

    Returns:
        BatchResult with rows list and aggregate BatchStats.

    Raises:
        ValueError: if the profile is invalid or the reference image cannot be loaded.
    """
    errors = profile.validate()
    if errors:
        raise ValueError("Invalid profile: " + "; ".join(errors))

    ref_raw = cv2.imread(str(profile.reference_path), cv2.IMREAD_GRAYSCALE)
    if ref_raw is None:
        raise ValueError(f"Cannot load reference image: {profile.reference_path}")

    mask = profile.roi.create_mask(ref_raw.shape) if profile.roi is not None else None

    rows = _run_batch(
        ref_raw, image_folder,
        mask=mask,
        algorithm=profile.algorithm,
        max_iter=profile.ecc_max_iter,
        epsilon=profile.ecc_epsilon,
        mm_per_px=profile.scale_mm_per_px,
    )

    # exclude the reference file itself if it ended up in the batch folder
    ref_name = Path(profile.reference_path).name
    rows = [r for r in rows if r["filename"] != ref_name]

    stats = _compute_stats(rows)
    batch_result = BatchResult(rows=rows, stats=stats)

    if export_csv is not None:
        _write_csv(rows, Path(export_csv))
    if export_json is not None:
        _write_json_batch(batch_result, Path(export_json))

    return batch_result
