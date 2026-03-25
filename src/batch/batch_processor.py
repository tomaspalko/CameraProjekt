"""Batch processor: align a folder of images against a reference."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

from src.core.preprocessor import preprocess
from src.core.aligner import align


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
        List of result dicts. Each dict has at minimum:
            filename, status ("OK" | "ERROR"),
            dx_px, dy_px, angle_deg, confidence  (None on ERROR)
    """
    reference_path = Path(reference_path)
    image_folder = Path(image_folder)

    ref_raw = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    if ref_raw is None:
        raise ValueError(f"Cannot load reference image: {reference_path}")
    ref_pre = preprocess(ref_raw)

    image_paths = sorted(
        p for p in image_folder.glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        and p.name != reference_path.name
    )

    results = []
    for img_path in image_paths:
        entry: dict = {"filename": img_path.name}
        try:
            img_raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_raw is None:
                raise ValueError("cv2.imread returned None")
            img_pre = preprocess(img_raw)
            r = align(ref_pre, img_pre)
            entry.update({
                "status": "OK",
                "dx_px": r["dx_px"],
                "dy_px": r["dy_px"],
                "angle_deg": r["angle_deg"],
                "confidence": r["confidence"],
            })
        except Exception:
            entry.update({
                "status": "ERROR",
                "dx_px": None,
                "dy_px": None,
                "angle_deg": None,
                "confidence": None,
            })
        results.append(entry)

    if export_csv is not None:
        _write_csv(results, Path(export_csv))

    if export_json is not None:
        _write_json(results, Path(export_json))

    return results


def _write_csv(results: list[dict], path: Path) -> None:
    fieldnames = ["filename", "dx_px", "dy_px", "angle_deg", "confidence", "status"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})


def _write_json(results: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
