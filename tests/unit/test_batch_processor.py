"""Unit tests for batch_processor.py."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.batch.batch_processor import process_batch, BatchStats
from tests.synthetic.generator import make_textured, transform


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_batch(tmp_path):
    """Create a temp folder with 3 valid images + a reference image."""
    ref = make_textured(size=(128, 128), seed=0)
    ref_path = tmp_path / "ref.png"
    cv2.imwrite(str(ref_path), ref)

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    for i in range(3):
        img = transform(ref, dx=float(i), dy=0.0, angle_deg=0.0)
        cv2.imwrite(str(batch_dir / f"img_{i:02d}.png"), img)

    return ref_path, batch_dir


# ── Return type ───────────────────────────────────────────────────────────────

def test_process_batch_returns_list(tmp_batch):
    ref_path, batch_dir = tmp_batch
    result = process_batch(ref_path, batch_dir)
    assert isinstance(result, list)


def test_process_batch_length(tmp_batch):
    ref_path, batch_dir = tmp_batch
    result = process_batch(ref_path, batch_dir)
    assert len(result) == 3


# ── Row structure ─────────────────────────────────────────────────────────────

REQUIRED_KEYS = {"filename", "status", "dx_px", "dy_px", "dx_mm", "dy_mm", "angle_deg", "confidence"}

def test_process_batch_ok_rows_have_required_keys(tmp_batch):
    ref_path, batch_dir = tmp_batch
    rows = process_batch(ref_path, batch_dir)
    for row in rows:
        assert REQUIRED_KEYS <= row.keys(), f"Missing keys in row: {row}"


def test_process_batch_ok_rows_numeric_values(tmp_batch):
    ref_path, batch_dir = tmp_batch
    rows = process_batch(ref_path, batch_dir)
    for row in rows:
        if row["status"] == "OK":
            for key in ("dx_px", "dy_px", "dx_mm", "dy_mm", "angle_deg", "confidence"):
                assert isinstance(row[key], float), f"{key} is not float: {row[key]}"
            assert 0.0 <= row["confidence"] <= 1.0


# ── Corrupt image handling ────────────────────────────────────────────────────

def test_process_batch_corrupt_image_gives_error_row(tmp_path):
    ref = make_textured(size=(128, 128), seed=0)
    ref_path = tmp_path / "ref.png"
    cv2.imwrite(str(ref_path), ref)

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    (batch_dir / "corrupt.png").write_bytes(b"this is not an image")

    rows = process_batch(ref_path, batch_dir)
    assert len(rows) == 1
    assert rows[0]["status"] == "ERROR"
    for key in ("dx_px", "dy_px", "dx_mm", "dy_mm", "angle_deg", "confidence"):
        assert rows[0][key] is None


# ── Reference exclusion ───────────────────────────────────────────────────────

def test_process_batch_excludes_reference_when_in_folder(tmp_path):
    """Reference image accidentally copied into the batch folder is excluded."""
    ref = make_textured(size=(128, 128), seed=0)
    ref_path = tmp_path / "ref.png"
    cv2.imwrite(str(ref_path), ref)

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    # Copy reference into batch folder
    cv2.imwrite(str(batch_dir / "ref.png"), ref)
    # One real batch image
    img = transform(ref, dx=2.0, dy=0.0, angle_deg=0.0)
    cv2.imwrite(str(batch_dir / "img_01.png"), img)

    rows = process_batch(ref_path, batch_dir)
    filenames = [r["filename"] for r in rows]
    assert "ref.png" not in filenames
    assert len(rows) == 1


# ── Algorithm parameter ───────────────────────────────────────────────────────

def test_process_batch_algorithm_poc_smoke(tmp_batch):
    """algorithm='POC' must not raise and must return valid rows."""
    ref_path, batch_dir = tmp_batch
    rows = process_batch(ref_path, batch_dir, algorithm="POC")
    assert isinstance(rows, list)
    assert len(rows) == 3
    for row in rows:
        assert REQUIRED_KEYS <= row.keys()


# ── CSV export ────────────────────────────────────────────────────────────────

def test_process_batch_csv_export(tmp_batch, tmp_path):
    ref_path, batch_dir = tmp_batch
    csv_path = tmp_path / "out.csv"
    process_batch(ref_path, batch_dir, export_csv=csv_path)

    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    assert "filename" in headers
    assert "dx_px" in headers
    assert "status" in headers
    assert len(rows) == 3


# ── JSON export ───────────────────────────────────────────────────────────────

def test_process_batch_json_export(tmp_batch, tmp_path):
    ref_path, batch_dir = tmp_batch
    json_path = tmp_path / "out.json"
    process_batch(ref_path, batch_dir, export_json=json_path)

    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert isinstance(data, list)
    assert len(data) == 3


# ── BatchStats structure ──────────────────────────────────────────────────────

def test_batch_stats_has_required_fields():
    """BatchStats dataclass must expose count fields and per-metric stats."""
    stats = BatchStats(
        count_total=5, count_ok=4, count_error=1,
        dx_px_mean=1.0,  dx_px_std=0.1,  dx_px_min=0.5,  dx_px_max=1.5,
        dy_px_mean=0.0,  dy_px_std=0.0,  dy_px_min=0.0,  dy_px_max=0.0,
        dx_mm_mean=1.0,  dx_mm_std=0.1,  dx_mm_min=0.5,  dx_mm_max=1.5,
        dy_mm_mean=0.0,  dy_mm_std=0.0,  dy_mm_min=0.0,  dy_mm_max=0.0,
        angle_deg_mean=0.1, angle_deg_std=0.01, angle_deg_min=0.0, angle_deg_max=0.2,
        confidence_mean=0.9, confidence_std=0.05, confidence_min=0.8, confidence_max=1.0,
    )
    assert stats.count_total == 5
    assert stats.count_ok == 4
    assert stats.count_error == 1
    d = stats.to_dict()
    for metric in ("dx_px", "dy_px", "dx_mm", "dy_mm", "angle_deg", "confidence"):
        for suffix in ("mean", "std", "min", "max"):
            assert f"{metric}_{suffix}" in d
