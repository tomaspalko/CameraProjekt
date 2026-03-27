"""Integration tests — full pipeline end-to-end."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.preprocessor import preprocess
from src.core.aligner import align
from src.core.calibration import Calibration
from src.core.result import AlignResult
from src.core.roi import ROI
from src.config.profile import Profile
from src.batch.batch_processor import (
    process_batch,
    process_batch_profile,
    BatchResult,
    BatchStats,
)

from tests.synthetic.generator import make_image_pair


# ── Full pipeline ─────────────────────────────────────────────────────────────

def test_full_pipeline_returns_all_fields():
    ref, img = make_image_pair(3.0, -2.0, 0.5)
    pre_ref = preprocess(ref)
    pre_img = preprocess(img)
    result  = align(pre_ref, pre_img)

    cal = Calibration(mm_per_px=0.05)
    ar  = AlignResult.from_dict(result, cal)

    assert hasattr(ar, "dx_px")
    assert hasattr(ar, "dy_px")
    assert hasattr(ar, "angle_deg")
    assert hasattr(ar, "dx_mm")
    assert hasattr(ar, "dy_mm")
    assert hasattr(ar, "confidence")

def test_full_pipeline_mm_conversion():
    ref, img = make_image_pair(10.0, 0.0, 0.0)
    pre_ref  = preprocess(ref)
    pre_img  = preprocess(img)
    result   = align(pre_ref, pre_img)

    mm_per_px = 0.05
    cal = Calibration(mm_per_px=mm_per_px)
    ar  = AlignResult.from_dict(result, cal)

    assert abs(ar.dx_mm - ar.dx_px * mm_per_px) < 1e-6

def test_full_pipeline_accuracy():
    dx, dy, angle = 4.0, -3.0, 0.7
    ref, img = make_image_pair(dx, dy, angle)
    result   = align(preprocess(ref), preprocess(img))

    assert abs(result["dx_px"]    - dx)    < 0.1
    assert abs(result["dy_px"]    - dy)    < 0.1
    assert abs(result["angle_deg"] - angle) < 0.05


# ── Batch processor (process_batch) ───────────────────────────────────────────

@pytest.fixture
def batch_folder(tmp_path):
    """Create a folder of 20 synthetic images with a reference.

    All test images are derived from the same base texture (seed=0) so that ECC
    can align them against the reference.  Displacements are seeded for
    reproducibility.
    """
    ref, _ = make_image_pair(0, 0, 0, seed=0)
    ref_path = tmp_path / "reference.png"
    cv2.imwrite(str(ref_path), ref)

    rng = np.random.default_rng(42)
    displacements = []
    for i in range(20):
        dx    = float(rng.uniform(-8, 8))
        dy    = float(rng.uniform(-8, 8))
        angle = float(rng.uniform(-1, 1))
        _, img = make_image_pair(dx, dy, angle, seed=0)   # same base as reference
        cv2.imwrite(str(tmp_path / f"img_{i:03d}.png"), img)
        displacements.append({"dx": dx, "dy": dy, "angle": angle})

    return tmp_path, ref_path, displacements


def test_batch_processes_all_images(batch_folder):
    folder, ref_path, _ = batch_folder
    results = process_batch(ref_path, folder)
    image_files = list(folder.glob("img_*.png"))
    assert len(results) == len(image_files), "Not all images were processed"

def test_batch_no_crash_on_corrupt_image(batch_folder, tmp_path):
    folder, ref_path, _ = batch_folder
    # inject a corrupt file
    (folder / "corrupt.png").write_bytes(b"not an image")
    results = process_batch(ref_path, folder)
    corrupt = [r for r in results if "corrupt" in r["filename"]]
    assert len(corrupt) == 1
    assert corrupt[0]["status"] == "ERROR"

def test_batch_csv_export(batch_folder, tmp_path):
    folder, ref_path, _ = batch_folder
    csv_path = tmp_path / "results.csv"
    process_batch(ref_path, folder, export_csv=csv_path)

    assert csv_path.exists()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 20
    for row in rows:
        assert "dx_px" in row
        assert "dy_px" in row
        assert "angle_deg" in row
        assert "status" in row

def test_batch_json_export(batch_folder, tmp_path):
    folder, ref_path, _ = batch_folder
    json_path = tmp_path / "results.json"
    process_batch(ref_path, folder, export_json=json_path)

    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
    assert len(data) == 20

def test_batch_accuracy_rms(batch_folder):
    """RMS error across all batch results must stay within tolerance."""
    folder, ref_path, displacements = batch_folder
    results = process_batch(ref_path, folder)

    ok_results = [r for r in results if r["status"] == "OK"]
    assert len(ok_results) >= 18, "Too many failures in batch"

    dx_sq, dy_sq, a_sq = [], [], []
    for r, gt in zip(
        sorted(ok_results, key=lambda x: x["filename"]),
        sorted(displacements, key=lambda x: x.get("filename", ""))
    ):
        dx_sq.append((r["dx_px"] - gt["dx"]) ** 2)
        dy_sq.append((r["dy_px"] - gt["dy"]) ** 2)
        a_sq.append((r["angle_deg"] - gt["angle"]) ** 2)

    assert np.sqrt(np.mean(dx_sq)) < 0.15, "Batch RMS dx too high"
    assert np.sqrt(np.mean(dy_sq)) < 0.15, "Batch RMS dy too high"


# ── Batch processor (process_batch_profile) — Phase 3 ────────────────────────

@pytest.fixture
def profile_batch_folder(tmp_path):
    """Like batch_folder but returns a Profile configured for the folder."""
    ref, _ = make_image_pair(0, 0, 0, size=(256, 256), seed=0)
    ref_path = tmp_path / "reference.png"
    cv2.imwrite(str(ref_path), ref)

    rng = np.random.default_rng(7)
    displacements = []
    for i in range(20):
        dx    = float(rng.uniform(-5, 5))
        dy    = float(rng.uniform(-5, 5))
        angle = float(rng.uniform(-0.8, 0.8))
        _, img = make_image_pair(dx, dy, angle, seed=0)   # same base as reference
        cv2.imwrite(str(tmp_path / f"img_{i:03d}.png"), img)
        displacements.append({"dx": dx, "dy": dy, "angle": angle})

    profile = Profile(
        name="test_profile",
        reference_path=str(ref_path),
        roi=ROI(20, 20, 236, 236),
        scale_mm_per_px=0.05,
        algorithm="ECC",
        ecc_max_iter=2000,
        ecc_epsilon=1e-8,
    )
    return tmp_path, ref_path, profile, displacements


def test_batch_profile_returns_batch_result(profile_batch_folder, tmp_path):
    folder, _, profile, _ = profile_batch_folder
    result = process_batch_profile(profile, folder)
    assert isinstance(result, BatchResult)
    assert isinstance(result.stats, BatchStats)
    assert isinstance(result.rows, list)


def test_batch_profile_stats_counts(profile_batch_folder, tmp_path):
    folder, _, profile, _ = profile_batch_folder
    result = process_batch_profile(profile, folder)
    assert result.stats.count_total == 20
    assert result.stats.count_ok >= 18
    assert result.stats.count_ok + result.stats.count_error == result.stats.count_total


def test_batch_profile_stats_values(profile_batch_folder, tmp_path):
    folder, _, profile, _ = profile_batch_folder
    result = process_batch_profile(profile, folder)
    stats = result.stats
    assert isinstance(stats.dx_px_mean, float)
    assert stats.dx_px_std >= 0.0
    assert stats.confidence_mean is not None
    assert 0.0 <= stats.confidence_min <= stats.confidence_max <= 1.0


def test_batch_profile_mm_in_rows(profile_batch_folder, tmp_path):
    folder, _, profile, _ = profile_batch_folder
    result = process_batch_profile(profile, folder)
    ok_rows = [r for r in result.rows if r["status"] == "OK"]
    assert len(ok_rows) > 0
    for row in ok_rows:
        assert "dx_mm" in row
        assert "dy_mm" in row
        assert abs(row["dx_mm"] - row["dx_px"] * profile.scale_mm_per_px) < 1e-9
        assert abs(row["dy_mm"] - row["dy_px"] * profile.scale_mm_per_px) < 1e-9


def test_batch_profile_csv_has_mm_columns(profile_batch_folder, tmp_path):
    folder, _, profile, _ = profile_batch_folder
    csv_path = tmp_path / "out.csv"
    process_batch_profile(profile, folder, export_csv=csv_path)

    assert csv_path.exists()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 20
    for row in rows:
        assert "dx_mm" in row
        assert "dy_mm" in row


def test_batch_profile_json_has_stats(profile_batch_folder, tmp_path):
    folder, _, profile, _ = profile_batch_folder
    json_path = tmp_path / "out.json"
    process_batch_profile(profile, folder, export_json=json_path)

    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
    assert "results" in data
    assert "stats" in data
    assert len(data["results"]) == 20
    assert "count_total" in data["stats"]
    assert "dx_px_mean" in data["stats"]


def test_batch_profile_invalid_profile_raises(tmp_path):
    bad_profile = Profile(name="")   # empty name → invalid
    with pytest.raises(ValueError, match="Invalid profile"):
        process_batch_profile(bad_profile, tmp_path)


# ── Profile save/load ─────────────────────────────────────────────────────────

def test_profile_round_trip(tmp_path):
    from src.config.config_manager import ConfigManager
    cm = ConfigManager(tmp_path / "profiles")
    profile = {
        "name": "test_profile",
        "reference_path": "data/reference/ref.png",
        "roi": [[10, 10], [200, 10], [200, 200], [10, 200]],
        "scale_mm_per_px": 0.05,
        "ecc": {"max_iter": 1000, "epsilon": 1e-8, "pyramid_levels": 1},
    }
    cm.save(profile)
    loaded = cm.load("test_profile")
    assert loaded["name"] == profile["name"]
    assert loaded["scale_mm_per_px"] == profile["scale_mm_per_px"]
    assert loaded["ecc"]["max_iter"] == 1000
