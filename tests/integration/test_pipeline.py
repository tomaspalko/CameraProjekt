"""Integration tests — full pipeline end-to-end."""
import pytest

from src.core.preprocessor import preprocess
from src.core.aligner import align
from src.core.calibration import Calibration
from src.core.result import AlignResult

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


# ── Profile save/load ─────────────────────────────────────────────────────────

def test_profile_round_trip(tmp_path):
    from src.config.config_manager import ConfigManager
    from src.config.profile import Profile
    cm = ConfigManager(tmp_path / "profiles")
    p = Profile(name="test_profile", scale_mm_per_px=0.05, ecc_max_iter=1000)
    cm.save_profile(p)
    loaded = cm.load_profile("test_profile")
    assert loaded.name == "test_profile"
    assert loaded.scale_mm_per_px == 0.05
    assert loaded.ecc_max_iter == 1000
    # ID auto-assigned
    assert loaded.id == 1

def test_profile_round_trip_legacy(tmp_path):
    """Legacy dict-based API still works."""
    from src.config.config_manager import ConfigManager
    cm = ConfigManager(tmp_path / "profiles")
    profile = {
        "name": "legacy_profile",
        "reference_path": "data/reference/ref.png",
        "scale_mm_per_px": 0.05,
        "ecc": {"max_iter": 1000, "epsilon": 1e-8},
    }
    cm.save(profile)
    loaded = cm.load("legacy_profile")
    assert loaded["name"] == profile["name"]
    assert loaded["scale_mm_per_px"] == profile["scale_mm_per_px"]
    assert loaded["ecc"]["max_iter"] == 1000
