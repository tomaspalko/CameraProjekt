"""Unit tests for Profile and enhanced ConfigManager."""
import pytest
from src.core.roi import ROI
from src.config.profile import Profile
from src.config.config_manager import ConfigManager


# ── Profile.validate ─────────────────────────────────────────────────────────

def test_valid_profile_no_errors():
    p = Profile(name="job_01", scale_mm_per_px=0.05)
    assert p.validate() == []

def test_valid_profile_with_roi():
    p = Profile(name="job_01", roi=ROI(0, 0, 100, 100), scale_mm_per_px=0.05)
    assert p.validate() == []

def test_invalid_empty_name():
    errors = Profile(name="", scale_mm_per_px=0.05).validate()
    assert any("name" in e.lower() for e in errors)

def test_invalid_whitespace_name():
    errors = Profile(name="   ", scale_mm_per_px=0.05).validate()
    assert any("name" in e.lower() for e in errors)

def test_invalid_scale_zero():
    errors = Profile(name="x", scale_mm_per_px=0.0).validate()
    assert any("scale" in e.lower() for e in errors)

def test_invalid_scale_negative():
    errors = Profile(name="x", scale_mm_per_px=-1.0).validate()
    assert any("scale" in e.lower() for e in errors)

def test_invalid_algorithm():
    errors = Profile(name="x", scale_mm_per_px=0.05, algorithm="SIFT").validate()
    assert any("algorithm" in e.lower() for e in errors)

def test_valid_algorithm_poc():
    errors = Profile(name="x", scale_mm_per_px=0.05, algorithm="POC").validate()
    assert errors == []

def test_invalid_roi():
    errors = Profile(name="x", roi=ROI(100, 0, 50, 100), scale_mm_per_px=0.05).validate()
    assert any("roi" in e.lower() for e in errors)

def test_invalid_ecc_max_iter():
    errors = Profile(name="x", scale_mm_per_px=0.05, ecc_max_iter=0).validate()
    assert any("iter" in e.lower() for e in errors)

def test_invalid_ecc_epsilon():
    errors = Profile(name="x", scale_mm_per_px=0.05, ecc_epsilon=0.0).validate()
    assert any("epsilon" in e.lower() for e in errors)

def test_multiple_errors_returned():
    errors = Profile(name="", scale_mm_per_px=-1.0).validate()
    assert len(errors) >= 2


# ── Profile serialisation ─────────────────────────────────────────────────────

def test_to_dict_keys():
    p = Profile(name="job_01", scale_mm_per_px=0.05)
    d = p.to_dict()
    for key in ("name", "reference_path", "roi", "scale_mm_per_px", "algorithm", "ecc"):
        assert key in d

def test_to_dict_ecc_keys():
    d = Profile(name="x").to_dict()
    assert "max_iter" in d["ecc"]
    assert "epsilon" in d["ecc"]

def test_to_dict_roi_none():
    d = Profile(name="x").to_dict()
    assert d["roi"] is None

def test_to_dict_roi_present():
    p = Profile(name="x", roi=ROI(1, 2, 3, 4))
    d = p.to_dict()
    assert d["roi"] == {"x0": 1, "y0": 2, "x1": 3, "y1": 4}

def test_from_dict_round_trip():
    p = Profile(
        name="job_02",
        reference_path="data/ref.png",
        roi=ROI(10, 20, 200, 180),
        scale_mm_per_px=0.05,
        algorithm="ECC",
        ecc_max_iter=3000,
        ecc_epsilon=1e-9,
    )
    restored = Profile.from_dict(p.to_dict())
    assert restored == p

def test_from_dict_defaults():
    p = Profile.from_dict({"name": "x"})
    assert p.scale_mm_per_px == 1.0
    assert p.algorithm == "ECC"
    assert p.roi is None
    assert p.ecc_max_iter == 2000

def test_from_dict_with_roi():
    d = {"name": "x", "roi": {"x0": 5, "y0": 6, "x1": 50, "y1": 60}}
    p = Profile.from_dict(d)
    assert p.roi == ROI(5, 6, 50, 60)

def test_from_dict_without_roi():
    p = Profile.from_dict({"name": "x", "roi": None})
    assert p.roi is None


# ── ConfigManager profile API ─────────────────────────────────────────────────

@pytest.fixture
def cm(tmp_path):
    return ConfigManager(tmp_path / "profiles")


def test_save_and_load_profile(cm):
    p = Profile(name="job_01", scale_mm_per_px=0.05, roi=ROI(0, 0, 100, 100))
    cm.save_profile(p)
    loaded = cm.load_profile("job_01")
    assert loaded == p

def test_list_profiles_empty(cm):
    assert cm.list_profiles() == []

def test_list_profiles_after_save(cm):
    cm.save_profile(Profile(name="alpha"))
    cm.save_profile(Profile(name="beta"))
    cm.save_profile(Profile(name="gamma"))
    assert cm.list_profiles() == ["alpha", "beta", "gamma"]

def test_list_profiles_sorted(cm):
    cm.save_profile(Profile(name="c"))
    cm.save_profile(Profile(name="a"))
    cm.save_profile(Profile(name="b"))
    assert cm.list_profiles() == ["a", "b", "c"]

def test_delete_profile(cm):
    cm.save_profile(Profile(name="to_delete"))
    cm.delete_profile("to_delete")
    assert "to_delete" not in cm.list_profiles()

def test_delete_missing_profile_raises(cm):
    with pytest.raises(FileNotFoundError):
        cm.delete_profile("nonexistent")

def test_load_profile_after_delete_raises(cm):
    cm.save_profile(Profile(name="temp"))
    cm.delete_profile("temp")
    with pytest.raises(FileNotFoundError):
        cm.load_profile("temp")

def test_overwrite_profile(cm):
    cm.save_profile(Profile(name="job", scale_mm_per_px=0.05))
    cm.save_profile(Profile(name="job", scale_mm_per_px=0.10))
    loaded = cm.load_profile("job")
    assert loaded.scale_mm_per_px == 0.10

def test_legacy_save_load_still_works(cm):
    """Existing dict-based API must remain functional."""
    d = {"name": "legacy", "scale_mm_per_px": 0.05, "ecc": {"max_iter": 1000, "epsilon": 1e-8}}
    cm.save(d)
    loaded = cm.load("legacy")
    assert loaded["scale_mm_per_px"] == 0.05


# ── Aligner algorithm parameter ───────────────────────────────────────────────

def test_align_poc_returns_result():
    import numpy as np
    from src.core.aligner import align
    ref = np.zeros((64, 64), dtype=np.uint8)
    result = align(ref, ref, algorithm="POC")
    assert "dx_px" in result

def test_align_unknown_algorithm_raises():
    import numpy as np
    from src.core.aligner import align
    ref = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError):
        align(ref, ref, algorithm="SIFT")

def test_align_with_mask():
    import cv2
    import numpy as np
    from src.core.aligner import align
    rng = np.random.default_rng(0)
    ref = rng.integers(40, 220, (128, 128), dtype=np.uint8).astype(np.float32)
    for _ in range(10):
        cx, cy = rng.integers(10, 118), rng.integers(10, 118)
        cv2.circle(ref, (int(cx), int(cy)), 8, float(rng.integers(50, 200)), -1)
    ref = np.clip(ref, 0, 255).astype(np.uint8)
    roi = ROI(10, 10, 118, 118)
    mask = roi.create_mask(ref.shape)
    M = cv2.getRotationMatrix2D((64, 64), 0.0, 1.0)
    M[0, 2] += 2.0
    img = cv2.warpAffine(ref, M, (128, 128))
    result = align(ref, img, mask=mask)
    assert "dx_px" in result
    assert abs(result["dx_px"] - 2.0) < 0.5
