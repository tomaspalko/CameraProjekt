"""AlignResult dataclass — container for alignment output with mm conversion."""
from __future__ import annotations

from dataclasses import dataclass

from src.core.calibration import Calibration


@dataclass
class AlignResult:
    dx_px: float
    dy_px: float
    angle_deg: float
    dx_mm: float
    dy_mm: float
    confidence: float
    ncc_score: float = 0.0
    elapsed_s: float = 0.0

    @classmethod
    def from_dict(cls, d: dict, cal: Calibration) -> "AlignResult":
        """Create an AlignResult from an align() output dict and a Calibration."""
        return cls(
            dx_px=d["dx_px"],
            dy_px=d["dy_px"],
            angle_deg=d["angle_deg"],
            dx_mm=d["dx_px"] * cal.mm_per_px,
            dy_mm=d["dy_px"] * cal.mm_per_px,
            confidence=d["confidence"],
            ncc_score=d.get("ncc_score", 0.0),
            elapsed_s=d.get("elapsed_s", 0.0),
        )
