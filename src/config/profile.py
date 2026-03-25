"""Job configuration profile — structured, validated, serialisable."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.core.roi import ROI


@dataclass
class Profile:
    """All parameters needed to run one alignment job.

    Attributes:
        name:             Unique profile name (used as filename).
        reference_path:   Path to the reference image file.
        roi:              Optional rectangle ROI; None means full image.
        scale_mm_per_px:  Physical scale factor for mm conversion.
        algorithm:        Alignment algorithm — "ECC" or "POC".
        ecc_max_iter:     Maximum ECC iterations.
        ecc_epsilon:      ECC convergence threshold.
    """

    name: str
    reference_path: str = ""
    roi: Optional[ROI] = None
    scale_mm_per_px: float = 1.0
    algorithm: str = "ECC"
    ecc_max_iter: int = 2000
    ecc_epsilon: float = 1e-8

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of validation error strings.  Empty list = valid."""
        errors: list[str] = []
        if not self.name or not self.name.strip():
            errors.append("Profile name cannot be empty.")
        if self.scale_mm_per_px <= 0:
            errors.append("scale_mm_per_px must be positive.")
        if self.algorithm not in ("ECC", "POC"):
            errors.append(f"Unknown algorithm '{self.algorithm}'. Expected 'ECC' or 'POC'.")
        if self.roi is not None and not self.roi.is_valid():
            errors.append("ROI is invalid (x1 must be > x0 and y1 must be > y0).")
        if self.ecc_max_iter < 1:
            errors.append("ecc_max_iter must be >= 1.")
        if self.ecc_epsilon <= 0:
            errors.append("ecc_epsilon must be positive.")
        return errors

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "reference_path": self.reference_path,
            "roi": self.roi.to_dict() if self.roi is not None else None,
            "scale_mm_per_px": self.scale_mm_per_px,
            "algorithm": self.algorithm,
            "ecc": {
                "max_iter": self.ecc_max_iter,
                "epsilon": self.ecc_epsilon,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        roi_data = d.get("roi")
        roi = ROI.from_dict(roi_data) if roi_data is not None else None
        ecc = d.get("ecc", {})
        return cls(
            name=d["name"],
            reference_path=d.get("reference_path", ""),
            roi=roi,
            scale_mm_per_px=float(d.get("scale_mm_per_px", 1.0)),
            algorithm=d.get("algorithm", "ECC"),
            ecc_max_iter=int(ecc.get("max_iter", 2000)),
            ecc_epsilon=float(ecc.get("epsilon", 1e-8)),
        )
