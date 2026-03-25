"""ROI (Region of Interest) — axis-aligned rectangle with mask generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ROI:
    """Axis-aligned rectangular region of interest in image pixel coordinates.

    (x0, y0) is the top-left corner; (x1, y1) is the bottom-right corner
    (exclusive, matching numpy/OpenCV slice convention).
    """

    x0: int
    y0: int
    x1: int
    y1: int

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_valid(self, shape: Optional[Tuple[int, ...]] = None) -> bool:
        """Return True if the ROI has positive area and fits within *shape*.

        Args:
            shape: Optional (height, width[, ...]) image shape. When given,
                   the ROI must lie entirely within [0, width) × [0, height).
        """
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            return False
        if shape is not None:
            h, w = shape[:2]
            if self.x0 < 0 or self.y0 < 0 or self.x1 > w or self.y1 > h:
                return False
        return True

    # ------------------------------------------------------------------
    # Mask generation
    # ------------------------------------------------------------------

    def create_mask(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Return a uint8 binary mask for *shape* with 255 inside the ROI.

        Pixels outside the ROI are 0. Coordinates are clamped to image bounds.
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x0 = max(0, self.x0)
        y0 = max(0, self.y0)
        x1 = min(w, self.x1)
        y1 = min(h, self.y1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
        return mask

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_dict(cls, d: dict) -> "ROI":
        return cls(x0=int(d["x0"]), y0=int(d["y0"]), x1=int(d["x1"]), y1=int(d["y1"]))
