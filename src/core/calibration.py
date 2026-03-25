"""Camera calibration: pixel-to-millimetre scale factor."""
from dataclasses import dataclass


@dataclass
class Calibration:
    """Holds the scale factor used to convert pixel measurements to millimetres."""
    mm_per_px: float = 1.0
