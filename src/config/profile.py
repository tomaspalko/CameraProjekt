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
        id:               Auto-increment numeric ID (0 = not yet assigned).
        reference_path:   Path to the reference image file.
        roi:              Optional rectangle ROI; None means full image.
        scale_mm_per_px:  Physical scale factor for mm conversion.
        algorithm:        Alignment algorithm — "ECC" or "POC".
        ecc_max_iter:     Maximum ECC iterations.
        ecc_epsilon:      ECC convergence threshold.
        auto_clahe:       Derive CLAHE clip limit from gradient statistics
                          instead of a fixed value.  Off by default.
    """

    name: str
    id: int = 0
    reference_path: str = ""
    roi: Optional[ROI] = None
    scale_mm_per_px: float = 1.0
    algorithm: str = "ECC"
    ecc_max_iter: int = 2000
    ecc_epsilon: float = 1e-8
    ecc_gauss_filt_size: int = 3   # gaussFiltSize pre findTransformECC (1=vypnutý, 3/5/7)
    auto_clahe: bool = False

    # ---- edge detection ----
    edge_method: str = "Canny"       # Canny | Scharr | LoG | PhaseCongruency

    # Canny params
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    canny_blur: int = 3

    # Scharr params
    scharr_threshold: int = 30
    scharr_blur: int = 3

    # LoG params
    log_sigma: float = 1.5
    log_threshold: int = 10
    log_blur: int = 3

    # Phase Congruency params
    pc_nscale: int = 4
    pc_min_wavelength: int = 6
    pc_mult: float = 2.1
    pc_k: float = 2.0

    # DexiNed params
    dexined_weights: str = ""       # path to .pth weights file (empty = default location)
    dexined_threshold: float = 0.5  # sigmoid threshold (0–1)
    dexined_device: str = "cpu"     # "cpu" or "cuda"

    # ---- inspection / segment state ----
    insp_roi: Optional[ROI] = None                              # ROI pre inšpekčný obraz
    min_seg_len: int = 0                                        # min. dĺžka segmentu [px]
    selected_segment_centroid: Optional[tuple] = None           # (cx, cy) ťažiska vybraného segmentu

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
        from src.core.edge_detector import METHODS
        if self.edge_method not in METHODS:
            errors.append(f"Unknown edge method '{self.edge_method}'. Expected one of {METHODS}.")
        if self.roi is not None and not self.roi.is_valid():
            errors.append("ROI is invalid (x1 must be > x0 and y1 must be > y0).")
        if self.ecc_max_iter < 1:
            errors.append("ecc_max_iter must be >= 1.")
        if self.ecc_epsilon <= 0:
            errors.append("ecc_epsilon must be positive.")
        if self.canny_threshold1 < 0:
            errors.append("canny_threshold1 must be >= 0.")
        if self.canny_threshold2 < 0:
            errors.append("canny_threshold2 must be >= 0.")
        if self.canny_blur < 1 or self.canny_blur % 2 == 0:
            errors.append("canny_blur must be a positive odd integer.")
        return errors

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "reference_path": self.reference_path,
            "roi": self.roi.to_dict() if self.roi is not None else None,
            "scale_mm_per_px": self.scale_mm_per_px,
            "algorithm": self.algorithm,
            "ecc": {
                "max_iter": self.ecc_max_iter,
                "epsilon": self.ecc_epsilon,
                "gauss_filt_size": self.ecc_gauss_filt_size,
            },
            "auto_clahe": self.auto_clahe,
            "edge_method": self.edge_method,
            "canny": {
                "threshold1": self.canny_threshold1,
                "threshold2": self.canny_threshold2,
                "blur": self.canny_blur,
            },
            "scharr": {
                "threshold": self.scharr_threshold,
                "blur": self.scharr_blur,
            },
            "log": {
                "sigma": self.log_sigma,
                "threshold": self.log_threshold,
                "blur": self.log_blur,
            },
            "pc": {
                "nscale": self.pc_nscale,
                "min_wavelength": self.pc_min_wavelength,
                "mult": self.pc_mult,
                "k": self.pc_k,
            },
            "dexined": {
                "weights": self.dexined_weights,
                "threshold": self.dexined_threshold,
                "device": self.dexined_device,
            },
            "insp_roi": self.insp_roi.to_dict() if self.insp_roi is not None else None,
            "min_seg_len": self.min_seg_len,
            "selected_segment_centroid": list(self.selected_segment_centroid)
                if self.selected_segment_centroid is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        roi_data = d.get("roi")
        roi = ROI.from_dict(roi_data) if roi_data is not None else None
        insp_roi_data = d.get("insp_roi")
        insp_roi = ROI.from_dict(insp_roi_data) if insp_roi_data is not None else None
        sc = d.get("selected_segment_centroid")
        selected_segment_centroid = (float(sc[0]), float(sc[1])) if sc is not None else None
        ecc      = d.get("ecc", {})
        canny    = d.get("canny", {})
        scharr   = d.get("scharr", {})
        log      = d.get("log", {})
        pc       = d.get("pc", {})
        dexined  = d.get("dexined", {})
        return cls(
            name=d["name"],
            id=int(d.get("id", 0)),
            reference_path=d.get("reference_path", ""),
            roi=roi,
            scale_mm_per_px=float(d.get("scale_mm_per_px", 1.0)),
            algorithm=d.get("algorithm", "ECC"),
            ecc_max_iter=int(ecc.get("max_iter", 2000)),
            ecc_epsilon=float(ecc.get("epsilon", 1e-8)),
            ecc_gauss_filt_size=int(ecc.get("gauss_filt_size", 1)),
            auto_clahe=bool(d.get("auto_clahe", False)),
            edge_method=d.get("edge_method", "Canny"),
            canny_threshold1=int(canny.get("threshold1", 50)),
            canny_threshold2=int(canny.get("threshold2", 150)),
            canny_blur=int(canny.get("blur", 3)),
            scharr_threshold=int(scharr.get("threshold", 30)),
            scharr_blur=int(scharr.get("blur", 3)),
            log_sigma=float(log.get("sigma", 1.5)),
            log_threshold=int(log.get("threshold", 10)),
            log_blur=int(log.get("blur", 3)),
            pc_nscale=int(pc.get("nscale", 4)),
            pc_min_wavelength=int(pc.get("min_wavelength", 6)),
            pc_mult=float(pc.get("mult", 2.1)),
            pc_k=float(pc.get("k", 2.0)),
            dexined_weights=str(dexined.get("weights", "")),
            dexined_threshold=float(dexined.get("threshold", 0.5)),
            dexined_device=str(dexined.get("device", "cpu")),
            insp_roi=insp_roi,
            min_seg_len=int(d.get("min_seg_len", 0)),
            selected_segment_centroid=selected_segment_centroid,
        )
