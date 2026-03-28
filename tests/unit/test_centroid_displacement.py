"""tests/unit/test_centroid_displacement.py — presnosť merania polohy ťažiska segmentu.

Testuje dvoj-krokový pipeline z inspection_panel.py:
  1. Projekčný vzorec ťažiska (čistá matematika, bez GUI)
  2. Integrácia: syntetický pár → align() → projekcia → overenie polohy

Vzorec (inspection_panel.py:1022-1025, getRotationMatrix2D konvencia):
    cx_new = cos(θ)*(cx_seg - cx_img) + sin(θ)*(cy_seg - cy_img) + cx_img + dx_px
    cy_new = -sin(θ)*(cx_seg - cx_img) + cos(θ)*(cy_seg - cy_img) + cy_img + dy_px
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from src.core.aligner import align
from tests.constants import TRANSLATION_TOL_PX, ROTATION_TOL_DEG
from tests.synthetic.generator import make_pair


# ── Projekčný vzorec (replika z inspection_panel.py:1022-1025) ───────────────

def _project_centroid(
    cx_seg: float, cy_seg: float,
    cx_img: float, cy_img: float,
    angle_deg: float,
    dx_px: float, dy_px: float,
) -> tuple[float, float]:
    """Premietne referenčné ťažisko do súradníc inšpekčného obrazu.

    Zodpovedá vzorcu v inspection_panel.py:1022-1025
    (getRotationMatrix2D konvencia: kladný uhol = CCW).
    """
    theta = math.radians(angle_deg)
    cx_new = (
        math.cos(theta) * (cx_seg - cx_img)
        + math.sin(theta) * (cy_seg - cy_img)
        + cx_img + dx_px
    )
    cy_new = (
        -math.sin(theta) * (cx_seg - cx_img)
        + math.cos(theta) * (cy_seg - cy_img)
        + cy_img + dy_px
    )
    return cx_new, cy_new


# ── Tolerancie ────────────────────────────────────────────────────────────────

# Kombinovaná tolerancia polohy ťažiska [px]:
#   príspevok prekladu: TRANSLATION_TOL_PX = 0.10 px
#   príspevok uhla pri ťažisku 40 px od stredu: 0.05° × π/180 × 40 ≈ 0.035 px
#   spolu: ~0.14 px  →  tolerancia 0.25 px je konzervatívna
CENTROID_POS_TOL = 0.25  # [px]


# ── Časť 1: Unit testy projekčného vzorca ────────────────────────────────────

def test_no_motion_centroid_stays():
    """Nulový pohyb — ťažisko ostáva na mieste."""
    cx_seg, cy_seg = 80.0, 100.0
    cx_img, cy_img = 128.0, 128.0
    cx_new, cy_new = _project_centroid(cx_seg, cy_seg, cx_img, cy_img, 0.0, 0.0, 0.0)
    assert abs(cx_new - cx_seg) < 1e-9
    assert abs(cy_new - cy_seg) < 1e-9


def test_pure_translation_centroid_shifts_exactly():
    """Nulová rotácia — ťažisko sa posunie presne o (dx, dy)."""
    cx_seg, cy_seg = 80.0, 100.0
    cx_img, cy_img = 128.0, 128.0
    dx, dy = 5.3, -2.7
    cx_new, cy_new = _project_centroid(cx_seg, cy_seg, cx_img, cy_img, 0.0, dx, dy)
    assert abs(cx_new - (cx_seg + dx)) < 1e-9
    assert abs(cy_new - (cy_seg + dy)) < 1e-9


def test_centroid_at_image_center_pure_rotation_no_displacement():
    """Ťažisko na strede obrazu + čistá rotácia → nulové posunutie."""
    cx_img, cy_img = 128.0, 128.0
    cx_new, cy_new = _project_centroid(cx_img, cy_img, cx_img, cy_img, 1.0, 0.0, 0.0)
    assert abs(cx_new - cx_img) < 1e-9
    assert abs(cy_new - cy_img) < 1e-9


def test_formula_matches_warp_affine():
    """Vzorec musí byť konzistentný s cv2.getRotationMatrix2D + cv2.transform."""
    cx_seg, cy_seg = 70.0, 90.0
    img_w, img_h = 256, 256
    cx_img, cy_img = img_w / 2.0, img_h / 2.0
    angle_deg, dx, dy = 1.0, 3.5, -2.0

    M = cv2.getRotationMatrix2D((cx_img, cy_img), angle_deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    pt = np.array([[[cx_seg, cy_seg]]], dtype=np.float64)
    warped = cv2.transform(pt, M)[0, 0]

    cx_new, cy_new = _project_centroid(cx_seg, cy_seg, cx_img, cy_img, angle_deg, dx, dy)
    assert abs(cx_new - warped[0]) < 1e-4, f"cx: {cx_new:.6f} vs {warped[0]:.6f}"
    assert abs(cy_new - warped[1]) < 1e-4, f"cy: {cy_new:.6f} vs {warped[1]:.6f}"


def test_formula_matches_warp_affine_multiple_angles():
    """Konzistencia vzorca s OpenCV pre viacero uhlov a polôh ťažiska."""
    cx_img, cy_img = 128.0, 128.0
    cases = [
        (50.0, 70.0, 0.5, 2.0, -1.5),
        (150.0, 90.0, -1.0, -3.0, 4.0),
        (128.0, 60.0, 1.5, 0.0, 0.0),
        (30.0, 200.0, -0.3, 1.0, 1.0),
    ]
    for cx_seg, cy_seg, angle_deg, dx, dy in cases:
        M = cv2.getRotationMatrix2D((cx_img, cy_img), angle_deg, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        pt = np.array([[[cx_seg, cy_seg]]], dtype=np.float64)
        warped = cv2.transform(pt, M)[0, 0]

        cx_new, cy_new = _project_centroid(cx_seg, cy_seg, cx_img, cy_img, angle_deg, dx, dy)
        assert abs(cx_new - warped[0]) < 1e-4, (
            f"angle={angle_deg}° cx: {cx_new:.6f} vs {warped[0]:.6f}"
        )
        assert abs(cy_new - warped[1]) < 1e-4, (
            f"angle={angle_deg}° cy: {cy_new:.6f} vs {warped[1]:.6f}"
        )


# ── Časť 2: Integračné testy (align + projekcia) ─────────────────────────────

# (dx, dy, angle_deg, centroid_offset_x, centroid_offset_y)
# offset je relatívny voči stredu obrazu (128, 128)
CENTROID_CASES = [
    (  3.0,  0.0,  0.0,   0.0,   0.0),  # čistý posun X, ťažisko v strede
    (  0.0, -4.0,  0.0,  20.0, -10.0),  # čistý posun Y, ťažisko mimo stredu
    (  0.0,  0.0,  1.0,  30.0,  20.0),  # čistá rotácia +, ťažisko mimo stredu
    (  0.0,  0.0, -1.0, -30.0,  20.0),  # čistá rotácia −
    (  5.0, -3.0,  0.8,  40.0, -30.0),  # kombinovaný pohyb
    ( -2.5,  1.5, -0.5, -20.0,  15.0),  # kombinovaný negatívny
]

_CASE_IDS = [
    "pure-X",
    "pure-Y",
    "rot+",
    "rot-",
    "combined",
    "combined-neg",
]


@pytest.mark.parametrize("dx,dy,angle,cx_off,cy_off", CENTROID_CASES, ids=_CASE_IDS)
def test_centroid_displacement_accuracy(
    dx: float, dy: float, angle: float, cx_off: float, cy_off: float
) -> None:
    """Poloha ťažiska po zarovnaní musí zodpovedať ground-truth v medziach CENTROID_POS_TOL.

    Postup:
      1. make_pair() → syntetický pár so známou transformáciou
      2. align() → nameraný uhol + posun
      3. _project_centroid() → nameraná poloha ťažiska
      4. _project_centroid() s GT hodnotami → očakávaná poloha ťažiska
      5. Chyba = rozdiel polôh < CENTROID_POS_TOL
    """
    ref, img, gt = make_pair(dx, dy, angle)
    h, w = ref.shape[:2]
    cx_img, cy_img = w / 2.0, h / 2.0
    cx_seg = cx_img + cx_off
    cy_seg = cy_img + cy_off

    # Očakávaná poloha (GT)
    cx_exp, cy_exp = _project_centroid(
        cx_seg, cy_seg, cx_img, cy_img,
        gt["angle_deg"], gt["dx"], gt["dy"],
    )

    # Nameraná poloha
    result = align(ref, img)
    cx_meas, cy_meas = _project_centroid(
        cx_seg, cy_seg, cx_img, cy_img,
        result["angle_deg"], result["dx_px"], result["dy_px"],
    )

    err_cx = abs(cx_meas - cx_exp)
    err_cy = abs(cy_meas - cy_exp)

    assert err_cx < CENTROID_POS_TOL, (
        f"cx chyba {err_cx:.4f} px > {CENTROID_POS_TOL} px "
        f"(angle={angle}°, offset=({cx_off},{cy_off}))\n"
        f"  GT: angle={gt['angle_deg']}° dx={gt['dx']} dy={gt['dy']}\n"
        f"  Nameraný: angle={result['angle_deg']:.4f}° "
        f"dx={result['dx_px']:.4f} dy={result['dy_px']:.4f}"
    )
    assert err_cy < CENTROID_POS_TOL, (
        f"cy chyba {err_cy:.4f} px > {CENTROID_POS_TOL} px "
        f"(angle={angle}°, offset=({cx_off},{cy_off}))"
    )


def test_centroid_displacement_rms() -> None:
    """RMS chyba polohy ťažiska cez všetky prípady musí byť < CENTROID_POS_TOL."""
    cx_sq_errors: list[float] = []
    cy_sq_errors: list[float] = []

    for dx, dy, angle, cx_off, cy_off in CENTROID_CASES:
        ref, img, gt = make_pair(dx, dy, angle)
        h, w = ref.shape[:2]
        cx_img, cy_img = w / 2.0, h / 2.0
        cx_seg = cx_img + cx_off
        cy_seg = cy_img + cy_off

        cx_exp, cy_exp = _project_centroid(
            cx_seg, cy_seg, cx_img, cy_img,
            gt["angle_deg"], gt["dx"], gt["dy"],
        )
        result = align(ref, img)
        cx_meas, cy_meas = _project_centroid(
            cx_seg, cy_seg, cx_img, cy_img,
            result["angle_deg"], result["dx_px"], result["dy_px"],
        )

        cx_sq_errors.append((cx_meas - cx_exp) ** 2)
        cy_sq_errors.append((cy_meas - cy_exp) ** 2)

    rms_cx = math.sqrt(sum(cx_sq_errors) / len(cx_sq_errors))
    rms_cy = math.sqrt(sum(cy_sq_errors) / len(cy_sq_errors))

    assert rms_cx < CENTROID_POS_TOL, f"RMS cx chyba {rms_cx:.4f} px"
    assert rms_cy < CENTROID_POS_TOL, f"RMS cy chyba {rms_cy:.4f} px"
