"""tests/constants.py — single source of truth for test tolerances.

Import from here in conftest.py and in any test file that needs raw constants
(e.g. test_aligner.py parametrised cases).
"""

TRANSLATION_TOL_PX = 0.10   # max allowed error in dx/dy [px]
ROTATION_TOL_DEG   = 0.05   # max allowed error in angle [°]
MM_CONVERSION_TOL  = 1e-5   # relative error for px→mm conversion
MIN_CONFIDENCE     = 0.60   # minimum acceptable ECC confidence
