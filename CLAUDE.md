# CLAUDE.md — Weld Inspection Vision System

@PROCESSES.md

## Specifications

| Parameter | Value |
|---|---|
| Camera | Static, < 2 MP (1280×1024) |
| Object | Metal weld joint |
| Accuracy | Sub-pixel (< 0.05 px, target 1/100 px) |
| Motion range | Small translation, rotation ~1° |
| Lighting | Controlled, stable |
| Output | dx, dy, angle (°), confidence |
| Mode | Batch (stored images) |
| UI | Desktop GUI (PyQt6) |

---

## Algorithm Pipeline

Rotation ~1° allows ECC to converge directly from identity matrix — the coarse AKAZE/ORB step is not needed. This simplifies the pipeline and enables higher accuracy.

```
Input → Preprocessing (CLAHE + blur) → ROI mask
      → ECC Euclidean (identity init, 1000–5000 iter, ε=1e-8)
      → Scale: px → mm
      → Output: {dx_px, dy_px, angle_deg, dx_mm, dy_mm, confidence}
```

**Fallback algorithm:** Phase-Only Correlation (POC) — more accurate than Fourier-Mellin for small displacements, capable of 1/100 px accuracy.

---

## Project Structure

```
weld_vision/
├── config/
│   ├── default_config.yaml
│   └── profiles/*.json         # configuration profiles
├── data/
│   ├── reference/              # reference images
│   └── batch/                  # images to process
├── results/                    # CSV / JSON outputs
├── src/
│   ├── core/
│   │   ├── preprocessor.py     # CLAHE, blur, grayscale
│   │   ├── aligner.py          # ECC direct (no coarse step)
│   │   ├── poc_correlator.py   # fallback: Phase-Only Correlation
│   │   ├── calibration.py      # px → mm
│   │   └── result.py           # Result dataclass + export
│   ├── config/
│   │   └── config_manager.py   # YAML/JSON profiles
│   ├── batch/
│   │   └── batch_processor.py
│   └── gui/
│       ├── main_window.py
│       ├── reference_editor.py # ROI + scale + preview
│       ├── batch_panel.py
│       └── image_viewer.py     # overlay visualisation
├── tests/
│   ├── conftest.py             # shared fixtures, single source of tolerances
│   ├── unit/
│   │   ├── test_preprocessor.py
│   │   └── test_aligner.py
│   ├── integration/
│   │   └── test_pipeline.py    # end-to-end + batch + profile round-trip
│   └── synthetic/
│       └── generator.py        # synthetic test pair generator
├── pytest.ini
├── watch_tests.py              # file watcher + retry runner
└── main.py
```

---

## Development Phases

| Phase | Contents | Success Criterion |
|---|---|---|
| **1 — Core Engine** | preprocessor, aligner (ECC only), calibration, result + unit tests | Pipeline returns result, accuracy < 0.1 px on synthetic data |
| **2 — Configuration** | config_manager, ROI editor, profiles, algorithm switching | Profile save/load works, live preview functional |
| **3 — Batch** | batch_processor, CSV/JSON export, error handling, statistics | 20-image batch runs without crash, correct CSV output |
| **4 — GUI** | PyQt6: main_window, reference_editor, batch_panel, overlay | Full workflow operable via GUI |
| **5 — Tests & calibration** | accuracy benchmarks, ECC tuning, documentation | RMS error < 0.05 px, angle < 0.02° |

**Implementation order within each phase:** `synthetic/generator.py` → `src/core/` → unit tests → integration → GUI

---

## Testing Strategy

### Philosophy
Tests run after every change. The watch runner retries until all tests pass — never commit broken code.

### Tolerances (single source of truth: `tests/conftest.py`)

| Constant | Value |
|---|---|
| `TRANSLATION_TOL_PX` | 0.10 px |
| `ROTATION_TOL_DEG` | 0.05° |
| `MIN_CONFIDENCE` | 0.60 |
| `MM_CONVERSION_TOL` | 1e-5 |

Change these in `conftest.py` only — all tests pick them up automatically.

### Test layers

**Layer 1 — Unit tests** (`tests/unit/`, < 1s each): shape, dtype, value range, edge cases, no crashes.

**Layer 2 — Accuracy tests** (`test_aligner.py`): 8 parametrised cases with known ground truth (pure translation, pure rotation, combined, sub-pixel only, negative values). Also an RMS test that aggregates error across all cases.

**Layer 3 — Integration tests** (`tests/integration/`): full pipeline with MM conversion, batch of 20 images, corrupt image handling, CSV/JSON export, profile save/load round-trip.

### Test cases for aligner

| Case | dx | dy | angle |
|---|---|---|---|
| Pure X translation | 3.0 | 0.0 | 0.0° |
| Pure Y translation | 0.0 | -4.0 | 0.0° |
| Pure rotation + | 0.0 | 0.0 | 1.0° |
| Pure rotation − | 0.0 | 0.0 | -1.0° |
| Combined | 5.0 | -3.0 | 0.8° |
| Combined negative | -2.5 | 1.5 | -0.5° |
| Sub-pixel only | 0.3 | 0.1 | 0.1° |
| Larger displacement | 10.0 | 8.0 | 1.5° |

### Watch runner (`watch_tests.py`)

```bash
python watch_tests.py           # watch mode: re-run on every .py save
python watch_tests.py --fast    # unit tests only (during active development)
python watch_tests.py --until-pass   # retry on change until green, then exit
python watch_tests.py --once    # single run, non-zero exit on failure (CI)
```

**Recommended workflow:** keep `watch_tests.py --fast` running in a side terminal while coding. Switch to full `watch_tests.py` before committing.

---

## Key Implementation Notes

- **ECC initialise from identity** — for ~1° rotation this is sufficient and avoids the complexity of a coarse step
- **ECC known bug for small angles** — OpenCV computes angle update via `acos` which has rounding error near 0°; patch: use `asin(mapPtr[3])` instead for better precision at small rotations
- **Pyramid levels = 1** — small motion does not benefit from multi-scale; single level gives better accuracy
- **max_iterations = 1000–5000, epsilon = 1e-8** — aggressive settings are feasible since batch speed is not critical
- **Confidence score** = return value of `findTransformECC`; OK threshold: > 0.7
- dx positive = right, dy positive = down (OpenCV convention)

---

## Dependencies

```
opencv-python>=4.9.0  numpy>=1.26.0  PyQt6>=6.6.0
PyYAML>=6.0  pandas>=2.2.0  scipy>=1.13.0  pytest>=8.0.0  watchdog>=4.0.0
```

---

## GUI Layout

UI design mockup: `UGILayout/UGIlayout.pdf`

### Job Configuration
- Load and display the reference image
- Draw / edit the ROI (region of interest) on the reference image
- Set the scale (mm/px calibration)
- Select the alignment algorithm (ECC / POC fallback)
- Save / load a named configuration profile

### Testing Mode
- Load an inspection image
- Run alignment against the reference using the active profile
- Display result: dx, dy, angle (°), confidence
- Visual overlay on the inspection image:
  - Highlight detected edges / object contour (e.g. Canny edges drawn in a contrasting colour)
  - Show displacement vector or bounding-box offset relative to the reference position
