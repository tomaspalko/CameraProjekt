# PROCESSES.md — Vývojové procesy a stav projektu

> Tento súbor je importovaný v CLAUDE.md a načítaný pri každom resete kontextu.
> Aktualizuj ho po každej dokončenej fáze.

---

## Stav fáz

| Fáza | Názov | Stav | Branch |
|---|---|---|---|
| 1 | Core Engine | ✅ Hotová | phase-2 |
| 2 | Konfigurácia | ✅ Hotová | phase-2 |
| 3 | Batch | ✅ Hotová | phase-2 |
| 4 | GUI | ✅ Hotová | phase-2 |
| 5 | Testy & kalibrácia (POC) | ✅ Hotová | phase-2 |

**Testov:** 124 (108 unit + 16 integration) — všetky zelené.

---

## Spustenie

```bash
# GUI (hlavný spôsob)
python main.py --gui

# CLI — batch bez profilu
python main.py --reference data/reference/ref.png --folder data/batch/ --csv results/out.csv

# CLI — batch s profilom
python main.py --profile job_01 --folder data/batch/ --json results/out.json --verbose
```

---

## Testovanie

```bash
# Všetky testy (pomalé — ~12 min kvôli ECC iteráciám)
pytest

# Len unit testy (rýchle)
pytest tests/unit/

# Watch mód — re-run pri každej zmene .py súboru
python watch_tests.py

# Len unit testy v watch móde (počas vývoja)
python watch_tests.py --fast

# Jednorazový beh, nenulový exit pri zlyhaní (CI)
python watch_tests.py --once
```

**Správny Python interpreter:**
```
C:\Users\tomas.palko\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\pytest.exe
```

---

## Štruktúra zdrojových súborov

```
src/
├── core/
│   ├── preprocessor.py     # CLAHE + blur → uint8 grayscale
│   ├── aligner.py          # ECC (cv2.findTransformECC), routes to POC
│   ├── poc_correlator.py   # Phase-Only Correlation + log-polar rotation
│   ├── roi.py              # ROI dataclass + create_mask()
│   ├── calibration.py      # Calibration(mm_per_px)
│   └── result.py           # AlignResult dataclass
├── config/
│   ├── profile.py          # Profile dataclass + validate()
│   └── config_manager.py   # save/load/list/delete JSON profilov
├── batch/
│   └── batch_processor.py  # process_batch(), process_batch_profile(), BatchResult, BatchStats
└── gui/
    ├── image_viewer.py     # QGraphicsView + ROI rubber-band + overlay (hrany, šípka)
    ├── reference_editor.py # Záložka Konfigurácia (ref obraz, ROI, kalibrácia, profil, test)
    ├── batch_panel.py      # Záložka Batch (QThread worker, tabuľka live-update, štatistiky)
    └── main_window.py      # QMainWindow, QTabWidget, menu, status bar
```

---

## GUI — architektúra

```
MainWindow (QMainWindow)
├── QTabWidget
│   ├── Tab 0: ReferenceEditor   ← ConfigManager inject
│   └── Tab 1: BatchPanel        ← ConfigManager inject
└── QStatusBar

ReferenceEditor
├── ImageViewer (QGraphicsView)  ← rubber-band ROI, overlay
└── QScrollArea
    ├── Group: Referenčný obraz  (načítaj, cesta)
    ├── Group: ROI               (kresliť/zmazať, x0/y0/x1/y1 spinboxy)
    ├── Group: Kalibrácia        (mm/px)
    ├── Group: Algoritmus        (ECC/POC, max_iter, epsilon)
    ├── Group: Profil            (uložiť/načítať/zmazať)
    └── Group: Test zarovnania   (načítaj test, spustiť, výsledky + overlay)

BatchPanel
├── QGroupBox: Nastavenia        (priečinok, profil, export CSV/JSON, [Spustiť], progress)
├── QGroupBox: Výsledky          (QTableWidget: Súbor|Stav|dx_px|dy_px|dx_mm|dy_mm|Uhol|Conf)
└── QGroupBox: Štatistiky        (count OK/total, mean±std min/max pre každú metriku)
```

**Batch worker:** `_BatchWorker(QThread)` — emituje `row_ready(int, dict)` per-image signal,
`finished(BatchResult)` na konci. UI zostáva responzívne.

---

## Kľúčové implementačné detaily

### ECC aligner (`src/core/aligner.py`)
- `cv2.findTransformECC` s `cv2.MOTION_EUCLIDEAN`
- Inicializácia z identity matice (dostatočné pre ~1° rotáciu)
- **Oprava presnosti uhla:** `asin(warp[1,0])` namiesto `acos(warp[0,0])` — lepšia presnosť pri malých uhloch
- Korekcia centra rotácie: odpočítanie offsetu `getRotationMatrix2D`
- Pyramída = 1 (žiadna multi-scale pre malé posuny)
- Parametrize: `max_iter=2000`, `epsilon=1e-8` (defaults v Profile)

### ROI (`src/core/roi.py`)
- Koordináty: `(x0, y0, x1, y1)` — top-left, bottom-right, **exkluzívne** (numpy konvencia)
- Maska: `uint8` (255 = použi, 0 = ignoruj), auto-clamp na hranice

### Profile (`src/config/profile.py`)
- Uložené ako JSON v `config/profiles/<name>.json`
- Validácia: name non-empty, scale > 0, algorithm ∈ {ECC, POC}, ecc_max_iter ≥ 1, epsilon > 0

### Tolerancie testov (`tests/constants.py`)
```python
TRANSLATION_TOL_PX = 0.10   # max chyba dx/dy [px]
ROTATION_TOL_DEG   = 0.05   # max chyba uhol [°]
MM_CONVERSION_TOL  = 1e-5   # relatívna chyba px→mm
MIN_CONFIDENCE     = 0.60   # minimálna ECC spoľahlivosť
```

---

### POC aligner (`src/core/poc_correlator.py`)
- Dvojkrokový pipeline: log-polar FFT → rotácia, potom POC → transl
- **Rotácia:** FFT magnitúdy oboch obrazov → log-polar warp (`cv2.warpPolar`) → POC → uhol
- **Preklad:** derotovaný obraz + POC s Hann oknom → sub-pixel parabolic fit
- Tolerancie v `tests/unit/test_poc.py`: 0.05 px, 0.05°

---

## Git workflow

```bash
# Vetva pre vývoj
git checkout -b phase-5

# Po dokončení fázy
git add src/ tests/ main.py PROCESSES.md
git commit -m "faza X: ..."
```
