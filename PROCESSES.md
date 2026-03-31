# PROCESSES.md — Vývojové procesy a stav projektu

> Tento súbor je importovaný v CLAUDE.md a načítaný pri každom resete kontextu.
> Aktualizuj ho po každej dokončenej fáze.

---

## Stav fáz

| Fáza | Názov | Stav | Branch |
|---|---|---|---|
| 1 | Core Engine | ✅ Hotová | phase-2 |
| 2 | Konfigurácia | ✅ Hotová | phase-2 |
| 3 | GUI | ✅ Hotová | phase-2 |
| 4 | Testy & kalibrácia (POC) | ✅ Hotová | phase-2 |
| 5 | Vylepšenia (web research) | ✅ Hotová | phase-2 |
| 6 | GUI redesign — záložky + startup | ✅ Hotová | phase-2 |

**Testov:** 102 unit + 5 integration = 107 — všetky zelené.

---

## Spustenie

```bash
# GUI (jediný spôsob spustenia)
python main.py
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
│   ├── preprocessor.py        # CLAHE + blur → uint8 grayscale
│   ├── aligner.py             # ECC (cv2.findTransformECC), routes to POC
│   ├── poc_correlator.py      # Phase-Only Correlation + log-polar rotation
│   ├── roi.py                 # ROI dataclass + create_mask()
│   ├── calibration.py         # Calibration(mm_per_px)
│   └── result.py              # AlignResult dataclass
├── config/
│   ├── profile.py             # Profile dataclass + id + validate()
│   └── config_manager.py      # save/load/list/delete + auto-increment ID
└── gui/
    ├── image_viewer.py        # QGraphicsView + ROI/CALIBRATION modes + overlay
    ├── startup_dialog.py      # Startup — výber/vytvorenie profilu
    ├── profile_editor_tab.py  # Tab 1: konfigurácia profilu (1 viewer + slider params)
    ├── inspection_tab.py      # Tab 2: inšpekcia (2 viewery + výsledky)
    └── main_window.py         # QMainWindow + QTabWidget + startup dialog
```

---

## GUI — architektúra

```
Startup (StartupDialog)
  → new / edit / inspect

MainWindow (QMainWindow)
├── QTabWidget
│   ├── Tab 1: ProfileEditorTab     ← 1 viewer, slider params, kalibrácia, uloženie
│   └── Tab 2: InspectionTab        ← ref viewer + insp viewer, výsledky
└── QStatusBar

ProfileEditorTab
├── ImageViewer (referenčný)        ← ROI, CALIBRATION, segment select/remove
└── QScrollArea
    ├── Group: Referenčný obraz     (načítaj/zmazať)
    ├── Group: ROI                  (kresliť/zmazať, spinboxy, toggle hrany)
    ├── Group: Detekcia hrán        (metóda, sliders+spinboxy, min dĺžka segmentu)
    ├── Group: Segmenty             (odstrániť klik/oblasť, undo, reset, vybrať)
    ├── Group: Kalibrácia           (2-bodová, mm/px)
    └── Group: Algoritmus & Profil  (ECC/POC params, ID label, názov, uložiť)

InspectionTab
├── Profile combo + load btn
├── ImageViewer (referenčný — read-only, segmenty z profilu)
├── ImageViewer (inšpekčný — ROI, overlay výsledku)
└── QScrollArea
    ├── Group: Inšpekčný obraz      (načítaj/zmazať)
    ├── Group: Inšpekčné ROI        (kresliť/zmazať, spinboxy, uložiť do profilu)
    ├── Group: Parametre zarovnania (ECC/POC, max_iter slider, epsilon, uložiť)
    ├── Toggle: Zobraziť hrany v ROI
    ├── Button: Spustiť vyhľadávanie
    └── Group: Výsledky             (dx/dy px+mm, uhol, confidence, NCC, čas, ťažisko)
```

### Profile (src/config/profile.py)
- Pole `id: int = 0` — auto-increment pri prvom uložení (0 = nepridelené)
- Uložené ako JSON v `config/profiles/<name>.json`
- `ConfigManager._next_id()` nájde max ID zo súborov → max+1
- `list_profiles_full()` vracia `[{id, name}, ...]` zoradené podľa ID

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

### Fáza 5 — Vylepšenia (na základe web research 2024–2025)

Empiricky testované na syntetických dátach; výsledky:

| Zmena | Výsledok |
|---|---|
| INTER_CUBIC pre derotáciu (P1-C) | Mierne horšie — INTER_LINEAR zostáva |
| Vyššie log-polar rozlíšenie (P1-B) | Spôsobuje boundary issues — zamietnuté |
| Gaussian log-domain peak fit (P1-A) | Horšie pre Hann POC — zamietnuté |
| ECC two-pass gaussFiltSize=1 (P1-E) | Diverguje pre kombinovaný pohyb — `two_pass=False` default |
| Adaptívny CLAHE `auto_clahe` (P2-B) | ✅ Pridané do `preprocess()` a `Profile` |
| NCC confidence skóre (P2-C) | ✅ Pridané ako `ncc_score` do všetkých výsledkov |

**Záver:** ECC presnosť je systematicky limitovaná na ~0.059 px pre testovacie dáta (256×256 syntetické). Nie je to kódový problém — ďalšie zlepšenie vyžaduje reálne obrazy alebo DL prístup.

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
