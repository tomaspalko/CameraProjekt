"""InspectionPanel — hlavná záložka: referenčný obraz vľavo, inšpekčný vpravo, s detekciou hrán."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtWidgets import (
    QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QDoubleSpinBox,
    QSpinBox, QComboBox, QScrollArea, QFileDialog, QMessageBox,
    QCheckBox, QSizePolicy,
)

from src.core.roi import ROI
from src.core.preprocessor import preprocess
from src.core.aligner import align
from src.core.calibration import Calibration
from src.core.result import AlignResult
from src.core.edge_detector import detect_edges, METHODS as EDGE_METHODS
from src.config.profile import Profile
from src.config.config_manager import ConfigManager
from src.gui.image_viewer import ImageViewer


class _EdgeWorker(QThread):
    """Background thread pre pomalé metódy detekcie hrán (napr. DexiNed)."""

    result_ready = pyqtSignal(object)   # np.ndarray
    error        = pyqtSignal(str)

    def __init__(self, gray: np.ndarray, roi, method: str, params: dict) -> None:
        super().__init__()
        self._gray   = gray
        self._roi    = roi
        self._method = method
        self._params = params

    def run(self) -> None:
        try:
            if self._roi is not None:
                h_img, w_img = self._gray.shape
                x0c = max(0, self._roi.x0)
                y0c = max(0, self._roi.y0)
                x1c = min(w_img, self._roi.x1)
                y1c = min(h_img, self._roi.y1)
                if x1c <= x0c or y1c <= y0c:
                    return
                edges_full = np.zeros(self._gray.shape, dtype=np.uint8)
                edges_full[y0c:y1c, x0c:x1c] = detect_edges(
                    self._gray[y0c:y1c, x0c:x1c], self._method, **self._params
                )
            else:
                edges_full = detect_edges(self._gray, self._method, **self._params)
            self.result_ready.emit(edges_full)
        except Exception as exc:
            self.error.emit(str(exc))


class InspectionPanel(QWidget):
    """Panel s dvoma zobrazovačmi (referencia vľavo, inšpekcia vpravo) a ovládacím panelom.

    Signals:
        profile_changed(Profile): emitovaný po každej zmene konfigurácie.
    """

    profile_changed = pyqtSignal(object)

    def __init__(self, config_mgr: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config_mgr = config_mgr
        self._ref_image: np.ndarray | None = None
        self._insp_image: np.ndarray | None = None
        self._roi: ROI | None = None
        self._insp_roi: ROI | None = None              # ROI pre inšpekčný obraz
        self._ref_edges: np.ndarray | None = None  # cache edge mask for projection
        self._insp_edges: np.ndarray | None = None     # edge maska inšpekčného obrazu
        self._edge_worker: _EdgeWorker | None = None  # background DexiNed thread
        self._segment_labels: np.ndarray | None = None   # int32 label mapa (cv2.connectedComponents)
        self._removed_labels: set[int] = set()
        self._erased_mask: np.ndarray | None = None      # uint8 pixel-level maska; 0 = vymazaný pixel
        self._undo_stack: list[tuple[frozenset[int], np.ndarray | None]] = []
        self._selected_label: int | None = None          # vybraný segment pre template matching

        self._build_ui()
        self._connect_signals()
        self._refresh_profile_list()

    # ------------------------------------------------------------------
    # Verejné metódy
    # ------------------------------------------------------------------

    def get_profile(self) -> Profile:
        """Vráti aktuálny profil z formulára."""
        try:
            epsilon = float(self._epsilon_edit.text())
        except ValueError:
            epsilon = 1e-8
        return Profile(
            name=self._name_edit.text().strip() or "unnamed",
            reference_path=self._ref_path_label.text() if self._ref_path_label.text() != "—" else "",
            roi=self._roi,
            scale_mm_per_px=self._scale_spin.value(),
            algorithm=self._algo_combo.currentText(),
            ecc_max_iter=self._iter_spin.value(),
            ecc_epsilon=epsilon,
            auto_clahe=self._auto_clahe_check.isChecked(),
            ecc_gauss_filt_size=self._gauss_filt_combo.currentData() or 1,
            edge_method=self._edge_method_combo.currentText(),
            canny_threshold1=self._canny_t1_spin.value(),
            canny_threshold2=self._canny_t2_spin.value(),
            canny_blur=self._canny_blur_combo.currentData() or 3,
            scharr_threshold=self._scharr_thresh_spin.value(),
            scharr_blur=self._scharr_blur_combo.currentData() or 3,
            log_sigma=self._log_sigma_spin.value(),
            log_threshold=self._log_thresh_spin.value(),
            log_blur=self._log_blur_combo.currentData() or 1,
            pc_nscale=self._pc_nscale_spin.value(),
            pc_min_wavelength=self._pc_minwave_spin.value(),
            pc_mult=self._pc_mult_spin.value(),
            pc_k=self._pc_k_spin.value(),
            dexined_weights=self._dexined_weights_edit.text().strip(),
            dexined_threshold=self._dexined_thresh_spin.value(),
            dexined_device=self._dexined_device_combo.currentText(),
        )

    def set_profile(self, profile: Profile) -> None:
        """Naplní formulár hodnotami z profilu."""
        self._name_edit.setText(profile.name)
        ref_path = profile.reference_path
        if ref_path:
            self._ref_path_label.setText(ref_path)
        self._scale_spin.setValue(profile.scale_mm_per_px)

        idx = self._algo_combo.findText(profile.algorithm)
        if idx >= 0:
            self._algo_combo.setCurrentIndex(idx)

        self._iter_spin.setValue(profile.ecc_max_iter)
        self._epsilon_edit.setText(str(profile.ecc_epsilon))
        gfs_idx = self._gauss_filt_combo.findData(profile.ecc_gauss_filt_size)
        if gfs_idx >= 0:
            self._gauss_filt_combo.setCurrentIndex(gfs_idx)
        self._auto_clahe_check.setChecked(profile.auto_clahe)
        self._canny_t1_spin.setValue(profile.canny_threshold1)
        self._canny_t2_spin.setValue(profile.canny_threshold2)

        blur_idx = self._canny_blur_combo.findData(profile.canny_blur)
        if blur_idx >= 0:
            self._canny_blur_combo.setCurrentIndex(blur_idx)

        # Edge method + params
        idx = self._edge_method_combo.findText(profile.edge_method)
        if idx >= 0:
            self._edge_method_combo.blockSignals(True)
            self._edge_method_combo.setCurrentIndex(idx)
            self._edge_params_stack.setCurrentIndex(idx)
            self._edge_method_combo.blockSignals(False)

        self._canny_t1_spin.setValue(profile.canny_threshold1)
        self._canny_t2_spin.setValue(profile.canny_threshold2)
        blur_idx = self._canny_blur_combo.findData(profile.canny_blur)
        if blur_idx >= 0:
            self._canny_blur_combo.setCurrentIndex(blur_idx)

        self._scharr_thresh_spin.setValue(profile.scharr_threshold)
        s_blur_idx = self._scharr_blur_combo.findData(profile.scharr_blur)
        if s_blur_idx >= 0:
            self._scharr_blur_combo.setCurrentIndex(s_blur_idx)

        self._log_sigma_spin.setValue(profile.log_sigma)
        self._log_thresh_spin.setValue(profile.log_threshold)
        l_blur_idx = self._log_blur_combo.findData(profile.log_blur)
        if l_blur_idx >= 0:
            self._log_blur_combo.setCurrentIndex(l_blur_idx)

        self._pc_nscale_spin.setValue(profile.pc_nscale)
        self._pc_minwave_spin.setValue(profile.pc_min_wavelength)
        self._pc_mult_spin.setValue(profile.pc_mult)
        self._pc_k_spin.setValue(profile.pc_k)

        self._dexined_weights_edit.setText(profile.dexined_weights)
        self._dexined_thresh_spin.setValue(profile.dexined_threshold)
        dev_idx = self._dexined_device_combo.findText(profile.dexined_device)
        if dev_idx >= 0:
            self._dexined_device_combo.setCurrentIndex(dev_idx)

        self._roi = profile.roi
        if ref_path and Path(ref_path).exists():
            self._load_ref_from_path(ref_path)
        elif self._ref_image is not None:
            if self._roi is not None:
                self._ref_viewer.draw_roi(self._roi)
                self._update_roi_spinboxes()
            self._update_ref_edges()

    # ------------------------------------------------------------------
    # Budovanie UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Ľavý viewer — referenčný obraz
        self._ref_viewer = ImageViewer()
        self._ref_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        ref_container = QWidget()
        ref_layout = QVBoxLayout(ref_container)
        ref_layout.setContentsMargins(0, 0, 0, 0)
        ref_label = QLabel("Referenčný obraz")
        ref_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ref_label.setStyleSheet("font-weight: bold; padding: 2px;")
        ref_layout.addWidget(ref_label)
        ref_layout.addWidget(self._ref_viewer)
        splitter.addWidget(ref_container)

        # Pravý viewer — inšpekčný obraz
        self._insp_viewer = ImageViewer()
        self._insp_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        insp_container = QWidget()
        insp_layout = QVBoxLayout(insp_container)
        insp_layout.setContentsMargins(0, 0, 0, 0)
        insp_label = QLabel("Inšpekčný obraz")
        insp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        insp_label.setStyleSheet("font-weight: bold; padding: 2px;")
        insp_layout.addWidget(insp_label)
        insp_layout.addWidget(self._insp_viewer)
        splitter.addWidget(insp_container)

        # Pravý panel — ovládanie (scroll area)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(300)
        right_scroll.setMaximumWidth(380)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)

        right_layout.addWidget(self._build_ref_group())
        right_layout.addWidget(self._build_roi_group())
        right_layout.addWidget(self._build_edge_group())
        right_layout.addWidget(self._build_insp_group())
        right_layout.addWidget(self._build_insp_roi_group())
        right_layout.addWidget(self._build_algo_group())
        right_layout.addWidget(self._build_calib_group())
        right_layout.addWidget(self._build_profile_group())

        self._run_btn = QPushButton("Spustiť zarovnanie")
        self._run_btn.setMinimumHeight(36)
        right_layout.addWidget(self._run_btn)

        right_layout.addWidget(self._build_result_group())
        right_layout.addStretch()

        right_scroll.setWidget(right_widget)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(splitter)

    def _build_ref_group(self) -> QGroupBox:
        grp = QGroupBox("Referenčný obraz")
        layout = QVBoxLayout(grp)
        btn = QPushButton("Načítaj obraz…")
        btn.clicked.connect(self._on_load_reference)
        self._ref_path_label = QLabel("—")
        self._ref_path_label.setWordWrap(True)
        layout.addWidget(btn)
        layout.addWidget(self._ref_path_label)
        return grp

    def _build_roi_group(self) -> QGroupBox:
        grp = QGroupBox("ROI (Region of Interest)")
        layout = QVBoxLayout(grp)

        btn_row = QHBoxLayout()
        self._draw_roi_btn = QPushButton("Kresliť ROI")
        self._draw_roi_btn.setCheckable(True)
        self._clear_roi_btn = QPushButton("Zmazať ROI")
        btn_row.addWidget(self._draw_roi_btn)
        btn_row.addWidget(self._clear_roi_btn)
        layout.addLayout(btn_row)

        form = QFormLayout()
        self._roi_x0 = QSpinBox(); self._roi_x0.setRange(0, 9999)
        self._roi_y0 = QSpinBox(); self._roi_y0.setRange(0, 9999)
        self._roi_x1 = QSpinBox(); self._roi_x1.setRange(0, 9999)
        self._roi_y1 = QSpinBox(); self._roi_y1.setRange(0, 9999)
        form.addRow("x0:", self._roi_x0)
        form.addRow("y0:", self._roi_y0)
        form.addRow("x1:", self._roi_x1)
        form.addRow("y1:", self._roi_y1)
        layout.addLayout(form)
        return grp

    def _build_edge_group(self) -> QGroupBox:
        from PyQt6.QtWidgets import QStackedWidget
        grp = QGroupBox("Detekcia hrán")
        layout = QVBoxLayout(grp)

        # Method selector
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Metóda:"))
        self._edge_method_combo = QComboBox()
        self._edge_method_combo.addItems(EDGE_METHODS)
        method_row.addWidget(self._edge_method_combo, stretch=1)
        layout.addLayout(method_row)

        # Stacked parameter pages
        self._edge_params_stack = QStackedWidget()

        # -- Page 0: Canny --
        canny_page = QWidget()
        canny_form = QFormLayout(canny_page)
        self._canny_t1_spin = QSpinBox()
        self._canny_t1_spin.setRange(0, 255)
        self._canny_t1_spin.setValue(50)
        self._canny_t1_spin.setToolTip("Dolný prah (hrana sa zachová ak ≥ t1)")
        self._canny_t2_spin = QSpinBox()
        self._canny_t2_spin.setRange(0, 255)
        self._canny_t2_spin.setValue(150)
        self._canny_t2_spin.setToolTip("Horný prah (silná hrana ak ≥ t2)")
        self._canny_blur_combo = QComboBox()
        for k in (1, 3, 5, 7, 9):
            self._canny_blur_combo.addItem("vypnutý" if k == 1 else f"{k}×{k}", k)
        self._canny_blur_combo.setCurrentIndex(1)
        canny_form.addRow("Prah 1:", self._canny_t1_spin)
        canny_form.addRow("Prah 2:", self._canny_t2_spin)
        canny_form.addRow("Blur:", self._canny_blur_combo)
        self._edge_params_stack.addWidget(canny_page)  # index 0

        # -- Page 1: Scharr --
        scharr_page = QWidget()
        scharr_form = QFormLayout(scharr_page)
        self._scharr_thresh_spin = QSpinBox()
        self._scharr_thresh_spin.setRange(0, 255)
        self._scharr_thresh_spin.setValue(30)
        self._scharr_thresh_spin.setToolTip("Prah gradientu (0–255, normalizovaný)")
        self._scharr_blur_combo = QComboBox()
        for k in (1, 3, 5, 7, 9):
            self._scharr_blur_combo.addItem("vypnutý" if k == 1 else f"{k}×{k}", k)
        self._scharr_blur_combo.setCurrentIndex(1)
        scharr_form.addRow("Prah:", self._scharr_thresh_spin)
        scharr_form.addRow("Blur:", self._scharr_blur_combo)
        self._edge_params_stack.addWidget(scharr_page)  # index 1

        # -- Page 2: LoG --
        log_page = QWidget()
        log_form = QFormLayout(log_page)
        self._log_sigma_spin = QDoubleSpinBox()
        self._log_sigma_spin.setRange(0.1, 10.0)
        self._log_sigma_spin.setSingleStep(0.1)
        self._log_sigma_spin.setValue(1.5)
        self._log_sigma_spin.setToolTip("Sigma Gaussovho filtra (väčšia = hrubšie hrany)")
        self._log_thresh_spin = QSpinBox()
        self._log_thresh_spin.setRange(0, 255)
        self._log_thresh_spin.setValue(10)
        self._log_thresh_spin.setToolTip("Prah LoG odozvy (0–255, normalizovaný)")
        self._log_blur_combo = QComboBox()
        for k in (1, 3, 5, 7, 9):
            self._log_blur_combo.addItem("vypnutý" if k == 1 else f"{k}×{k}", k)
        self._log_blur_combo.setCurrentIndex(0)
        log_form.addRow("Sigma:", self._log_sigma_spin)
        log_form.addRow("Prah:", self._log_thresh_spin)
        log_form.addRow("Blur:", self._log_blur_combo)
        self._edge_params_stack.addWidget(log_page)  # index 2

        # -- Page 3: Phase Congruency --
        pc_page = QWidget()
        pc_form = QFormLayout(pc_page)
        self._pc_nscale_spin = QSpinBox()
        self._pc_nscale_spin.setRange(1, 8)
        self._pc_nscale_spin.setValue(4)
        self._pc_nscale_spin.setToolTip("Počet mieriek (viac = jemnejší detail)")
        self._pc_minwave_spin = QSpinBox()
        self._pc_minwave_spin.setRange(2, 40)
        self._pc_minwave_spin.setValue(6)
        self._pc_minwave_spin.setToolTip("Min. vlnová dĺžka v px (väčšia = hrubšie hrany)")
        self._pc_mult_spin = QDoubleSpinBox()
        self._pc_mult_spin.setRange(1.1, 4.0)
        self._pc_mult_spin.setSingleStep(0.1)
        self._pc_mult_spin.setValue(2.1)
        self._pc_mult_spin.setToolTip("Multiplikátor mierky medzi filtrami")
        self._pc_k_spin = QDoubleSpinBox()
        self._pc_k_spin.setRange(0.1, 10.0)
        self._pc_k_spin.setSingleStep(0.1)
        self._pc_k_spin.setValue(2.0)
        self._pc_k_spin.setToolTip("Citlivosť (vyššie = menej hrán / menej šumu)")
        pc_form.addRow("Mierky:", self._pc_nscale_spin)
        pc_form.addRow("Min. vlnová dĺžka:", self._pc_minwave_spin)
        pc_form.addRow("Multiplikátor:", self._pc_mult_spin)
        pc_form.addRow("Prah k:", self._pc_k_spin)
        self._edge_params_stack.addWidget(pc_page)  # index 3

        # -- Page 4: DexiNed --
        dexined_page = QWidget()
        dexined_layout = QVBoxLayout(dexined_page)
        dexined_form = QFormLayout()

        self._dexined_weights_edit = QLineEdit()
        self._dexined_weights_edit.setPlaceholderText("models/dexined.onnx (default)")
        self._dexined_weights_edit.setToolTip(
            "Cesta k .pth súboru s váhami DexiNed modelu.\n"
            "Prázdne = automaticky hľadá models/dexined.pth\n"
            "Stiahni váhy: py -3.12 models/download_dexined.py"
        )
        weights_row = QHBoxLayout()
        weights_row.addWidget(self._dexined_weights_edit)
        weights_browse_btn = QPushButton("…")
        weights_browse_btn.setFixedWidth(28)
        weights_browse_btn.clicked.connect(self._on_browse_dexined_weights)  # type: ignore
        weights_row.addWidget(weights_browse_btn)
        dexined_form.addRow("Váhy (.pth):", weights_row)

        self._dexined_thresh_spin = QDoubleSpinBox()
        self._dexined_thresh_spin.setRange(0.01, 0.99)
        self._dexined_thresh_spin.setSingleStep(0.05)
        self._dexined_thresh_spin.setValue(0.5)
        self._dexined_thresh_spin.setDecimals(2)
        self._dexined_thresh_spin.setToolTip("Prah sigmoid výstupu (0–1). Vyšší = menej hrán.")
        dexined_form.addRow("Prah:", self._dexined_thresh_spin)

        self._dexined_device_combo = QComboBox()
        self._dexined_device_combo.addItems(["cpu", "cuda"])
        self._dexined_device_combo.setToolTip("cpu = bez GPU (pomalšie, vždy funguje)\ncuda = NVIDIA GPU (rýchlejšie)")
        dexined_form.addRow("Zariadenie:", self._dexined_device_combo)

        dexined_layout.addLayout(dexined_form)

        # Download button
        dl_btn = QPushButton("Stiahnuť váhy (Hugging Face)")
        dl_btn.setToolTip("Spustí models/download_dexined.py — stiahne ~15 MB z Hugging Face")
        dl_btn.clicked.connect(self._on_download_dexined_weights)
        dexined_layout.addWidget(dl_btn)

        self._dexined_status_label = QLabel("")
        self._dexined_status_label.setWordWrap(True)
        self._dexined_status_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        dexined_layout.addWidget(self._dexined_status_label)
        dexined_layout.addStretch()

        self._edge_params_stack.addWidget(dexined_page)  # index 4

        layout.addWidget(self._edge_params_stack)

        # Min. dĺžka segmentu
        min_len_row = QHBoxLayout()
        min_len_row.addWidget(QLabel("Min. dĺžka segmentu [px]:"))
        self._min_seg_len_spin = QSpinBox()
        self._min_seg_len_spin.setRange(0, 9999)
        self._min_seg_len_spin.setValue(0)
        self._min_seg_len_spin.setToolTip(
            "Segmenty kratšie ako tento počet pixelov budú automaticky odfiltrované.\n"
            "0 = žiadny filter (zachová sa všetko)."
        )
        min_len_row.addWidget(self._min_seg_len_spin)
        layout.addLayout(min_len_row)

        # Odstraňovanie segmentov hrán — riadok 1: klik + oblasť
        seg_row1 = QHBoxLayout()
        self._remove_seg_btn = QPushButton("Odstrániť (klik)")
        self._remove_seg_btn.setCheckable(True)
        self._remove_seg_btn.setToolTip("Klikni na hranu v referenčnom obraze pre jej odstránenie.")
        self._area_seg_btn = QPushButton("Odstrániť (oblasť)")
        self._area_seg_btn.setCheckable(True)
        self._area_seg_btn.setToolTip("Ťahaj obdĺžnik — odstráni všetky segmenty v oblasti.")
        seg_row1.addWidget(self._remove_seg_btn)
        seg_row1.addWidget(self._area_seg_btn)
        layout.addLayout(seg_row1)

        # Riadok 2: undo + reset
        seg_row2 = QHBoxLayout()
        self._undo_seg_btn = QPushButton("Späť (Ctrl+Z)")
        self._undo_seg_btn.setEnabled(False)
        self._undo_seg_btn.setToolTip("Vráti posledné odstránenie segmentu.")
        self._reset_seg_btn = QPushButton("Resetovať")
        self._reset_seg_btn.setEnabled(False)
        seg_row2.addWidget(self._undo_seg_btn)
        seg_row2.addWidget(self._reset_seg_btn)
        layout.addLayout(seg_row2)

        self._seg_count_label = QLabel("Odstránených: 0 segmentov")
        self._seg_count_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        layout.addWidget(self._seg_count_label)

        # ── Select mode + template matching ──────────────────────────────
        from PyQt6.QtWidgets import QFrame
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #555;")
        layout.addWidget(sep)

        sel_row = QHBoxLayout()
        self._select_seg_btn = QPushButton("Vybrať segment")
        self._select_seg_btn.setCheckable(True)
        self._select_seg_btn.setToolTip(
            "Klikni na segment v referenčnom obraze pre jeho výber.\n"
            "Vybraný segment sa použije pre template matching na inšpekčnom obraze."
        )
        self._clear_selection_btn = QPushButton("Zrušiť výber")
        self._clear_selection_btn.setToolTip("Zruší výber segmentu.")
        self._clear_selection_btn.setEnabled(False)
        sel_row.addWidget(self._select_seg_btn)
        sel_row.addWidget(self._clear_selection_btn)
        layout.addLayout(sel_row)

        self._seg_info_label = QLabel("Vybraný segment: —")
        self._seg_info_label.setStyleSheet("color: #ffaa44; font-size: 10px;")
        layout.addWidget(self._seg_info_label)

        self._find_segment_btn = QPushButton("Hľadať segment na inšpekčnom obraze")
        self._find_segment_btn.setEnabled(False)
        self._find_segment_btn.setToolTip(
            "Nájde vybraný segment na inšpekčnom obraze pomocou NCC template matchingu.\n"
            "Ak je nastavené Inšpekčné ROI, vyhľadáva len v tej oblasti."
        )
        layout.addWidget(self._find_segment_btn)

        self._match_result_label = QLabel("Výsledok: —")
        self._match_result_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self._match_result_label.setWordWrap(True)
        layout.addWidget(self._match_result_label)

        return grp

    def _build_insp_group(self) -> QGroupBox:
        grp = QGroupBox("Inšpekčný obraz")
        layout = QVBoxLayout(grp)
        btn = QPushButton("Načítaj obraz…")
        btn.clicked.connect(self._on_load_inspection)
        self._insp_path_label = QLabel("—")
        self._insp_path_label.setWordWrap(True)
        layout.addWidget(btn)
        layout.addWidget(self._insp_path_label)
        return grp

    def _build_insp_roi_group(self) -> QGroupBox:
        grp = QGroupBox("Inšpekčné ROI (oblasť vyhľadávania)")
        layout = QVBoxLayout(grp)

        btn_row = QHBoxLayout()
        self._draw_insp_roi_btn = QPushButton("Kresliť ROI")
        self._draw_insp_roi_btn.setCheckable(True)
        self._draw_insp_roi_btn.setToolTip("Ťahaj obdĺžnik na inšpekčnom obraze pre nastavenie ROI.")
        self._clear_insp_roi_btn = QPushButton("Zmazať ROI")
        self._clear_insp_roi_btn.setToolTip("Zmaže inšpekčné ROI — hrany sa budú hľadať v celom obraze.")
        btn_row.addWidget(self._draw_insp_roi_btn)
        btn_row.addWidget(self._clear_insp_roi_btn)
        layout.addLayout(btn_row)

        form = QFormLayout()
        self._insp_roi_x0 = QSpinBox(); self._insp_roi_x0.setRange(0, 9999)
        self._insp_roi_y0 = QSpinBox(); self._insp_roi_y0.setRange(0, 9999)
        self._insp_roi_x1 = QSpinBox(); self._insp_roi_x1.setRange(0, 9999)
        self._insp_roi_y1 = QSpinBox(); self._insp_roi_y1.setRange(0, 9999)
        form.addRow("x0:", self._insp_roi_x0)
        form.addRow("y0:", self._insp_roi_y0)
        form.addRow("x1:", self._insp_roi_x1)
        form.addRow("y1:", self._insp_roi_y1)
        layout.addLayout(form)
        return grp

    def _build_algo_group(self) -> QGroupBox:
        grp = QGroupBox("Algoritmus zarovnania")
        form = QFormLayout(grp)
        self._algo_combo = QComboBox()
        self._algo_combo.addItems(["ECC", "POC"])
        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(100, 10000)
        self._iter_spin.setValue(2000)
        self._epsilon_edit = QLineEdit("1e-08")
        self._auto_clahe_check = QCheckBox("Auto CLAHE")
        self._gauss_filt_combo = QComboBox()
        for val, lbl in ((1, "1 — vypnutý"), (3, "3×3"), (5, "5×5"), (7, "7×7")):
            self._gauss_filt_combo.addItem(lbl, val)
        self._gauss_filt_combo.setCurrentIndex(1)  # default: 3×3
        self._gauss_filt_combo.setToolTip(
            "Gaussovský filter pre výpočet gradientov v ECC.\n"
            "1 = vypnutý (najvyššia presnosť pre čisté obrazy).\n"
            "3–7 = potlačenie šumu (znižuje presnosť)."
        )
        form.addRow("Algoritmus:", self._algo_combo)
        form.addRow("Max iterácií:", self._iter_spin)
        form.addRow("Epsilon:", self._epsilon_edit)
        form.addRow("Gauss filter:", self._gauss_filt_combo)
        form.addRow("", self._auto_clahe_check)
        return grp

    def _build_calib_group(self) -> QGroupBox:
        grp = QGroupBox("Kalibrácia")
        form = QFormLayout(grp)
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.0001, 100.0)
        self._scale_spin.setDecimals(6)
        self._scale_spin.setSingleStep(0.001)
        self._scale_spin.setValue(1.0)
        self._scale_spin.setSuffix(" mm/px")
        form.addRow("Mierka:", self._scale_spin)
        return grp

    def _build_profile_group(self) -> QGroupBox:
        grp = QGroupBox("Profil")
        layout = QVBoxLayout(grp)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("názov profilu")
        form.addRow("Názov:", self._name_edit)
        layout.addLayout(form)

        save_btn = QPushButton("Uložiť profil")
        save_btn.clicked.connect(self._on_save_profile)
        layout.addWidget(save_btn)

        load_row = QHBoxLayout()
        self._profile_combo = QComboBox()
        load_btn = QPushButton("Načítať")
        load_btn.clicked.connect(self._on_load_profile)
        load_row.addWidget(self._profile_combo, stretch=1)
        load_row.addWidget(load_btn)
        layout.addLayout(load_row)

        del_btn = QPushButton("Zmazať profil")
        del_btn.clicked.connect(self._on_delete_profile)
        layout.addWidget(del_btn)
        return grp

    def _build_result_group(self) -> QGroupBox:
        grp = QGroupBox("Výsledok zarovnania")
        form = QFormLayout(grp)
        self._res_dx_px   = QLabel("—")
        self._res_dy_px   = QLabel("—")
        self._res_dx_mm   = QLabel("—")
        self._res_dy_mm   = QLabel("—")
        self._res_angle   = QLabel("—")
        self._res_conf    = QLabel("—")
        self._res_ncc     = QLabel("—")
        for lbl in (self._res_dx_px, self._res_dy_px, self._res_dx_mm,
                    self._res_dy_mm, self._res_angle, self._res_conf, self._res_ncc):
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("dx [px]:", self._res_dx_px)
        form.addRow("dy [px]:", self._res_dy_px)
        form.addRow("dx [mm]:", self._res_dx_mm)
        form.addRow("dy [mm]:", self._res_dy_mm)
        form.addRow("Uhol [°]:", self._res_angle)
        form.addRow("Spoľahlivosť:", self._res_conf)
        form.addRow("NCC:", self._res_ncc)
        return grp

    # ------------------------------------------------------------------
    # Signály
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._ref_viewer.roi_selected.connect(self._on_roi_selected)
        self._draw_roi_btn.toggled.connect(self._ref_viewer.set_roi_mode)
        self._clear_roi_btn.clicked.connect(self._on_clear_roi)

        # Metóda hrán → prepni stránku + live update
        self._edge_method_combo.currentIndexChanged.connect(self._on_edge_method_changed)

        # Canny parametre
        self._canny_t1_spin.valueChanged.connect(self._on_edge_changed)
        self._canny_t2_spin.valueChanged.connect(self._on_edge_changed)
        self._canny_blur_combo.currentIndexChanged.connect(self._on_edge_changed)

        # Scharr parametre
        self._scharr_thresh_spin.valueChanged.connect(self._on_edge_changed)
        self._scharr_blur_combo.currentIndexChanged.connect(self._on_edge_changed)

        # LoG parametre
        self._log_sigma_spin.valueChanged.connect(self._on_edge_changed)
        self._log_thresh_spin.valueChanged.connect(self._on_edge_changed)
        self._log_blur_combo.currentIndexChanged.connect(self._on_edge_changed)

        # Phase Congruency parametre
        self._pc_nscale_spin.valueChanged.connect(self._on_edge_changed)
        self._pc_minwave_spin.valueChanged.connect(self._on_edge_changed)
        self._pc_mult_spin.valueChanged.connect(self._on_edge_changed)
        self._pc_k_spin.valueChanged.connect(self._on_edge_changed)

        # DexiNed parametre
        self._dexined_thresh_spin.valueChanged.connect(self._on_edge_changed)
        self._dexined_device_combo.currentTextChanged.connect(self._on_edge_changed)
        self._dexined_weights_edit.editingFinished.connect(self._on_edge_changed)

        # ROI spinboxy
        for spin in (self._roi_x0, self._roi_y0, self._roi_x1, self._roi_y1):
            spin.valueChanged.connect(self._on_roi_spinbox_changed)

        # Ostatné nastavenia → profile_changed
        self._scale_spin.valueChanged.connect(self._emit_profile_changed)
        self._algo_combo.currentTextChanged.connect(self._emit_profile_changed)
        self._iter_spin.valueChanged.connect(self._emit_profile_changed)
        self._epsilon_edit.textChanged.connect(self._emit_profile_changed)
        self._name_edit.textChanged.connect(self._emit_profile_changed)
        self._auto_clahe_check.toggled.connect(self._emit_profile_changed)

        self._run_btn.clicked.connect(self._on_run_alignment)

        self._remove_seg_btn.toggled.connect(self._on_remove_seg_mode_toggled)
        self._area_seg_btn.toggled.connect(self._on_area_seg_mode_toggled)
        self._reset_seg_btn.clicked.connect(self._on_reset_segments)
        self._undo_seg_btn.clicked.connect(self._on_undo_segment)
        self._ref_viewer.image_clicked.connect(self._on_ref_image_clicked)
        self._ref_viewer.segment_area_selected.connect(self._on_segment_area_selected)

        # Select mode + template matching
        self._select_seg_btn.toggled.connect(self._on_select_seg_mode_toggled)
        self._clear_selection_btn.clicked.connect(self._on_clear_selection)
        self._find_segment_btn.clicked.connect(self._find_segment_on_inspection)

        # Min-length filter
        self._min_seg_len_spin.valueChanged.connect(self._on_edge_changed)

        # Inšpekčné ROI
        self._draw_insp_roi_btn.toggled.connect(self._insp_viewer.set_roi_mode)
        self._clear_insp_roi_btn.clicked.connect(self._on_clear_insp_roi)
        self._insp_viewer.roi_selected.connect(self._on_insp_roi_selected)
        for spin in (self._insp_roi_x0, self._insp_roi_y0, self._insp_roi_x1, self._insp_roi_y1):
            spin.valueChanged.connect(self._on_insp_roi_spinbox_changed)

        from PyQt6.QtGui import QKeySequence, QShortcut
        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._undo_shortcut.activated.connect(self._on_undo_segment)

    # ------------------------------------------------------------------
    # Handlery
    # ------------------------------------------------------------------

    def _on_load_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Vyber referenčný obraz", "",
            "Obrázky (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if path:
            self._load_ref_from_path(path)
            self._ref_path_label.setText(path)
            self._emit_profile_changed()

    def _load_ref_from_path(self, path: str) -> None:
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Chyba", f"Nemôžem načítať obraz:\n{path}")
            return
        self._ref_image = img
        self._ref_viewer.set_image(img)
        if self._roi is not None:
            self._ref_viewer.draw_roi(self._roi)
            self._update_roi_spinboxes()
        # Vždy spusti detekciu hrán (aj bez ROI)
        self._update_ref_edges()

    def _on_load_inspection(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Vyber inšpekčný obraz", "",
            "Obrázky (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if path:
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, "Chyba", f"Nemôžem načítať obraz:\n{path}")
                return
            self._insp_image = img
            self._insp_viewer.set_image(img)
            if self._insp_roi is not None:
                self._insp_viewer.draw_roi(self._insp_roi)
                self._update_insp_roi_spinboxes()
            self._insp_path_label.setText(path)
            self._update_insp_edges()
            # Aktivuj "Hľadať" ak je segment vybraný
            if self._selected_label is not None:
                self._find_segment_btn.setEnabled(True)

    def _on_roi_selected(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._draw_roi_btn.setChecked(False)
        self._remove_seg_btn.setChecked(False)
        self._area_seg_btn.setChecked(False)
        try:
            roi = ROI(x0, y0, x1, y1)
            if not roi.is_valid():
                return
            self._roi = roi
            self._update_roi_spinboxes()
            self._ref_viewer.draw_roi(roi)
            self._update_ref_edges()
            self._emit_profile_changed()
        except ValueError:
            pass

    def _on_clear_roi(self) -> None:
        self._roi = None
        self._ref_edges = None
        self._segment_labels = None
        self._removed_labels.clear()
        self._undo_stack.clear()
        self._update_seg_ui()
        self._ref_viewer.clear_roi()
        self._ref_viewer.clear_overlay()
        self._insp_viewer.clear_overlay()
        for spin in (self._roi_x0, self._roi_y0, self._roi_x1, self._roi_y1):
            spin.blockSignals(True)
            spin.setValue(0)
            spin.blockSignals(False)
        self._emit_profile_changed()

    def _on_roi_spinbox_changed(self) -> None:
        x0 = self._roi_x0.value()
        y0 = self._roi_y0.value()
        x1 = self._roi_x1.value()
        y1 = self._roi_y1.value()
        if x1 > x0 and y1 > y0:
            try:
                roi = ROI(x0, y0, x1, y1)
                self._roi = roi
                self._ref_viewer.draw_roi(roi)
                self._update_ref_edges()
                self._emit_profile_changed()
            except ValueError:
                pass

    def _on_browse_dexined_weights(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Vyber model DexiNed", "", "ONNX model (*.onnx);;PyTorch checkpoint (*.pth *.pt)"
        )
        if path:
            self._dexined_weights_edit.setText(path)
            self._on_edge_changed()

    def _on_download_dexined_weights(self) -> None:
        import subprocess, sys
        script = str(Path(__file__).parent.parent.parent / "models" / "download_dexined.py")
        self._dexined_status_label.setText("Sťahujem váhy… (môže trvať ~30 s)")
        self._dexined_status_label.setStyleSheet("color: #ffcc00; font-size: 10px;")
        try:
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                self._dexined_status_label.setText("Váhy stiahnuté ✓")
                self._dexined_status_label.setStyleSheet("color: #44ff88; font-size: 10px;")
            else:
                self._dexined_status_label.setText(f"Chyba: {result.stderr[:120]}")
                self._dexined_status_label.setStyleSheet("color: #ff6666; font-size: 10px;")
        except Exception as e:
            self._dexined_status_label.setText(f"Chyba: {e}")
            self._dexined_status_label.setStyleSheet("color: #ff6666; font-size: 10px;")

    def _on_edge_method_changed(self, idx: int) -> None:
        self._edge_params_stack.setCurrentIndex(idx)
        self._on_edge_changed()

    def _on_edge_changed(self) -> None:
        self._update_ref_edges()
        self._update_insp_edges()
        self._emit_profile_changed()

    def _on_run_alignment(self) -> None:
        if self._ref_image is None:
            QMessageBox.warning(self, "Chyba", "Najprv načítaj referenčný obraz.")
            return
        if self._insp_image is None:
            QMessageBox.warning(self, "Chyba", "Najprv načítaj inšpekčný obraz.")
            return

        profile = self.get_profile()

        # Zmenšenie inšpekčného obrazu na veľkosť referenčného ak sa líšia
        ref_h, ref_w = self._ref_image.shape[:2]
        insp_h, insp_w = self._insp_image.shape[:2]
        insp_work = self._insp_image
        if (insp_h, insp_w) != (ref_h, ref_w):
            insp_work = cv2.resize(
                self._insp_image, (ref_w, ref_h), interpolation=cv2.INTER_AREA
            )

        ref_pre = preprocess(
            self._ref_image,
            auto_clahe=profile.auto_clahe,
        )
        insp_pre = preprocess(
            insp_work,
            auto_clahe=profile.auto_clahe,
        )

        mask = None
        if self._roi is not None and self._roi.is_valid(ref_pre.shape[:2]):
            mask = self._roi.create_mask(ref_pre.shape[:2])

        try:
            raw = align(
                ref_pre, insp_pre,
                max_iter=profile.ecc_max_iter,
                epsilon=profile.ecc_epsilon,
                mask=mask,
                algorithm=profile.algorithm,
                gauss_filt_size=profile.ecc_gauss_filt_size,
            )
        except Exception as e:
            QMessageBox.critical(self, "Chyba zarovnania", str(e))
            return

        cal = Calibration(mm_per_px=profile.scale_mm_per_px)
        result = AlignResult.from_dict(raw, cal)

        # Aktualizuj labely výsledkov
        self._res_dx_px.setText(f"{result.dx_px:.4f}")
        self._res_dy_px.setText(f"{result.dy_px:.4f}")
        self._res_dx_mm.setText(f"{result.dx_mm:.4f}")
        self._res_dy_mm.setText(f"{result.dy_mm:.4f}")
        self._res_angle.setText(f"{result.angle_deg:.4f}")
        self._res_conf.setText(f"{result.confidence:.4f}")
        self._res_ncc.setText(f"{result.ncc_score:.4f}")

        _MIN_CONF = 0.60

        # Pri príliš nízkej spoľahlivosti — zastav, zobraz chybu, nekresli overlay
        if result.confidence < _MIN_CONF:
            for lbl in (self._res_dx_px, self._res_dy_px, self._res_dx_mm,
                        self._res_dy_mm, self._res_angle):
                lbl.setStyleSheet("color: #ff6666;")
            self._res_conf.setStyleSheet("color: #ff6666; font-weight: bold;")
            self._res_ncc.setStyleSheet("color: #ff6666;")
            self._insp_viewer.set_image(insp_work)
            return

        # Resetuj farbu výsledkových labelov
        for lbl in (self._res_dx_px, self._res_dy_px, self._res_dx_mm,
                    self._res_dy_mm, self._res_angle, self._res_conf, self._res_ncc):
            lbl.setStyleSheet("")

        # Aktualizuj inšpekčný viewer — zobraz pracovnú (príp. resizovanú) verziu
        self._insp_viewer.set_image(insp_work)

        # Projektuj referenčné hrany na inšpekčný obraz (zelená farba)
        projected_rgba: np.ndarray | None = None
        active_edges = self._active_ref_edges
        if active_edges is not None:
            h_e, w_e = active_edges.shape
            h_i, w_i = self._insp_image.shape[:2]
            # Použi rozmery inšpekčného obrazu pre projekciu
            h_out = min(h_e, h_i)
            w_out = min(w_e, w_i)
            cx, cy = w_e / 2.0, h_e / 2.0
            M = cv2.getRotationMatrix2D((cx, cy), result.angle_deg, 1.0)
            M[0, 2] += result.dx_px
            M[1, 2] += result.dy_px
            projected = cv2.warpAffine(active_edges, M, (w_out, h_out))
            # Predpriprav RGBA s zelenou farbou
            rgba = np.zeros((h_out, w_out, 4), dtype=np.uint8)
            rgba[projected > 0] = [0, 255, 80, 220]  # RGBA zelená
            projected_rgba = np.ascontiguousarray(rgba)

        # Zlúč s inšpekčnými hranami (cyan) ak sú dostupné
        # Cyan vrstva ide pod zelenú (ref. hrany majú prednosť)
        if self._insp_edges is not None:
            h_i, w_i = self._insp_image.shape[:2]
            h_ie, w_ie = self._insp_edges.shape
            h_c = min(h_i, h_ie)
            w_c = min(w_i, w_ie)
            cyan_rgba = np.zeros((h_c, w_c, 4), dtype=np.uint8)
            cyan_rgba[self._insp_edges[:h_c, :w_c] > 0] = [0, 220, 255, 180]  # cyan
            if projected_rgba is not None:
                # Prepíš cyan pixely zelenou kde sú ref. hrany
                h_p, w_p = projected_rgba.shape[:2]
                h_m = min(h_c, h_p)
                w_m = min(w_c, w_p)
                green_mask = projected_rgba[:h_m, :w_m, 3] > 0
                combined = cyan_rgba.copy()
                combined[:h_m, :w_m][green_mask] = projected_rgba[:h_m, :w_m][green_mask]
                projected_rgba = np.ascontiguousarray(combined)
            else:
                projected_rgba = cyan_rgba

        self._insp_viewer.draw_overlay(result.dx_px, result.dy_px, projected_rgba)

    # ------------------------------------------------------------------
    # Pomocné metódy
    # ------------------------------------------------------------------

    def _get_edge_params(self) -> dict:
        """Vráti parametre aktívnej metódy detekcie hrán."""
        method = self._edge_method_combo.currentText()
        if method == "Canny":
            return {
                "t1": self._canny_t1_spin.value(),
                "t2": self._canny_t2_spin.value(),
                "blur": self._canny_blur_combo.currentData() or 3,
            }
        elif method == "Scharr":
            return {
                "threshold": self._scharr_thresh_spin.value(),
                "blur": self._scharr_blur_combo.currentData() or 3,
            }
        elif method == "LoG":
            return {
                "sigma": self._log_sigma_spin.value(),
                "threshold": self._log_thresh_spin.value(),
                "blur": self._log_blur_combo.currentData() or 1,
            }
        elif method == "PhaseCongruency":
            return {
                "nscale": self._pc_nscale_spin.value(),
                "min_wavelength": self._pc_minwave_spin.value(),
                "mult": self._pc_mult_spin.value(),
                "k": self._pc_k_spin.value(),
            }
        elif method == "DexiNed":
            return {
                "weights_path": self._dexined_weights_edit.text().strip(),
                "threshold":    self._dexined_thresh_spin.value(),
                "device":       self._dexined_device_combo.currentText(),
            }
        return {}

    def _update_ref_edges(self) -> None:
        """Prepočíta hrany referenčného obrazu a zobrazí ich.

        Ak je nastavené ROI, hrany sa počítajú len vo vnútri ROI.
        Ak ROI nie je nastavené, hrany sa počítajú z celého obrazu —
        vždy dostupné pre projekciu na inšpekčný obraz.
        """
        if self._ref_image is None:
            return

        gray = (cv2.cvtColor(self._ref_image, cv2.COLOR_BGR2GRAY)
                if self._ref_image.ndim == 3 else self._ref_image.copy())

        method = self._edge_method_combo.currentText()
        params = self._get_edge_params()

        # DexiNed je pomalý — spusti v background threade aby nezamrzlo GUI
        if method == "DexiNed":
            # Zastav predchádzajúci worker ak beží
            if self._edge_worker is not None and self._edge_worker.isRunning():
                self._edge_worker.requestInterruption()
                self._edge_worker.wait(500)
            self._dexined_status_label.setText("Počítam hrany (DexiNed)…")
            self._dexined_status_label.setStyleSheet("color: #ffcc00; font-size: 10px;")
            self._edge_worker = _EdgeWorker(gray, self._roi, method, params)
            self._edge_worker.result_ready.connect(self._on_edge_result)
            self._edge_worker.error.connect(self._on_edge_error)
            self._edge_worker.start()
            return

        # Ostatné metódy — synchrónne (rýchle)
        if self._roi is not None:
            h_img, w_img = gray.shape
            x0c = max(0, self._roi.x0)
            y0c = max(0, self._roi.y0)
            x1c = min(w_img, self._roi.x1)
            y1c = min(h_img, self._roi.y1)
            if x1c <= x0c or y1c <= y0c:
                return
            edges_full = np.zeros(gray.shape, dtype=np.uint8)
            edges_full[y0c:y1c, x0c:x1c] = detect_edges(
                gray[y0c:y1c, x0c:x1c], method, **params
            )
        else:
            edges_full = detect_edges(gray, method, **params)

        # Min-length filter pred uložením
        _, lbl_full = cv2.connectedComponents(edges_full, connectivity=8)
        edges_full = self._apply_min_length_filter(edges_full, lbl_full)

        self._ref_edges = edges_full
        self._removed_labels.clear()
        self._erased_mask = np.ones_like(edges_full, dtype=np.uint8) * 255
        self._undo_stack.clear()
        self._segment_labels = self._compute_segment_labels(edges_full)
        self._update_seg_ui()
        self._ref_viewer.draw_edges(edges_full)

    def _update_roi_spinboxes(self) -> None:
        if self._roi is None:
            return
        for spin, val in zip(
            (self._roi_x0, self._roi_y0, self._roi_x1, self._roi_y1),
            (self._roi.x0, self._roi.y0, self._roi.x1, self._roi.y1),
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

    def _on_save_profile(self) -> None:
        profile = self.get_profile()
        errors = profile.validate()
        if errors:
            QMessageBox.warning(self, "Neplatný profil", "\n".join(errors))
            return
        try:
            self._config_mgr.save_profile(profile)
            self._refresh_profile_list()
            QMessageBox.information(self, "OK", f"Profil '{profile.name}' uložený.")
        except Exception as e:
            QMessageBox.critical(self, "Chyba", str(e))

    def _on_load_profile(self) -> None:
        name = self._profile_combo.currentText()
        if not name:
            return
        try:
            profile = self._config_mgr.load_profile(name)
            self.set_profile(profile)
            self._emit_profile_changed()
        except Exception as e:
            QMessageBox.critical(self, "Chyba", str(e))

    def _on_delete_profile(self) -> None:
        name = self._profile_combo.currentText()
        if not name:
            return
        reply = QMessageBox.question(
            self, "Zmazať profil?",
            f"Naozaj zmazať profil '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self._config_mgr.delete_profile(name)
                self._refresh_profile_list()
            except Exception as e:
                QMessageBox.critical(self, "Chyba", str(e))

    def _refresh_profile_list(self) -> None:
        names = self._config_mgr.list_profiles()
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        self._profile_combo.addItems(names)
        self._profile_combo.blockSignals(False)

    def _emit_profile_changed(self) -> None:
        self.profile_changed.emit(self.get_profile())

    @pyqtSlot(object)
    def _on_edge_result(self, edges: np.ndarray) -> None:
        """Prijme výsledok z EdgeWorker threadu (DexiNed)."""
        # Min-length filter
        _, lbl = cv2.connectedComponents(edges, connectivity=8)
        edges = self._apply_min_length_filter(edges, lbl)

        self._ref_edges = edges
        self._removed_labels.clear()
        self._erased_mask = np.ones_like(edges, dtype=np.uint8) * 255
        self._undo_stack.clear()
        self._segment_labels = self._compute_segment_labels(edges)
        self._update_seg_ui()
        self._ref_viewer.draw_edges(edges)
        self._dexined_status_label.setText("Hrany vypočítané ✓")
        self._dexined_status_label.setStyleSheet("color: #44ff88; font-size: 10px;")

    @pyqtSlot(str)
    def _on_edge_error(self, msg: str) -> None:
        """Zobrazí chybu z EdgeWorker threadu."""
        self._dexined_status_label.setText(f"Chyba: {msg}")
        self._dexined_status_label.setStyleSheet("color: #ff6666; font-size: 10px;")

    # ------------------------------------------------------------------
    # Odstraňovanie segmentov hrán
    # ------------------------------------------------------------------

    @property
    def _active_ref_edges(self) -> np.ndarray | None:
        """Edge maska bez odstránených segmentov a vymazaných pixelov."""
        if self._ref_edges is None:
            return None
        mask = self._ref_edges.copy()
        if self._removed_labels and self._segment_labels is not None:
            for lbl in self._removed_labels:
                mask[self._segment_labels == lbl] = 0
        if self._erased_mask is not None:
            mask &= self._erased_mask      # pixel-level erase
        return mask

    def _compute_segment_labels(self, edges: np.ndarray) -> np.ndarray:
        """Vráti int32 label mapu prepojených komponentov hrán."""
        _, labels = cv2.connectedComponents(edges, connectivity=8)
        return labels

    def _update_seg_ui(self) -> None:
        n_comp = len(self._removed_labels)
        n_px = 0
        if self._erased_mask is not None and self._ref_edges is not None:
            n_px = int(np.count_nonzero(self._ref_edges & (~self._erased_mask)))
        parts = []
        if n_comp:
            parts.append(f"{n_comp} segm.")
        if n_px:
            parts.append(f"{n_px} px")
        self._seg_count_label.setText("Odstránených: " + (", ".join(parts) or "0"))
        self._reset_seg_btn.setEnabled(n_comp > 0 or n_px > 0)
        self._undo_seg_btn.setEnabled(len(self._undo_stack) > 0)

    def _push_undo(self) -> None:
        """Uloží aktuálny stav pred každou deštruktívnou operáciou."""
        em = self._erased_mask.copy() if self._erased_mask is not None else None
        self._undo_stack.append((frozenset(self._removed_labels), em))

    def _on_remove_seg_mode_toggled(self, enabled: bool) -> None:
        if enabled:
            self._draw_roi_btn.setChecked(False)
            self._area_seg_btn.setChecked(False)
            self._select_seg_btn.setChecked(False)
        from src.gui.image_viewer import ViewerMode
        self._ref_viewer.set_mode(ViewerMode.CLICK if enabled else ViewerMode.NONE)

    def _on_area_seg_mode_toggled(self, enabled: bool) -> None:
        if enabled:
            self._draw_roi_btn.setChecked(False)
            self._remove_seg_btn.setChecked(False)
            self._select_seg_btn.setChecked(False)
        from src.gui.image_viewer import ViewerMode
        self._ref_viewer.set_mode(ViewerMode.SEGMENT_AREA if enabled else ViewerMode.NONE)

    def _on_ref_image_clicked(self, x: int, y: int) -> None:
        if self._segment_labels is None or self._ref_edges is None:
            return
        h, w = self._segment_labels.shape
        if not (0 <= y < h and 0 <= x < w):
            return
        label = int(self._segment_labels[y, x])
        if label == 0:
            return  # background

        if self._select_seg_btn.isChecked():
            # SELECT MODE: vyber segment pre template matching
            self._selected_label = label
            self._update_seg_info()
            self._ref_viewer.draw_edges_with_selection(
                self._active_ref_edges, self._segment_labels, label
            )
        else:
            # REMOVE MODE (pôvodné správanie)
            self._push_undo()
            self._removed_labels.add(label)
            self._update_seg_ui()
            self._ref_viewer.draw_edges(self._active_ref_edges)

    @pyqtSlot(int, int, int, int)
    def _on_segment_area_selected(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Vymaže pixely hrán priamo v oblasti (pixel-level erase, nie celé komponenty)."""
        if self._ref_edges is None or self._erased_mask is None:
            return
        h, w = self._ref_edges.shape
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(w, x1), min(h, y1)
        if x1c <= x0c or y1c <= y0c:
            return
        # Overenie či sú v oblasti nejaké aktívne hrany (aby sa nerobil zbytočný undo)
        active_in_region = self._ref_edges[y0c:y1c, x0c:x1c] & self._erased_mask[y0c:y1c, x0c:x1c]
        if not np.any(active_in_region):
            return
        self._push_undo()
        self._erased_mask[y0c:y1c, x0c:x1c] = 0   # vymaž všetky pixely v obdĺžniku
        # Recompute labels — pixel erase mohol rozbiť komponent na dva nezávislé
        active = self._active_ref_edges
        self._segment_labels = self._compute_segment_labels(active)
        self._update_seg_ui()
        self._ref_viewer.draw_edges(active)

    def _on_undo_segment(self) -> None:
        """Vráti posledné odstránenie segmentu (Ctrl+Z)."""
        if not self._undo_stack:
            return
        prev_labels, prev_mask = self._undo_stack.pop()
        self._removed_labels = set(prev_labels)
        self._erased_mask = prev_mask
        active = self._active_ref_edges
        if active is not None:
            self._segment_labels = self._compute_segment_labels(active)
        self._update_seg_ui()
        self._ref_viewer.draw_edges(active)

    def _on_reset_segments(self) -> None:
        self._removed_labels.clear()
        if self._ref_edges is not None:
            self._erased_mask = np.ones_like(self._ref_edges, dtype=np.uint8) * 255
            self._segment_labels = self._compute_segment_labels(self._ref_edges)
        self._undo_stack.clear()
        self._update_seg_ui()
        self._ref_viewer.draw_edges(self._active_ref_edges)

    # ------------------------------------------------------------------
    # Min-length filter
    # ------------------------------------------------------------------

    def _apply_min_length_filter(self, edges: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Odfiltruje segmenty kratšie ako _min_seg_len_spin pixelov."""
        min_len = self._min_seg_len_spin.value()
        if min_len <= 0:
            return edges
        result = edges.copy()
        n_labels = int(labels.max())
        for lbl in range(1, n_labels + 1):
            if np.count_nonzero(labels == lbl) < min_len:
                result[labels == lbl] = 0
        return result

    # ------------------------------------------------------------------
    # Detekcia hrán na inšpekčnom obraze
    # ------------------------------------------------------------------

    def _update_insp_edges(self) -> None:
        """Detekuje hrany na inšpekčnom obraze (rovnaká metóda ako ref.) a zobrazí ich."""
        if self._insp_image is None:
            return

        gray = (cv2.cvtColor(self._insp_image, cv2.COLOR_BGR2GRAY)
                if self._insp_image.ndim == 3 else self._insp_image.copy())

        method = self._edge_method_combo.currentText()
        params = self._get_edge_params()

        roi = self._insp_roi
        if roi is not None:
            h_img, w_img = gray.shape
            x0c = max(0, roi.x0)
            y0c = max(0, roi.y0)
            x1c = min(w_img, roi.x1)
            y1c = min(h_img, roi.y1)
            if x1c > x0c and y1c > y0c:
                edges_full = np.zeros(gray.shape, dtype=np.uint8)
                edges_full[y0c:y1c, x0c:x1c] = detect_edges(
                    gray[y0c:y1c, x0c:x1c], method, **params
                )
            else:
                edges_full = detect_edges(gray, method, **params)
        else:
            edges_full = detect_edges(gray, method, **params)

        # Min-length filter
        min_len = self._min_seg_len_spin.value()
        if min_len > 0:
            _, lbl = cv2.connectedComponents(edges_full, connectivity=8)
            edges_full = self._apply_min_length_filter(edges_full, lbl)

        self._insp_edges = edges_full
        self._insp_viewer.draw_edges(edges_full)

    # ------------------------------------------------------------------
    # Inšpekčné ROI handlery
    # ------------------------------------------------------------------

    def _on_insp_roi_selected(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._draw_insp_roi_btn.setChecked(False)
        try:
            roi = ROI(x0, y0, x1, y1)
            if not roi.is_valid():
                return
            self._insp_roi = roi
            self._update_insp_roi_spinboxes()
            self._insp_viewer.draw_roi(roi)
            self._update_insp_edges()
        except ValueError:
            pass

    def _on_clear_insp_roi(self) -> None:
        self._insp_roi = None
        self._insp_viewer.clear_roi()
        for spin in (self._insp_roi_x0, self._insp_roi_y0, self._insp_roi_x1, self._insp_roi_y1):
            spin.blockSignals(True)
            spin.setValue(0)
            spin.blockSignals(False)
        self._update_insp_edges()

    def _on_insp_roi_spinbox_changed(self) -> None:
        x0 = self._insp_roi_x0.value()
        y0 = self._insp_roi_y0.value()
        x1 = self._insp_roi_x1.value()
        y1 = self._insp_roi_y1.value()
        if x1 > x0 and y1 > y0:
            try:
                roi = ROI(x0, y0, x1, y1)
                self._insp_roi = roi
                self._insp_viewer.draw_roi(roi)
                self._update_insp_edges()
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Select mode + template matching
    # ------------------------------------------------------------------

    def _on_select_seg_mode_toggled(self, enabled: bool) -> None:
        """Prepína select mode — klik = vyber segment (nie odstrán)."""
        if enabled:
            self._draw_roi_btn.setChecked(False)
            self._remove_seg_btn.setChecked(False)
            self._area_seg_btn.setChecked(False)
        from src.gui.image_viewer import ViewerMode
        self._ref_viewer.set_mode(ViewerMode.CLICK if enabled else ViewerMode.NONE)

    def _on_clear_selection(self) -> None:
        """Zruší výber segmentu a resetuje overlay."""
        self._selected_label = None
        self._update_seg_info()
        # Obnov normálne zobrazenie hrán
        active = self._active_ref_edges
        if active is not None:
            self._ref_viewer.draw_edges(active)
        else:
            self._ref_viewer.clear_overlay()

    def _update_seg_info(self) -> None:
        """Aktualizuje informačný label o vybranom segmente."""
        if self._selected_label is None or self._segment_labels is None:
            self._seg_info_label.setText("Vybraný segment: —")
            self._clear_selection_btn.setEnabled(False)
            self._find_segment_btn.setEnabled(False)
        else:
            n_px = int(np.count_nonzero(self._segment_labels == self._selected_label))
            self._seg_info_label.setText(f"Vybraný segment: #{self._selected_label}  ({n_px} px)")
            self._clear_selection_btn.setEnabled(True)
            self._find_segment_btn.setEnabled(self._insp_image is not None)

    def _find_segment_on_inspection(self) -> None:
        """NCC template matching — nájde vybraný segment na inšpekčnom obraze.

        Template = grayscale výrez referenčného obrazu okolo vybraného segmentu.
        Výsledok = dx/dy odchýlka polohy od referenčnej pozície + NCC skóre.
        """
        if self._selected_label is None:
            self._match_result_label.setText("Najprv vyber segment kliknutím.")
            return
        if self._segment_labels is None or self._ref_edges is None:
            self._match_result_label.setText("Chýbajú hrany referenčného obrazu.")
            return
        if self._insp_image is None or self._ref_image is None:
            self._match_result_label.setText("Najprv načítaj oba obrazy.")
            return

        # 1. Bounding box vybraného segmentu v referenčnom obraze
        seg_px = np.where(self._segment_labels == self._selected_label)
        ys, xs = seg_px
        if len(xs) == 0:
            self._match_result_label.setText("Vybraný segment je prázdny.")
            return

        pad = 20   # väčší padding = stabilnejší NCC na grayscale
        h_ref, w_ref = self._ref_image.shape[:2]
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(w_ref, int(xs.max()) + pad + 1)
        y1 = min(h_ref, int(ys.max()) + pad + 1)
        tw, th = x1 - x0, y1 - y0

        if tw < 5 or th < 5:
            self._match_result_label.setText("Segment je príliš malý pre template matching.")
            return

        # 2. Template = grayscale výrez REFERENČNÉHO obrazu
        ref_gray = (cv2.cvtColor(self._ref_image, cv2.COLOR_BGR2GRAY)
                    if self._ref_image.ndim == 3 else self._ref_image.copy())
        template_base = ref_gray[y0:y1, x0:x1].astype(np.float32)

        # 3. Search image = grayscale inšpekčný obraz (alebo jeho ROI crop)
        insp_gray = (cv2.cvtColor(self._insp_image, cv2.COLOR_BGR2GRAY)
                     if self._insp_image.ndim == 3 else self._insp_image.copy())

        if self._insp_roi is not None:
            roi = self._insp_roi
            hi, wi = insp_gray.shape
            rx0 = max(0, roi.x0); ry0 = max(0, roi.y0)
            rx1 = min(wi, roi.x1); ry1 = min(hi, roi.y1)
            if rx1 > rx0 and ry1 > ry0:
                insp_crop = insp_gray[ry0:ry1, rx0:rx1].astype(np.float32)
                off_x, off_y = rx0, ry0
            else:
                insp_crop = insp_gray.astype(np.float32)
                off_x, off_y = 0, 0
        else:
            insp_crop = insp_gray.astype(np.float32)
            off_x, off_y = 0, 0

        # 4. Multi-scale template matching — hľadaj pri rôznych merítach
        #    Rozsah: 0.4× – 1.6× v krokoch 0.1 (zvláda veľké rozdiely zoom-u)
        scales = [round(s * 0.1, 1) for s in range(4, 17)]   # 0.4 … 1.6
        best_val = -2.0
        best_loc = (0, 0)
        best_scale = 1.0
        best_tw = tw
        best_th = th

        sh, sw = insp_crop.shape[:2]
        for scale in scales:
            s_tw = max(1, int(round(tw * scale)))
            s_th = max(1, int(round(th * scale)))
            if s_th > sh or s_tw > sw:
                continue   # template väčší ako search area — preskočiť
            if s_tw < 4 or s_th < 4:
                continue   # príliš malý template — nestabilný NCC
            scaled_tpl = cv2.resize(template_base, (s_tw, s_th),
                                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
            result_map = cv2.matchTemplate(insp_crop, scaled_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_l = cv2.minMaxLoc(result_map)
            if max_v > best_val:
                best_val = max_v
                best_loc = max_l
                best_scale = scale
                best_tw = s_tw
                best_th = s_th

        if best_val < -1.5:
            self._match_result_label.setText("Search area je príliš malá pre všetky merítka.")
            self._match_result_label.setStyleSheet("color: #ff6666; font-size: 10px;")
            return

        # 5. Absolútna poloha nájdeného templatu v inšpekčnom obraze
        match_x = best_loc[0] + off_x
        match_y = best_loc[1] + off_y

        # 6. Odchýlka od referenčnej pozície (dx = posun medzi ref a insp v px)
        dx = match_x - x0
        dy = match_y - y0

        # 7. Aktualizuj výsledkový label
        sign_x = "+" if dx >= 0 else ""
        sign_y = "+" if dy >= 0 else ""
        scale_info = f"  merítko={best_scale:.1f}×" if abs(best_scale - 1.0) > 0.05 else ""
        self._match_result_label.setText(
            f"dx={sign_x}{dx}px  dy={sign_y}{dy}px\n"
            f"Poloha: x={match_x} y={match_y}{scale_info}\n"
            f"NCC={best_val:.3f}"
        )
        color = "#44ff88" if best_val >= 0.7 else ("#ffcc00" if best_val >= 0.5 else "#ff6666")
        self._match_result_label.setStyleSheet(f"color: {color}; font-size: 10px;")

        # 8. Overlay: oranžový prerušovaný = ref. poloha, zelený/žltý/červený = nájdená
        self._insp_viewer.draw_match_result(x0, y0, tw, th, match_x, match_y, best_val)

    def _update_insp_roi_spinboxes(self) -> None:
        if self._insp_roi is None:
            return
        for spin, val in zip(
            (self._insp_roi_x0, self._insp_roi_y0, self._insp_roi_x1, self._insp_roi_y1),
            (self._insp_roi.x0, self._insp_roi.y0, self._insp_roi.x1, self._insp_roi.y1),
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
