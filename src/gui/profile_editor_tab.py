"""ProfileEditorTab — Tab 1: vytvorenie a editácia profilu."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QDoubleSpinBox,
    QSpinBox, QComboBox, QScrollArea, QFileDialog, QMessageBox,
    QCheckBox, QSizePolicy, QStackedWidget, QSlider, QFrame,
    QInputDialog,
)

from src.core.roi import ROI
from src.core.edge_detector import detect_edges, METHODS as EDGE_METHODS
from src.config.profile import Profile
from src.config.config_manager import ConfigManager
from src.gui.image_viewer import ImageViewer, ViewerMode


# ---------------------------------------------------------------------------
# Background worker pre pomalé metódy (DexiNed)
# ---------------------------------------------------------------------------

class _EdgeWorker(QThread):
    result_ready = pyqtSignal(object)
    error        = pyqtSignal(str)

    def __init__(self, gray: np.ndarray, roi, method: str, params: dict) -> None:
        super().__init__()
        self._gray, self._roi, self._method, self._params = gray, roi, method, params

    def run(self) -> None:
        try:
            if self._roi is not None:
                h, w = self._gray.shape
                x0c = max(0, self._roi.x0); y0c = max(0, self._roi.y0)
                x1c = min(w, self._roi.x1); y1c = min(h, self._roi.y1)
                if x1c <= x0c or y1c <= y0c:
                    return
                out = np.zeros(self._gray.shape, dtype=np.uint8)
                out[y0c:y1c, x0c:x1c] = detect_edges(
                    self._gray[y0c:y1c, x0c:x1c], self._method, **self._params
                )
            else:
                out = detect_edges(self._gray, self._method, **self._params)
            self.result_ready.emit(out)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Helper — slider + spinbox synchronizovaný riadok
# ---------------------------------------------------------------------------

def _make_int_row(
    label: str,
    lo: int, hi: int, default: int, step: int = 1,
    tooltip: str = "",
) -> tuple[QWidget, QSlider, QSpinBox]:
    """Vráti (widget, slider, spinbox) pre celé číslo."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)

    lbl = QLabel(label)
    lbl.setFixedWidth(110)
    h.addWidget(lbl)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(lo, hi)
    slider.setValue(default)
    slider.setSingleStep(step)
    slider.setPageStep(step * 5)

    spin = QSpinBox()
    spin.setRange(lo, hi)
    spin.setValue(default)
    spin.setSingleStep(step)
    spin.setFixedWidth(60)
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    if tooltip:
        slider.setToolTip(tooltip)
        spin.setToolTip(tooltip)

    # sync — equality guard breaks the feedback loop; spinbox.valueChanged fires normally
    slider.valueChanged.connect(lambda v: spin.setValue(v) if spin.value() != v else None)
    spin.valueChanged.connect(lambda v: slider.setValue(v) if slider.value() != v else None)

    h.addWidget(slider, stretch=1)
    h.addWidget(spin)
    return w, slider, spin


def _make_float_row(
    label: str,
    lo: float, hi: float, default: float,
    step: float = 0.1, decimals: int = 2,
    tooltip: str = "",
) -> tuple[QWidget, QSlider, QDoubleSpinBox]:
    """Vráti (widget, slider, spinbox) pre desatinné číslo.
    Slider pracuje s hodnotami * 10^decimals (celé čísla).
    """
    scale = 10 ** decimals
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)

    lbl = QLabel(label)
    lbl.setFixedWidth(110)
    h.addWidget(lbl)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(int(lo * scale), int(hi * scale))
    slider.setValue(int(default * scale))
    slider.setSingleStep(max(1, int(step * scale)))

    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setValue(default)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setFixedWidth(72)
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    if tooltip:
        slider.setToolTip(tooltip)
        spin.setToolTip(tooltip)

    # sync — equality guard breaks the feedback loop; spinbox.valueChanged fires normally
    slider.valueChanged.connect(
        lambda v: spin.setValue(v / scale) if abs(spin.value() - v / scale) > 1e-9 else None
    )
    spin.valueChanged.connect(
        lambda v: slider.setValue(int(v * scale)) if slider.value() != int(v * scale) else None
    )

    h.addWidget(slider, stretch=1)
    h.addWidget(spin)
    return w, slider, spin


# ---------------------------------------------------------------------------
# ProfileEditorTab
# ---------------------------------------------------------------------------

class ProfileEditorTab(QWidget):
    """Tab 1 — nastavenie referenčného obrazu, hrán, segmentov, kalibrácie a uloženie profilu."""

    profile_saved = pyqtSignal(object)   # emituje Profile po uložení

    def __init__(self, config_mgr: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config_mgr = config_mgr
        self._ref_image: np.ndarray | None = None
        self._roi: ROI | None = None
        self._ref_edges: np.ndarray | None = None
        self._edge_worker: _EdgeWorker | None = None
        self._segment_labels: np.ndarray | None = None
        self._removed_labels: set[int] = set()
        self._erased_mask: np.ndarray | None = None
        self._undo_stack: list[tuple[frozenset[int], np.ndarray | None]] = []
        self._selected_label: int | None = None
        self._segment_centroid_ref: tuple[float, float] | None = None
        self._show_edges: bool = True
        self._active_profile_id: int = 0   # 0 = nový (ešte neuložený)

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Verejné
    # ------------------------------------------------------------------

    def get_profile(self) -> Profile:
        try:
            epsilon = float(self._epsilon_edit.text())
        except ValueError:
            epsilon = 1e-8
        return Profile(
            name=self._name_edit.text().strip() or "unnamed",
            id=self._active_profile_id,
            reference_path=(
                self._ref_path_label.text()
                if self._ref_path_label.text() != "—" else ""
            ),
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
            min_seg_len=self._min_seg_len_spin.value(),
            selected_segment_centroid=self._segment_centroid_ref,
        )

    def set_profile(self, profile: Profile) -> None:
        """Naplní formulár hodnotami profilu."""
        self._active_profile_id = profile.id
        self._name_edit.setText(profile.name)
        self._id_label.setText(f"ID: {profile.id if profile.id else '—'}")
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

        self._min_seg_len_spin.setValue(profile.min_seg_len)
        self._segment_centroid_ref = profile.selected_segment_centroid

        self._roi = profile.roi
        ref_path = profile.reference_path
        if ref_path and Path(ref_path).exists():
            self._ref_path_label.setText(ref_path)
            self._load_ref_from_path(ref_path)
        elif self._roi is not None and self._ref_image is not None:
            self._viewer.draw_roi(self._roi)
            self._update_roi_spinboxes()
            self._update_ref_edges()

    # ------------------------------------------------------------------
    # Budovanie UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Ľavý viewer
        self._viewer = ImageViewer()
        self._viewer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        lbl = QLabel("Referenčný obraz")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("font-weight: bold; padding: 2px;")
        left_w = QWidget()
        lv = QVBoxLayout(left_w)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(lbl)
        lv.addWidget(self._viewer)
        splitter.addWidget(left_w)

        # Pravý scroll panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(420)
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setSpacing(8)
        rv.addWidget(self._build_ref_group())
        rv.addWidget(self._build_roi_group())
        rv.addWidget(self._build_edge_group())
        rv.addWidget(self._build_seg_group())
        rv.addWidget(self._build_calib_group())
        rv.addWidget(self._build_profile_group())
        rv.addStretch()
        scroll.setWidget(right_w)
        splitter.addWidget(scroll)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main = QVBoxLayout(self)
        main.setContentsMargins(4, 4, 4, 4)
        main.addWidget(splitter)

    # ---------- skupiny ----------

    def _build_ref_group(self) -> QGroupBox:
        grp = QGroupBox("Referenčný obraz")
        v = QVBoxLayout(grp)
        row = QHBoxLayout()
        btn_load = QPushButton("Načítaj…")
        btn_load.clicked.connect(self._on_load_reference)
        btn_del = QPushButton("Zmazať")
        btn_del.clicked.connect(self._on_delete_reference)
        row.addWidget(btn_load)
        row.addWidget(btn_del)
        v.addLayout(row)
        self._ref_path_label = QLabel("—")
        self._ref_path_label.setWordWrap(True)
        self._ref_path_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        v.addWidget(self._ref_path_label)
        return grp

    def _build_roi_group(self) -> QGroupBox:
        grp = QGroupBox("ROI (Region of Interest)")
        v = QVBoxLayout(grp)
        row = QHBoxLayout()
        self._draw_roi_btn = QPushButton("Kresliť ROI")
        self._draw_roi_btn.setCheckable(True)
        self._clear_roi_btn = QPushButton("Zmazať ROI")
        self._show_edges_btn = QPushButton("Zobraziť hrany")
        self._show_edges_btn.setCheckable(True)
        self._show_edges_btn.setChecked(True)
        row.addWidget(self._draw_roi_btn)
        row.addWidget(self._clear_roi_btn)
        v.addLayout(row)
        v.addWidget(self._show_edges_btn)
        form = QFormLayout()
        self._roi_x0 = QSpinBox(); self._roi_x0.setRange(0, 9999)
        self._roi_y0 = QSpinBox(); self._roi_y0.setRange(0, 9999)
        self._roi_x1 = QSpinBox(); self._roi_x1.setRange(0, 9999)
        self._roi_y1 = QSpinBox(); self._roi_y1.setRange(0, 9999)
        form.addRow("x0:", self._roi_x0); form.addRow("y0:", self._roi_y0)
        form.addRow("x1:", self._roi_x1); form.addRow("y1:", self._roi_y1)
        v.addLayout(form)
        return grp

    def _build_edge_group(self) -> QGroupBox:
        grp = QGroupBox("Detekcia hrán")
        v = QVBoxLayout(grp)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Metóda:"))
        self._edge_method_combo = QComboBox()
        self._edge_method_combo.addItems(EDGE_METHODS)
        method_row.addWidget(self._edge_method_combo, stretch=1)
        v.addLayout(method_row)

        self._edge_params_stack = QStackedWidget()

        # Canny
        canny_w = QWidget(); cl = QVBoxLayout(canny_w); cl.setContentsMargins(0,0,0,0)
        rw, _, self._canny_t1_spin = _make_int_row("Prah 1:", 0, 255, 50, 5, "Dolný prah")
        cl.addWidget(rw)
        rw, _, self._canny_t2_spin = _make_int_row("Prah 2:", 0, 255, 150, 5, "Horný prah")
        cl.addWidget(rw)
        blur_row = QHBoxLayout()
        blur_row.addWidget(QLabel("Blur:"))
        self._canny_blur_combo = QComboBox()
        for k in (1, 3, 5, 7, 9):
            self._canny_blur_combo.addItem("vypnutý" if k == 1 else f"{k}×{k}", k)
        self._canny_blur_combo.setCurrentIndex(1)
        blur_row.addWidget(self._canny_blur_combo)
        cl.addLayout(blur_row)
        self._edge_params_stack.addWidget(canny_w)

        # Scharr
        scharr_w = QWidget(); sl2 = QVBoxLayout(scharr_w); sl2.setContentsMargins(0,0,0,0)
        rw, _, self._scharr_thresh_spin = _make_int_row("Prah:", 0, 255, 30, 5, "Prah gradientu")
        sl2.addWidget(rw)
        s_blur_row = QHBoxLayout(); s_blur_row.addWidget(QLabel("Blur:"))
        self._scharr_blur_combo = QComboBox()
        for k in (1, 3, 5, 7, 9):
            self._scharr_blur_combo.addItem("vypnutý" if k == 1 else f"{k}×{k}", k)
        self._scharr_blur_combo.setCurrentIndex(1)
        s_blur_row.addWidget(self._scharr_blur_combo); sl2.addLayout(s_blur_row)
        self._edge_params_stack.addWidget(scharr_w)

        # LoG
        log_w = QWidget(); ll = QVBoxLayout(log_w); ll.setContentsMargins(0,0,0,0)
        rw, _, self._log_sigma_spin = _make_float_row("Sigma:", 0.1, 10.0, 1.5, 0.1, 2, "Sigma Gaussovho filtra")
        ll.addWidget(rw)
        rw, _, self._log_thresh_spin = _make_int_row("Prah:", 0, 255, 10, 5, "Prah LoG odozvy")
        ll.addWidget(rw)
        l_blur_row = QHBoxLayout(); l_blur_row.addWidget(QLabel("Blur:"))
        self._log_blur_combo = QComboBox()
        for k in (1, 3, 5, 7, 9):
            self._log_blur_combo.addItem("vypnutý" if k == 1 else f"{k}×{k}", k)
        l_blur_row.addWidget(self._log_blur_combo); ll.addLayout(l_blur_row)
        self._edge_params_stack.addWidget(log_w)

        # PhaseCongruency
        pc_w = QWidget(); pcl = QVBoxLayout(pc_w); pcl.setContentsMargins(0,0,0,0)
        rw, _, self._pc_nscale_spin = _make_int_row("Mierky:", 1, 8, 4, 1, "Počet mieriek")
        pcl.addWidget(rw)
        rw, _, self._pc_minwave_spin = _make_int_row("Min. vlnová dĺžka:", 2, 40, 6, 1, "Min. vlnová dĺžka [px]")
        pcl.addWidget(rw)
        rw, _, self._pc_mult_spin = _make_float_row("Multiplikátor:", 1.1, 4.0, 2.1, 0.1, 2)
        pcl.addWidget(rw)
        rw, _, self._pc_k_spin = _make_float_row("Prah k:", 0.1, 10.0, 2.0, 0.1, 2, "Citlivosť")
        pcl.addWidget(rw)
        self._edge_params_stack.addWidget(pc_w)

        # DexiNed
        dex_w = QWidget(); dxl = QVBoxLayout(dex_w); dxl.setContentsMargins(0,0,0,0)
        dex_form = QFormLayout()
        self._dexined_weights_edit = QLineEdit()
        self._dexined_weights_edit.setPlaceholderText("models/dexined.onnx (default)")
        wrow = QHBoxLayout()
        wrow.addWidget(self._dexined_weights_edit)
        wb = QPushButton("…"); wb.setFixedWidth(28)
        wb.clicked.connect(self._on_browse_dexined_weights)
        wrow.addWidget(wb)
        dex_form.addRow("Váhy:", wrow)
        rw2, _, self._dexined_thresh_spin = _make_float_row("Prah:", 0.01, 0.99, 0.5, 0.05, 2, "Sigmoid prah")
        dex_form.addRow(rw2)
        self._dexined_device_combo = QComboBox()
        self._dexined_device_combo.addItems(["cpu", "cuda"])
        dex_form.addRow("Zariadenie:", self._dexined_device_combo)
        dxl.addLayout(dex_form)
        dl_btn = QPushButton("Stiahnuť váhy")
        dl_btn.clicked.connect(self._on_download_dexined_weights)
        dxl.addWidget(dl_btn)
        self._dexined_status_label = QLabel("")
        self._dexined_status_label.setWordWrap(True)
        self._dexined_status_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        dxl.addWidget(self._dexined_status_label)
        self._edge_params_stack.addWidget(dex_w)

        v.addWidget(self._edge_params_stack)

        # Min segment length
        rw, _, self._min_seg_len_spin = _make_int_row(
            "Min. dĺžka segm. [px]:", 0, 9999, 0, 10,
            "Segmenty kratšie ako tento počet pixelov budú odfiltrované."
        )
        v.addWidget(rw)
        return grp

    def _build_seg_group(self) -> QGroupBox:
        grp = QGroupBox("Segmenty")
        v = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        self._remove_seg_btn = QPushButton("Odstrániť (klik)")
        self._remove_seg_btn.setCheckable(True)
        self._area_seg_btn = QPushButton("Odstrániť (oblasť)")
        self._area_seg_btn.setCheckable(True)
        row1.addWidget(self._remove_seg_btn); row1.addWidget(self._area_seg_btn)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        self._undo_seg_btn = QPushButton("Späť (Ctrl+Z)")
        self._undo_seg_btn.setEnabled(False)
        self._reset_seg_btn = QPushButton("Resetovať")
        self._reset_seg_btn.setEnabled(False)
        row2.addWidget(self._undo_seg_btn); row2.addWidget(self._reset_seg_btn)
        v.addLayout(row2)

        self._seg_count_label = QLabel("Odstránených: 0 segmentov")
        self._seg_count_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        v.addWidget(self._seg_count_label)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #555;"); v.addWidget(sep)

        sel_row = QHBoxLayout()
        self._select_seg_btn = QPushButton("Vybrať segment")
        self._select_seg_btn.setCheckable(True)
        self._clear_selection_btn = QPushButton("Zrušiť výber")
        self._clear_selection_btn.setEnabled(False)
        sel_row.addWidget(self._select_seg_btn); sel_row.addWidget(self._clear_selection_btn)
        v.addLayout(sel_row)

        self._seg_info_label = QLabel("Vybraný segment: —")
        self._seg_info_label.setStyleSheet("color: #ffaa44; font-size: 10px;")
        v.addWidget(self._seg_info_label)
        return grp

    def _build_calib_group(self) -> QGroupBox:
        grp = QGroupBox("Kalibrácia (mierka)")
        v = QVBoxLayout(grp)

        # Two-point calibration
        cal_row = QHBoxLayout()
        self._cal_btn = QPushButton("Označiť 2 body…")
        self._cal_btn.setCheckable(True)
        self._cal_btn.setToolTip(
            "Kliknite na dva body na obraze, potom zadajte vzdialenosť v mm."
        )
        cal_row.addWidget(self._cal_btn)
        self._cal_confirm_btn = QPushButton("✓ Potvrdiť body")
        self._cal_confirm_btn.setVisible(False)
        self._cal_confirm_btn.setStyleSheet("background-color: #1e5c1e; color: white;")
        cal_row.addWidget(self._cal_confirm_btn)
        v.addLayout(cal_row)

        form = QFormLayout()
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.0001, 9999.0)
        self._scale_spin.setDecimals(6)
        self._scale_spin.setValue(1.0)
        self._scale_spin.setSuffix(" mm/px")
        self._scale_spin.setToolTip("Počet milimetrov na pixel.")
        form.addRow("Mierka:", self._scale_spin)
        v.addLayout(form)

        self._cal_result_label = QLabel("Vzdialenosť: —")
        self._cal_result_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        v.addWidget(self._cal_result_label)
        return grp

    def _build_profile_group(self) -> QGroupBox:
        grp = QGroupBox("Algoritmus & Profil")
        v = QVBoxLayout(grp)

        # Algorithm
        algo_form = QFormLayout()
        self._algo_combo = QComboBox()
        self._algo_combo.addItems(["ECC", "POC"])
        algo_form.addRow("Algoritmus:", self._algo_combo)

        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 50000)
        self._iter_spin.setValue(2000)
        algo_form.addRow("Max. iter.:", self._iter_spin)

        self._epsilon_edit = QLineEdit("1e-8")
        self._epsilon_edit.setToolTip("Konvergencia ECC (napr. 1e-8)")
        algo_form.addRow("Epsilon:", self._epsilon_edit)

        self._gauss_filt_combo = QComboBox()
        for k, lbl in [(1, "vypnutý"), (3, "3"), (5, "5"), (7, "7")]:
            self._gauss_filt_combo.addItem(lbl, k)
        self._gauss_filt_combo.setCurrentIndex(1)
        algo_form.addRow("Gauss filter:", self._gauss_filt_combo)

        self._auto_clahe_check = QCheckBox("Auto CLAHE")
        algo_form.addRow("", self._auto_clahe_check)
        v.addLayout(algo_form)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); sep.setStyleSheet("color: #555;")
        v.addWidget(sep)

        # Profile name + ID
        self._id_label = QLabel("ID: —")
        self._id_label.setStyleSheet("color: #888; font-size: 10px;")
        v.addWidget(self._id_label)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Názov:"))
        self._name_edit = QLineEdit("unnamed")
        name_row.addWidget(self._name_edit, stretch=1)
        v.addLayout(name_row)

        self._save_btn = QPushButton("Uložiť profil")
        self._save_btn.setMinimumHeight(32)
        self._save_btn.setStyleSheet("font-weight: bold;")
        v.addWidget(self._save_btn)
        return grp

    # ------------------------------------------------------------------
    # Signálové prepojenia
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._viewer.roi_selected.connect(self._on_roi_selected)
        self._draw_roi_btn.toggled.connect(self._viewer.set_roi_mode)
        self._clear_roi_btn.clicked.connect(self._on_clear_roi)
        self._show_edges_btn.toggled.connect(self._on_show_edges_toggled)

        self._edge_method_combo.currentIndexChanged.connect(self._on_edge_method_changed)

        # Canny
        self._canny_t1_spin.valueChanged.connect(self._on_edge_changed)
        self._canny_t2_spin.valueChanged.connect(self._on_edge_changed)
        self._canny_blur_combo.currentIndexChanged.connect(self._on_edge_changed)
        # Scharr
        self._scharr_thresh_spin.valueChanged.connect(self._on_edge_changed)
        self._scharr_blur_combo.currentIndexChanged.connect(self._on_edge_changed)
        # LoG
        self._log_sigma_spin.valueChanged.connect(self._on_edge_changed)
        self._log_thresh_spin.valueChanged.connect(self._on_edge_changed)
        self._log_blur_combo.currentIndexChanged.connect(self._on_edge_changed)
        # PC
        self._pc_nscale_spin.valueChanged.connect(self._on_edge_changed)
        self._pc_minwave_spin.valueChanged.connect(self._on_edge_changed)
        self._pc_mult_spin.valueChanged.connect(self._on_edge_changed)
        self._pc_k_spin.valueChanged.connect(self._on_edge_changed)
        # DexiNed
        self._dexined_thresh_spin.valueChanged.connect(self._on_edge_changed)
        self._dexined_device_combo.currentTextChanged.connect(self._on_edge_changed)
        self._dexined_weights_edit.editingFinished.connect(self._on_edge_changed)
        # Min len
        self._min_seg_len_spin.valueChanged.connect(self._on_edge_changed)

        # ROI spinboxy
        for sp in (self._roi_x0, self._roi_y0, self._roi_x1, self._roi_y1):
            sp.valueChanged.connect(self._on_roi_spinbox_changed)

        # Segmenty
        self._remove_seg_btn.toggled.connect(self._on_remove_seg_mode_toggled)
        self._area_seg_btn.toggled.connect(self._on_area_seg_mode_toggled)
        self._undo_seg_btn.clicked.connect(self._on_undo_segment)
        self._reset_seg_btn.clicked.connect(self._on_reset_segments)
        self._select_seg_btn.toggled.connect(self._on_select_seg_mode_toggled)
        self._clear_selection_btn.clicked.connect(self._on_clear_selection)
        self._viewer.image_clicked.connect(self._on_viewer_clicked)
        self._viewer.segment_area_selected.connect(self._on_segment_area_selected)

        # Kalibrácia
        self._cal_btn.toggled.connect(self._on_cal_btn_toggled)
        self._viewer.calibration_points_selected.connect(self._on_calibration_points)
        self._viewer.calibration_ready.connect(self._on_calibration_ready)
        self._cal_confirm_btn.clicked.connect(self._on_confirm_calibration)

        # Uloženie
        self._save_btn.clicked.connect(self._on_save_profile)

        # Klávesová skratka
        undo_sc = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_sc.activated.connect(self._on_undo_segment)

    # ------------------------------------------------------------------
    # Handlery — obraz
    # ------------------------------------------------------------------

    def _on_load_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Vyber referenčný obraz", "",
            "Obrázky (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if path:
            self._ref_path_label.setText(path)
            self._load_ref_from_path(path)

    def _load_ref_from_path(self, path: str) -> None:
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Chyba", f"Nemôžem načítať obraz:\n{path}")
            return
        self._ref_image = img
        self._viewer.set_image(img)
        if self._roi is not None:
            self._viewer.draw_roi(self._roi)
            self._update_roi_spinboxes()
        self._update_ref_edges()

    def _on_delete_reference(self) -> None:
        self._ref_image = None
        self._ref_edges = None
        self._segment_labels = None
        self._removed_labels.clear()
        self._undo_stack.clear()
        self._ref_path_label.setText("—")
        self._viewer.set_image(np.zeros((400, 600), dtype=np.uint8))
        self._update_seg_ui()

    # ------------------------------------------------------------------
    # Handlery — ROI
    # ------------------------------------------------------------------

    def _on_roi_selected(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._draw_roi_btn.setChecked(False)
        try:
            roi = ROI(x0, y0, x1, y1)
            if not roi.is_valid():
                return
            self._roi = roi
            self._update_roi_spinboxes()
            self._viewer.draw_roi(roi)
            self._update_ref_edges()
        except ValueError:
            pass

    def _on_clear_roi(self) -> None:
        self._roi = None
        self._ref_edges = None
        self._segment_labels = None
        self._removed_labels.clear()
        self._undo_stack.clear()
        self._update_seg_ui()
        self._viewer.clear_roi()
        self._viewer.clear_overlay()
        for sp in (self._roi_x0, self._roi_y0, self._roi_x1, self._roi_y1):
            sp.blockSignals(True); sp.setValue(0); sp.blockSignals(False)

    def _on_roi_spinbox_changed(self) -> None:
        x0, y0 = self._roi_x0.value(), self._roi_y0.value()
        x1, y1 = self._roi_x1.value(), self._roi_y1.value()
        if x1 > x0 and y1 > y0:
            try:
                self._roi = ROI(x0, y0, x1, y1)
                self._viewer.draw_roi(self._roi)
                self._update_ref_edges()
            except ValueError:
                pass

    def _update_roi_spinboxes(self) -> None:
        if self._roi is None:
            return
        for sp, val in zip(
            (self._roi_x0, self._roi_y0, self._roi_x1, self._roi_y1),
            (self._roi.x0, self._roi.y0, self._roi.x1, self._roi.y1),
        ):
            sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)

    # ------------------------------------------------------------------
    # Handlery — hrany
    # ------------------------------------------------------------------

    def _on_show_edges_toggled(self, show: bool) -> None:
        self._show_edges = show
        if show:
            self._viewer.draw_edges(self._active_ref_edges)
        else:
            self._viewer.clear_overlay()
            if self._roi is not None:
                self._viewer.draw_roi(self._roi)

    def _on_edge_method_changed(self, idx: int) -> None:
        self._edge_params_stack.setCurrentIndex(idx)
        self._on_edge_changed()

    def _on_edge_changed(self) -> None:
        self._update_ref_edges()

    def _get_edge_params(self) -> dict:
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
                "threshold": self._dexined_thresh_spin.value(),
                "device": self._dexined_device_combo.currentText(),
            }
        return {}

    def _update_ref_edges(self) -> None:
        if self._ref_image is None:
            return
        gray = (cv2.cvtColor(self._ref_image, cv2.COLOR_BGR2GRAY)
                if self._ref_image.ndim == 3 else self._ref_image.copy())
        method = self._edge_method_combo.currentText()
        params = self._get_edge_params()

        if method == "DexiNed":
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

        if self._roi is not None:
            h, w = gray.shape
            x0c = max(0, self._roi.x0); y0c = max(0, self._roi.y0)
            x1c = min(w, self._roi.x1); y1c = min(h, self._roi.y1)
            if x1c <= x0c or y1c <= y0c:
                return
            edges_full = np.zeros(gray.shape, dtype=np.uint8)
            edges_full[y0c:y1c, x0c:x1c] = detect_edges(
                gray[y0c:y1c, x0c:x1c], method, **params
            )
        else:
            edges_full = detect_edges(gray, method, **params)

        _, lbl_full = cv2.connectedComponents(edges_full, connectivity=8)
        edges_full = self._apply_min_length_filter(edges_full, lbl_full)

        self._ref_edges = edges_full
        self._removed_labels.clear()
        self._erased_mask = np.ones_like(edges_full, dtype=np.uint8) * 255
        self._undo_stack.clear()
        self._segment_labels = self._compute_segment_labels(edges_full)
        self._update_seg_ui()
        if self._show_edges:
            self._viewer.draw_edges(edges_full)
            if self._roi is not None:
                self._viewer.draw_roi(self._roi)

    @pyqtSlot(object)
    def _on_edge_result(self, edges: np.ndarray) -> None:
        _, lbl = cv2.connectedComponents(edges, connectivity=8)
        edges = self._apply_min_length_filter(edges, lbl)
        self._ref_edges = edges
        self._removed_labels.clear()
        self._erased_mask = np.ones_like(edges, dtype=np.uint8) * 255
        self._undo_stack.clear()
        self._segment_labels = self._compute_segment_labels(edges)
        self._update_seg_ui()
        if self._show_edges:
            self._viewer.draw_edges(edges)
            if self._roi is not None:
                self._viewer.draw_roi(self._roi)
        self._dexined_status_label.setText("Hrany vypočítané ✓")
        self._dexined_status_label.setStyleSheet("color: #44ff88; font-size: 10px;")

    @pyqtSlot(str)
    def _on_edge_error(self, msg: str) -> None:
        self._dexined_status_label.setText(f"Chyba: {msg}")
        self._dexined_status_label.setStyleSheet("color: #ff6666; font-size: 10px;")

    def _on_browse_dexined_weights(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Vyber model DexiNed", "",
            "ONNX model (*.onnx);;PyTorch checkpoint (*.pth *.pt)"
        )
        if path:
            self._dexined_weights_edit.setText(path)
            self._on_edge_changed()

    def _on_download_dexined_weights(self) -> None:
        import subprocess, sys
        script = str(Path(__file__).parent.parent.parent / "models" / "download_dexined.py")
        self._dexined_status_label.setText("Sťahujem váhy…")
        self._dexined_status_label.setStyleSheet("color: #ffcc00; font-size: 10px;")
        try:
            res = subprocess.run([sys.executable, script],
                                 capture_output=True, text=True, timeout=120)
            if res.returncode == 0:
                self._dexined_status_label.setText("Váhy stiahnuté ✓")
                self._dexined_status_label.setStyleSheet("color: #44ff88; font-size: 10px;")
            else:
                self._dexined_status_label.setText(f"Chyba: {res.stderr[:120]}")
                self._dexined_status_label.setStyleSheet("color: #ff6666; font-size: 10px;")
        except Exception as e:
            self._dexined_status_label.setText(f"Chyba: {e}")
            self._dexined_status_label.setStyleSheet("color: #ff6666; font-size: 10px;")

    # ------------------------------------------------------------------
    # Handlery — segmenty
    # ------------------------------------------------------------------

    @property
    def _active_ref_edges(self) -> np.ndarray | None:
        if self._ref_edges is None:
            return None
        mask = self._ref_edges.copy()
        if self._removed_labels and self._segment_labels is not None:
            for lbl in self._removed_labels:
                mask[self._segment_labels == lbl] = 0
        if self._erased_mask is not None:
            mask &= self._erased_mask
        return mask

    def _compute_segment_labels(self, edges: np.ndarray) -> np.ndarray:
        _, labels = cv2.connectedComponents(edges, connectivity=8)
        return labels

    def _apply_min_length_filter(self, edges: np.ndarray, labels: np.ndarray) -> np.ndarray:
        min_len = self._min_seg_len_spin.value()
        if min_len <= 0:
            return edges
        result = edges.copy()
        for lbl in range(1, int(labels.max()) + 1):
            if np.count_nonzero(labels == lbl) < min_len:
                result[labels == lbl] = 0
        return result

    def _update_seg_ui(self) -> None:
        n = len(self._removed_labels)
        n_px = 0
        if self._erased_mask is not None and self._ref_edges is not None:
            n_px = int(np.count_nonzero(self._ref_edges & (~self._erased_mask)))
        parts = []
        if n: parts.append(f"{n} segm.")
        if n_px: parts.append(f"{n_px} px")
        self._seg_count_label.setText("Odstránených: " + (", ".join(parts) or "0"))
        self._reset_seg_btn.setEnabled(n > 0 or n_px > 0)
        self._undo_seg_btn.setEnabled(len(self._undo_stack) > 0)

    def _push_undo(self) -> None:
        em = self._erased_mask.copy() if self._erased_mask is not None else None
        self._undo_stack.append((frozenset(self._removed_labels), em))

    def _on_remove_seg_mode_toggled(self, enabled: bool) -> None:
        if enabled:
            self._draw_roi_btn.setChecked(False)
            self._area_seg_btn.setChecked(False)
            self._select_seg_btn.setChecked(False)
        self._viewer.set_mode(ViewerMode.CLICK if enabled else ViewerMode.NONE)

    def _on_area_seg_mode_toggled(self, enabled: bool) -> None:
        if enabled:
            self._draw_roi_btn.setChecked(False)
            self._remove_seg_btn.setChecked(False)
            self._select_seg_btn.setChecked(False)
        self._viewer.set_mode(ViewerMode.SEGMENT_AREA if enabled else ViewerMode.NONE)

    def _on_select_seg_mode_toggled(self, enabled: bool) -> None:
        if enabled:
            self._draw_roi_btn.setChecked(False)
            self._remove_seg_btn.setChecked(False)
            self._area_seg_btn.setChecked(False)
        self._viewer.set_mode(ViewerMode.CLICK if enabled else ViewerMode.NONE)

    def _on_clear_selection(self) -> None:
        self._selected_label = None
        self._segment_centroid_ref = None
        self._seg_info_label.setText("Vybraný segment: —")
        self._clear_selection_btn.setEnabled(False)
        if self._show_edges:
            self._viewer.draw_edges(self._active_ref_edges)
            if self._roi is not None:
                self._viewer.draw_roi(self._roi)

    def _on_viewer_clicked(self, x: int, y: int) -> None:
        if self._segment_labels is None or self._ref_edges is None:
            return
        h, w = self._segment_labels.shape
        if not (0 <= y < h and 0 <= x < w):
            return
        label = int(self._segment_labels[y, x])
        if label == 0:
            return

        if self._select_seg_btn.isChecked():
            self._selected_label = label
            seg_mask = (self._segment_labels == label).astype(np.uint8) * 255
            filled = binary_fill_holes(seg_mask > 0).astype(np.uint8) * 255
            mom = cv2.moments(filled)
            if mom["m00"] > 0:
                self._segment_centroid_ref = (
                    mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
                )
            else:
                self._segment_centroid_ref = None
            self._seg_info_label.setText(
                f"Segment #{label}  "
                + (f"ťažisko: ({self._segment_centroid_ref[0]:.1f}, "
                   f"{self._segment_centroid_ref[1]:.1f})"
                   if self._segment_centroid_ref else "")
            )
            self._clear_selection_btn.setEnabled(True)
            self._viewer.draw_edges_with_selection(
                self._active_ref_edges, self._segment_labels, label
            )
            if self._segment_centroid_ref is not None:
                self._viewer.draw_centroid_marker(*self._segment_centroid_ref)
        else:
            # Remove mode
            self._push_undo()
            self._removed_labels.add(label)
            self._update_seg_ui()
            self._viewer.draw_edges(self._active_ref_edges)

    @pyqtSlot(int, int, int, int)
    def _on_segment_area_selected(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self._ref_edges is None or self._erased_mask is None:
            return
        h, w = self._ref_edges.shape
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(w, x1), min(h, y1)
        if x1c <= x0c or y1c <= y0c:
            return
        active_in_region = self._ref_edges[y0c:y1c, x0c:x1c] & self._erased_mask[y0c:y1c, x0c:x1c]
        if not np.any(active_in_region):
            return
        self._push_undo()
        self._erased_mask[y0c:y1c, x0c:x1c] = 0
        active = self._active_ref_edges
        self._segment_labels = self._compute_segment_labels(active)
        self._update_seg_ui()
        self._viewer.draw_edges(active)

    def _on_undo_segment(self) -> None:
        if not self._undo_stack:
            return
        prev_labels, prev_mask = self._undo_stack.pop()
        self._removed_labels = set(prev_labels)
        self._erased_mask = prev_mask
        active = self._active_ref_edges
        if active is not None:
            self._segment_labels = self._compute_segment_labels(active)
        self._update_seg_ui()
        self._viewer.draw_edges(active)

    def _on_reset_segments(self) -> None:
        self._removed_labels.clear()
        if self._ref_edges is not None:
            self._erased_mask = np.ones_like(self._ref_edges, dtype=np.uint8) * 255
            self._segment_labels = self._compute_segment_labels(self._ref_edges)
        self._undo_stack.clear()
        self._update_seg_ui()
        self._viewer.draw_edges(self._active_ref_edges)

    # ------------------------------------------------------------------
    # Kalibrácia
    # ------------------------------------------------------------------

    def _on_cal_btn_toggled(self, enabled: bool) -> None:
        if enabled:
            self._draw_roi_btn.setChecked(False)
            self._remove_seg_btn.setChecked(False)
            self._area_seg_btn.setChecked(False)
            self._select_seg_btn.setChecked(False)
        else:
            self._cal_confirm_btn.setVisible(False)
        self._viewer.set_mode(ViewerMode.CALIBRATION if enabled else ViewerMode.NONE)

    def _on_calibration_ready(self) -> None:
        self._cal_confirm_btn.setVisible(True)

    def _on_confirm_calibration(self) -> None:
        self._cal_confirm_btn.setVisible(False)
        self._viewer.confirm_calibration()

    @pyqtSlot(float, float, float, float)
    def _on_calibration_points(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Prijme dva body z viewera, opýta sa na mm vzdialenosť, prepočíta mierku."""
        self._cal_confirm_btn.setVisible(False)
        self._cal_btn.setChecked(False)
        pixel_dist = math.hypot(x2 - x1, y2 - y1)
        if pixel_dist < 1.0:
            QMessageBox.warning(self, "Kalibrácia", "Body sú príliš blízko seba.")
            return
        mm_dist, ok = QInputDialog.getDouble(
            self, "Kalibrácia",
            f"Vzdialenosť medzi bodmi v mm\n(pixelová vzdialenosť: {pixel_dist:.1f} px):",
            decimals=4, min=0.0001, max=9999.0,
        )
        if not ok:
            return
        mm_per_px = mm_dist / pixel_dist
        self._scale_spin.setValue(mm_per_px)
        self._cal_result_label.setText(
            f"Vzdialenosť: {pixel_dist:.1f} px = {mm_dist:.4f} mm\n"
            f"Mierka: {mm_per_px:.6f} mm/px"
        )

    # ------------------------------------------------------------------
    # Profil
    # ------------------------------------------------------------------

    def _on_save_profile(self) -> None:
        profile = self.get_profile()
        errors = profile.validate()
        if errors:
            QMessageBox.warning(self, "Neplatný profil", "\n".join(errors))
            return
        try:
            self._config_mgr.save_profile(profile)
            self._active_profile_id = profile.id
            self._id_label.setText(f"ID: {profile.id}")
            self.profile_saved.emit(profile)
            QMessageBox.information(
                self, "OK",
                f"Profil '{profile.name}' uložený (ID {profile.id})."
            )
        except Exception as e:
            QMessageBox.critical(self, "Chyba", str(e))
