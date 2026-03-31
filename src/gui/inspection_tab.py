"""InspectionTab — Tab 2: vyhodnocovanie inšpekčného obrazu voči referenčnému profilu."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QDoubleSpinBox,
    QSpinBox, QComboBox, QScrollArea, QFileDialog, QMessageBox,
    QCheckBox, QSizePolicy, QFrame,
)

from src.core.roi import ROI
from src.core.preprocessor import preprocess
from src.core.aligner import align
from src.core.calibration import Calibration
from src.core.result import AlignResult
from src.core.edge_detector import detect_edges
from src.config.profile import Profile
from src.config.config_manager import ConfigManager
from src.gui.image_viewer import ImageViewer, ViewerMode
from src.gui.profile_editor_tab import _make_int_row


class InspectionTab(QWidget):
    """Tab 2 — inšpekcia: referenčný viewer + inšpekčný viewer + kontroly."""

    def __init__(self, config_mgr: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config_mgr = config_mgr
        self._active_profile: Profile | None = None
        self._ref_image: np.ndarray | None = None
        self._ref_edges: np.ndarray | None = None
        self._insp_image: np.ndarray | None = None
        self._insp_roi: ROI | None = None
        self._insp_edges: np.ndarray | None = None
        self._show_insp_edges: bool = False

        self._build_ui()
        self._connect_signals()
        self._refresh_profile_combo()

    # ------------------------------------------------------------------
    # Verejné
    # ------------------------------------------------------------------

    def refresh_profiles(self, _profile=None) -> None:
        """Obnoví zoznam profilov (volaný pri uložení nového profilu z Tab 1)."""
        self._refresh_profile_combo()

    def set_profile_by_name(self, name: str) -> None:
        """Načíta profil podľa mena a zobrazí ho."""
        idx = self._profile_combo.findText(name)
        if idx >= 0:
            self._profile_combo.setCurrentIndex(idx)
        self._on_load_profile()

    # ------------------------------------------------------------------
    # Budovanie UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # Top bar — výber profilu
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Profil:"))
        self._profile_combo = QComboBox()
        self._profile_combo.setMinimumWidth(200)
        top_bar.addWidget(self._profile_combo, stretch=1)
        self._load_profile_btn = QPushButton("Načítať profil")
        top_bar.addWidget(self._load_profile_btn)
        self._active_profile_label = QLabel("—")
        self._active_profile_label.setStyleSheet("color: #888; font-size: 10px;")
        top_bar.addWidget(self._active_profile_label)
        top_bar.addStretch()
        outer.addLayout(top_bar)

        # Splitter: ref viewer | insp viewer | controls
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Referenčný viewer (read-only)
        self._ref_viewer = ImageViewer()
        self._ref_viewer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        ref_lbl = QLabel("Referenčný obraz (z profilu)")
        ref_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ref_lbl.setStyleSheet("font-weight: bold; padding: 2px;")
        ref_cont = QWidget()
        rl = QVBoxLayout(ref_cont); rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(ref_lbl); rl.addWidget(self._ref_viewer)
        splitter.addWidget(ref_cont)

        # Inšpekčný viewer
        self._insp_viewer = ImageViewer()
        self._insp_viewer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        insp_lbl = QLabel("Inšpekčný obraz")
        insp_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        insp_lbl.setStyleSheet("font-weight: bold; padding: 2px;")
        insp_cont = QWidget()
        il = QVBoxLayout(insp_cont); il.setContentsMargins(0, 0, 0, 0)
        il.addWidget(insp_lbl); il.addWidget(self._insp_viewer)
        splitter.addWidget(insp_cont)

        # Pravý panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(280)
        right_w = QWidget()
        rv = QVBoxLayout(right_w); rv.setSpacing(8)
        rv.addWidget(self._build_insp_group())
        rv.addWidget(self._build_insp_roi_group())
        rv.addWidget(self._build_align_group())

        # Show/hide edges toggle
        self._show_edges_btn = QPushButton("Zobraziť hrany v ROI")
        self._show_edges_btn.setCheckable(True)
        self._show_edges_btn.setToolTip(
            "Zobrazí hrany v inšpekčnom ROI — parametre z profilu."
        )
        rv.addWidget(self._show_edges_btn)

        self._run_btn = QPushButton("▶  Spustiť vyhľadávanie")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setStyleSheet("font-weight: bold; font-size: 13px;")
        rv.addWidget(self._run_btn)

        rv.addWidget(self._build_result_group())
        rv.addStretch()
        scroll.setWidget(right_w)
        splitter.addWidget(scroll)

        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([500, 500, 280])
        outer.addWidget(splitter)

    def _build_insp_group(self) -> QGroupBox:
        grp = QGroupBox("Inšpekčný obraz")
        v = QVBoxLayout(grp)
        row = QHBoxLayout()
        btn_load = QPushButton("Načítaj…")
        btn_load.clicked.connect(self._on_load_inspection)
        btn_del = QPushButton("Zmazať")
        btn_del.clicked.connect(self._on_delete_inspection)
        row.addWidget(btn_load); row.addWidget(btn_del)
        v.addLayout(row)
        self._insp_path_label = QLabel("—")
        self._insp_path_label.setWordWrap(True)
        self._insp_path_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        v.addWidget(self._insp_path_label)
        return grp

    def _build_insp_roi_group(self) -> QGroupBox:
        grp = QGroupBox("Inšpekčné ROI (oblasť vyhľadávania)")
        v = QVBoxLayout(grp)
        row = QHBoxLayout()
        self._draw_insp_roi_btn = QPushButton("Kresliť ROI")
        self._draw_insp_roi_btn.setCheckable(True)
        self._clear_insp_roi_btn = QPushButton("Zmazať ROI")
        row.addWidget(self._draw_insp_roi_btn); row.addWidget(self._clear_insp_roi_btn)
        v.addLayout(row)
        form = QFormLayout()
        self._insp_roi_x0 = QSpinBox(); self._insp_roi_x0.setRange(0, 9999)
        self._insp_roi_y0 = QSpinBox(); self._insp_roi_y0.setRange(0, 9999)
        self._insp_roi_x1 = QSpinBox(); self._insp_roi_x1.setRange(0, 9999)
        self._insp_roi_y1 = QSpinBox(); self._insp_roi_y1.setRange(0, 9999)
        form.addRow("x0:", self._insp_roi_x0); form.addRow("y0:", self._insp_roi_y0)
        form.addRow("x1:", self._insp_roi_x1); form.addRow("y1:", self._insp_roi_y1)
        v.addLayout(form)
        save_roi_btn = QPushButton("Uložiť ROI do profilu")
        save_roi_btn.setToolTip("Uloží inšpekčné ROI do aktívneho profilu na disk.")
        save_roi_btn.clicked.connect(self._on_save_insp_roi_to_profile)
        v.addWidget(save_roi_btn)
        return grp

    def _build_align_group(self) -> QGroupBox:
        grp = QGroupBox("Parametre vyhľadávania")
        v = QVBoxLayout(grp)
        form = QFormLayout()
        self._algo_combo = QComboBox()
        self._algo_combo.addItems(["ECC", "POC"])
        form.addRow("Algoritmus:", self._algo_combo)

        rw, _, self._iter_spin = _make_int_row("Max. iter.:", 100, 50000, 2000, 100)
        form.addRow(rw)

        self._epsilon_edit = QLineEdit("1e-08")
        form.addRow("Epsilon:", self._epsilon_edit)

        self._gauss_filt_combo = QComboBox()
        for val, lbl in ((1, "1 — vypnutý"), (3, "3×3"), (5, "5×5"), (7, "7×7")):
            self._gauss_filt_combo.addItem(lbl, val)
        self._gauss_filt_combo.setCurrentIndex(1)
        form.addRow("Gauss filter:", self._gauss_filt_combo)

        self._auto_clahe_check = QCheckBox("Auto CLAHE")
        form.addRow("", self._auto_clahe_check)
        v.addLayout(form)

        save_params_btn = QPushButton("Uložiť parametre do profilu")
        save_params_btn.setToolTip("Uloží aktuálne parametre zarovnania do profilu na disk.")
        save_params_btn.clicked.connect(self._on_save_params_to_profile)
        v.addWidget(save_params_btn)
        return grp

    def _build_result_group(self) -> QGroupBox:
        grp = QGroupBox("Výsledky")
        form = QFormLayout(grp)
        self._res_dx_px   = QLabel("—")
        self._res_dy_px   = QLabel("—")
        self._res_dx_mm   = QLabel("—")
        self._res_dy_mm   = QLabel("—")
        self._res_angle   = QLabel("—")
        self._res_conf    = QLabel("—")
        self._res_ncc     = QLabel("—")
        self._res_time    = QLabel("—")
        self._res_centroid_pos = QLabel("—")
        self._res_dx_c_px = QLabel("—")
        self._res_dy_c_px = QLabel("—")
        self._res_dx_c_mm = QLabel("—")
        self._res_dy_c_mm = QLabel("—")
        for lbl in (self._res_dx_px, self._res_dy_px, self._res_dx_mm,
                    self._res_dy_mm, self._res_angle, self._res_conf, self._res_ncc,
                    self._res_time, self._res_centroid_pos, self._res_dx_c_px,
                    self._res_dy_c_px, self._res_dx_c_mm, self._res_dy_c_mm):
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("dx [px]:", self._res_dx_px)
        form.addRow("dy [px]:", self._res_dy_px)
        form.addRow("dx [mm]:", self._res_dx_mm)
        form.addRow("dy [mm]:", self._res_dy_mm)
        form.addRow("Uhol [°]:", self._res_angle)
        form.addRow("Spoľahlivosť:", self._res_conf)
        form.addRow("NCC:", self._res_ncc)
        form.addRow("Čas [s]:", self._res_time)
        sep = QLabel("── Ťažisko segmentu ──")
        sep.setStyleSheet("color: #888; font-size: 9px;")
        form.addRow(sep)
        form.addRow("Poloha ref [px]:", self._res_centroid_pos)
        form.addRow("dx_c [px]:", self._res_dx_c_px)
        form.addRow("dy_c [px]:", self._res_dy_c_px)
        form.addRow("dx_c [mm]:", self._res_dx_c_mm)
        form.addRow("dy_c [mm]:", self._res_dy_c_mm)
        return grp

    # ------------------------------------------------------------------
    # Signálové prepojenia
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._load_profile_btn.clicked.connect(self._on_load_profile)
        self._draw_insp_roi_btn.toggled.connect(self._insp_viewer.set_roi_mode)
        self._clear_insp_roi_btn.clicked.connect(self._on_clear_insp_roi)
        self._insp_viewer.roi_selected.connect(self._on_insp_roi_selected)
        for sp in (self._insp_roi_x0, self._insp_roi_y0,
                   self._insp_roi_x1, self._insp_roi_y1):
            sp.valueChanged.connect(self._on_insp_roi_spinbox_changed)
        self._show_edges_btn.toggled.connect(self._on_show_insp_edges_toggled)
        self._run_btn.clicked.connect(self._on_run_alignment)

    # ------------------------------------------------------------------
    # Handlery — profil
    # ------------------------------------------------------------------

    def _refresh_profile_combo(self) -> None:
        current = self._profile_combo.currentText()
        profiles = self._config_mgr.list_profiles_full()
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        for p in profiles:
            self._profile_combo.addItem(
                f"[{p['id']}] {p['name']}" if p["id"] else p["name"],
                userData=p["name"],
            )
        # Obnov výber ak možné
        for i in range(self._profile_combo.count()):
            if self._profile_combo.itemData(i) == current:
                self._profile_combo.setCurrentIndex(i)
                break
        self._profile_combo.blockSignals(False)

    def _on_load_profile(self) -> None:
        name = self._profile_combo.currentData()
        if not name:
            return
        try:
            profile = self._config_mgr.load_profile(name)
        except Exception as e:
            QMessageBox.critical(self, "Chyba", str(e))
            return
        self._active_profile = profile
        self._active_profile_label.setText(
            f"ID: {profile.id}  |  {profile.name}  |  mierka: {profile.scale_mm_per_px:.4f} mm/px"
        )
        # Nastav parametre zarovnania z profilu
        idx = self._algo_combo.findText(profile.algorithm)
        if idx >= 0:
            self._algo_combo.setCurrentIndex(idx)
        self._iter_spin.setValue(profile.ecc_max_iter)
        self._epsilon_edit.setText(str(profile.ecc_epsilon))
        gfs_idx = self._gauss_filt_combo.findData(profile.ecc_gauss_filt_size)
        if gfs_idx >= 0:
            self._gauss_filt_combo.setCurrentIndex(gfs_idx)
        self._auto_clahe_check.setChecked(profile.auto_clahe)

        # Načítaj referenčný obraz
        ref_path = profile.reference_path
        if ref_path and Path(ref_path).exists():
            img = cv2.imread(ref_path)
            if img is not None:
                self._ref_image = img
                self._ref_viewer.set_image(img)
                # Zobraz segmenty/hrany podľa profilu
                self._ref_edges = self._compute_ref_edges(profile)
                if self._ref_edges is not None:
                    self._ref_viewer.draw_edges(self._ref_edges)
                    if profile.roi is not None:
                        self._ref_viewer.draw_roi(profile.roi)
                    if profile.selected_segment_centroid is not None:
                        cx, cy = profile.selected_segment_centroid
                        self._ref_viewer.draw_centroid_marker(cx, cy)
        else:
            self._ref_image = None
            self._ref_edges = None

        # Obnov inšpekčné ROI z profilu (ak je uložené)
        if profile.insp_roi is not None:
            self._insp_roi = profile.insp_roi
            self._update_insp_roi_spinboxes()
            self._insp_viewer.draw_roi(profile.insp_roi)
        else:
            self._insp_roi = None

    def _compute_ref_edges(self, profile: Profile) -> np.ndarray | None:
        """Vypočíta hrany referenčného obrazu podľa parametrov profilu."""
        if self._ref_image is None:
            return None
        gray = (cv2.cvtColor(self._ref_image, cv2.COLOR_BGR2GRAY)
                if self._ref_image.ndim == 3 else self._ref_image.copy())
        params = self._get_edge_params_from_profile(profile)
        method = profile.edge_method
        roi = profile.roi
        if roi is not None:
            h, w = gray.shape
            x0c = max(0, roi.x0); y0c = max(0, roi.y0)
            x1c = min(w, roi.x1); y1c = min(h, roi.y1)
            if x1c <= x0c or y1c <= y0c:
                return None
            out = np.zeros(gray.shape, dtype=np.uint8)
            try:
                out[y0c:y1c, x0c:x1c] = detect_edges(
                    gray[y0c:y1c, x0c:x1c], method, **params
                )
            except Exception:
                return None
        else:
            try:
                out = detect_edges(gray, method, **params)
            except Exception:
                return None
        # Min-length filter
        min_len = profile.min_seg_len
        if min_len > 0:
            _, lbl = cv2.connectedComponents(out, connectivity=8)
            result = out.copy()
            for l in range(1, int(lbl.max()) + 1):
                if np.count_nonzero(lbl == l) < min_len:
                    result[lbl == l] = 0
            out = result
        return out

    @staticmethod
    def _get_edge_params_from_profile(profile: Profile) -> dict:
        m = profile.edge_method
        if m == "Canny":
            return {"t1": profile.canny_threshold1, "t2": profile.canny_threshold2,
                    "blur": profile.canny_blur}
        elif m == "Scharr":
            return {"threshold": profile.scharr_threshold, "blur": profile.scharr_blur}
        elif m == "LoG":
            return {"sigma": profile.log_sigma, "threshold": profile.log_threshold,
                    "blur": profile.log_blur}
        elif m == "PhaseCongruency":
            return {"nscale": profile.pc_nscale, "min_wavelength": profile.pc_min_wavelength,
                    "mult": profile.pc_mult, "k": profile.pc_k}
        elif m == "DexiNed":
            return {"weights_path": profile.dexined_weights,
                    "threshold": profile.dexined_threshold,
                    "device": profile.dexined_device}
        return {}

    # ------------------------------------------------------------------
    # Handlery — inšpekčný obraz
    # ------------------------------------------------------------------

    def _on_load_inspection(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Vyber inšpekčný obraz", "",
            "Obrázky (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Chyba", f"Nemôžem načítať obraz:\n{path}")
            return
        self._insp_image = img
        self._insp_viewer.set_image(img)
        self._insp_path_label.setText(path)
        if self._insp_roi is not None:
            self._insp_viewer.draw_roi(self._insp_roi)
        if self._show_insp_edges:
            self._update_insp_edges()

    def _on_delete_inspection(self) -> None:
        self._insp_image = None
        self._insp_edges = None
        self._insp_path_label.setText("—")
        self._insp_viewer.set_image(np.zeros((400, 600), dtype=np.uint8))

    # ------------------------------------------------------------------
    # Handlery — inšpekčné ROI
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
        except ValueError:
            pass

    def _on_clear_insp_roi(self) -> None:
        self._insp_roi = None
        self._insp_viewer.clear_roi()
        self._insp_edges = None
        for sp in (self._insp_roi_x0, self._insp_roi_y0,
                   self._insp_roi_x1, self._insp_roi_y1):
            sp.blockSignals(True); sp.setValue(0); sp.blockSignals(False)

    def _on_insp_roi_spinbox_changed(self) -> None:
        x0, y0 = self._insp_roi_x0.value(), self._insp_roi_y0.value()
        x1, y1 = self._insp_roi_x1.value(), self._insp_roi_y1.value()
        if x1 > x0 and y1 > y0:
            try:
                self._insp_roi = ROI(x0, y0, x1, y1)
                self._insp_viewer.draw_roi(self._insp_roi)
            except ValueError:
                pass

    def _update_insp_roi_spinboxes(self) -> None:
        if self._insp_roi is None:
            return
        for sp, val in zip(
            (self._insp_roi_x0, self._insp_roi_y0,
             self._insp_roi_x1, self._insp_roi_y1),
            (self._insp_roi.x0, self._insp_roi.y0,
             self._insp_roi.x1, self._insp_roi.y1),
        ):
            sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)

    def _on_save_insp_roi_to_profile(self) -> None:
        if self._active_profile is None:
            QMessageBox.warning(self, "Chyba", "Najprv načítaj profil.")
            return
        self._active_profile.insp_roi = self._insp_roi
        try:
            self._config_mgr.save_profile(self._active_profile)
            QMessageBox.information(self, "OK", "Inšpekčné ROI uložené do profilu.")
        except Exception as e:
            QMessageBox.critical(self, "Chyba", str(e))

    def _on_save_params_to_profile(self) -> None:
        if self._active_profile is None:
            QMessageBox.warning(self, "Chyba", "Najprv načítaj profil.")
            return
        try:
            epsilon = float(self._epsilon_edit.text())
        except ValueError:
            epsilon = 1e-8
        self._active_profile.algorithm = self._algo_combo.currentText()
        self._active_profile.ecc_max_iter = self._iter_spin.value()
        self._active_profile.ecc_epsilon = epsilon
        self._active_profile.ecc_gauss_filt_size = self._gauss_filt_combo.currentData() or 1
        self._active_profile.auto_clahe = self._auto_clahe_check.isChecked()
        try:
            self._config_mgr.save_profile(self._active_profile)
            QMessageBox.information(self, "OK", "Parametre zarovnania uložené do profilu.")
        except Exception as e:
            QMessageBox.critical(self, "Chyba", str(e))

    # ------------------------------------------------------------------
    # Show/hide hrán
    # ------------------------------------------------------------------

    def _on_show_insp_edges_toggled(self, show: bool) -> None:
        self._show_insp_edges = show
        if show:
            self._update_insp_edges()
        else:
            self._insp_viewer.clear_overlay()
            if self._insp_roi is not None:
                self._insp_viewer.draw_roi(self._insp_roi)

    def _update_insp_edges(self) -> None:
        if self._insp_image is None or self._active_profile is None:
            return
        gray = (cv2.cvtColor(self._insp_image, cv2.COLOR_BGR2GRAY)
                if self._insp_image.ndim == 3 else self._insp_image.copy())
        params = self._get_edge_params_from_profile(self._active_profile)
        method = self._active_profile.edge_method
        roi = self._insp_roi
        if roi is not None:
            h, w = gray.shape
            x0c, y0c = max(0, roi.x0), max(0, roi.y0)
            x1c, y1c = min(w, roi.x1), min(h, roi.y1)
            if x1c > x0c and y1c > y0c:
                out = np.zeros(gray.shape, dtype=np.uint8)
                try:
                    out[y0c:y1c, x0c:x1c] = detect_edges(
                        gray[y0c:y1c, x0c:x1c], method, **params
                    )
                except Exception:
                    return
            else:
                return
        else:
            try:
                out = detect_edges(gray, method, **params)
            except Exception:
                return
        self._insp_edges = out
        self._insp_viewer.draw_edges(out)
        if self._insp_roi is not None:
            self._insp_viewer.draw_roi(self._insp_roi)

    # ------------------------------------------------------------------
    # Zarovnanie
    # ------------------------------------------------------------------

    def _on_run_alignment(self) -> None:
        if self._active_profile is None:
            QMessageBox.warning(self, "Chyba", "Najprv načítaj profil.")
            return
        if self._ref_image is None:
            QMessageBox.warning(self, "Chyba",
                                "Referenčný obraz nie je dostupný. Skontroluj cestu v profile.")
            return
        if self._insp_image is None:
            QMessageBox.warning(self, "Chyba", "Najprv načítaj inšpekčný obraz.")
            return

        profile = self._active_profile
        try:
            epsilon = float(self._epsilon_edit.text())
        except ValueError:
            epsilon = profile.ecc_epsilon

        # Zosynchronizuj parametre zarovnania z UI do lokálnej kópie
        algo = self._algo_combo.currentText()
        max_iter = self._iter_spin.value()
        gauss_filt = self._gauss_filt_combo.currentData() or 1
        auto_clahe = self._auto_clahe_check.isChecked()
        scale = profile.scale_mm_per_px

        # Zmenšenie inšpekčného obrazu na veľkosť referenčného
        ref_h, ref_w = self._ref_image.shape[:2]
        insp_h, insp_w = self._insp_image.shape[:2]
        insp_work = self._insp_image
        if (insp_h, insp_w) != (ref_h, ref_w):
            insp_work = cv2.resize(self._insp_image, (ref_w, ref_h),
                                   interpolation=cv2.INTER_AREA)

        ref_pre = preprocess(self._ref_image, auto_clahe=auto_clahe)
        insp_pre = preprocess(insp_work, auto_clahe=auto_clahe)

        # Maska — preferuj inšpekčné ROI, fallback na profil ROI
        mask = None
        roi_for_mask = self._insp_roi or profile.roi
        if roi_for_mask is not None and roi_for_mask.is_valid(ref_pre.shape[:2]):
            mask = roi_for_mask.create_mask(ref_pre.shape[:2])

        try:
            raw = align(
                ref_pre, insp_pre,
                max_iter=max_iter, epsilon=epsilon,
                mask=mask, algorithm=algo,
                gauss_filt_size=gauss_filt,
            )
        except Exception as e:
            QMessageBox.critical(self, "Chyba zarovnania", str(e))
            return

        cal = Calibration(mm_per_px=scale)
        result = AlignResult.from_dict(raw, cal)

        # Výsledky
        self._res_dx_px.setText(f"{result.dx_px:.4f}")
        self._res_dy_px.setText(f"{result.dy_px:.4f}")
        self._res_dx_mm.setText(f"{result.dx_mm:.4f}")
        self._res_dy_mm.setText(f"{result.dy_mm:.4f}")
        self._res_angle.setText(f"{result.angle_deg:.4f}")
        self._res_conf.setText(f"{result.confidence:.4f}")
        self._res_ncc.setText(f"{result.ncc_score:.4f}")
        self._res_time.setText(f"{result.elapsed_s:.3f}")

        _MIN_CONF = 0.60
        if result.confidence < _MIN_CONF:
            for lbl in (self._res_dx_px, self._res_dy_px, self._res_dx_mm,
                        self._res_dy_mm, self._res_angle):
                lbl.setStyleSheet("color: #ff6666;")
            self._res_conf.setStyleSheet("color: #ff6666; font-weight: bold;")
            self._insp_viewer.set_image(insp_work, reset_zoom=False)
            return

        for lbl in (self._res_dx_px, self._res_dy_px, self._res_dx_mm,
                    self._res_dy_mm, self._res_angle, self._res_conf, self._res_ncc):
            lbl.setStyleSheet("")

        # Overlay
        self._insp_viewer.set_image(insp_work)
        ref_edges = self._ref_edges
        projected_rgba = None
        if ref_edges is not None:
            h_e, w_e = ref_edges.shape
            h_i, w_i = insp_work.shape[:2]
            h_out = min(h_e, h_i); w_out = min(w_e, w_i)
            cx, cy = w_e / 2.0, h_e / 2.0
            M = cv2.getRotationMatrix2D((cx, cy), result.angle_deg, 1.0)
            M[0, 2] += result.dx_px; M[1, 2] += result.dy_px
            projected = cv2.warpAffine(ref_edges, M, (w_out, h_out))
            rgba = np.zeros((h_out, w_out, 4), dtype=np.uint8)
            rgba[projected > 0] = [0, 255, 80, 220]
            projected_rgba = np.ascontiguousarray(rgba)

        if self._insp_edges is not None:
            h_i, w_i = insp_work.shape[:2]
            h_ie, w_ie = self._insp_edges.shape
            h_c = min(h_i, h_ie); w_c = min(w_i, w_ie)
            cyan_rgba = np.zeros((h_c, w_c, 4), dtype=np.uint8)
            cyan_rgba[self._insp_edges[:h_c, :w_c] > 0] = [0, 220, 255, 180]
            if projected_rgba is not None:
                h_p, w_p = projected_rgba.shape[:2]
                h_m = min(h_c, h_p); w_m = min(w_c, w_p)
                green_mask = projected_rgba[:h_m, :w_m, 3] > 0
                combined = cyan_rgba.copy()
                combined[:h_m, :w_m][green_mask] = projected_rgba[:h_m, :w_m][green_mask]
                projected_rgba = np.ascontiguousarray(combined)
            else:
                projected_rgba = cyan_rgba

        self._insp_viewer.draw_overlay(result.dx_px, result.dy_px, projected_rgba)

        # Ťažisko
        centroid_ref = profile.selected_segment_centroid
        if centroid_ref is None and ref_edges is not None:
            mom = cv2.moments(ref_edges)
            if mom["m00"] > 0:
                centroid_ref = (mom["m10"] / mom["m00"], mom["m01"] / mom["m00"])

        if centroid_ref is not None:
            cx_seg, cy_seg = centroid_ref
            h_r, w_r = self._ref_image.shape[:2]
            cx_img, cy_img = w_r / 2.0, h_r / 2.0
            θ = math.radians(result.angle_deg)
            cx_new = (math.cos(θ) * (cx_seg - cx_img) + math.sin(θ) * (cy_seg - cy_img)
                      + cx_img + result.dx_px)
            cy_new = (-math.sin(θ) * (cx_seg - cx_img) + math.cos(θ) * (cy_seg - cy_img)
                      + cy_img + result.dy_px)
            dx_c = cx_new - cx_seg
            dy_c = cy_new - cy_seg
            self._res_centroid_pos.setText(f"x={cx_seg:.1f}  y={cy_seg:.1f}")
            self._res_dx_c_px.setText(f"{dx_c:.4f}")
            self._res_dy_c_px.setText(f"{dy_c:.4f}")
            self._res_dx_c_mm.setText(f"{dx_c * scale:.4f}")
            self._res_dy_c_mm.setText(f"{dy_c * scale:.4f}")
            self._insp_viewer.draw_centroid_displacement(cx_seg, cy_seg, cx_new, cy_new)
        else:
            for lbl in (self._res_centroid_pos, self._res_dx_c_px, self._res_dy_c_px,
                        self._res_dx_c_mm, self._res_dy_c_mm):
                lbl.setText("—")
