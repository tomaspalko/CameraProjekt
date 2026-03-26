"""ReferenceEditor — záložka Konfigurácia: referenčný obraz, ROI, kalibrácia, profil, test."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QDoubleSpinBox,
    QSpinBox, QComboBox, QScrollArea, QFileDialog, QMessageBox,
    QSizePolicy,
)

from src.core.roi import ROI
from src.core.preprocessor import preprocess
from src.core.aligner import align
from src.core.calibration import Calibration
from src.core.result import AlignResult
from src.config.profile import Profile
from src.config.config_manager import ConfigManager
from src.gui.image_viewer import ImageViewer


class ReferenceEditor(QWidget):
    """Panel konfigurácie: načítanie referencie, kreslenie ROI, nastavenia, profil a test.

    Signals:
        profile_changed(Profile): emitovaný po každej zmene konfigurácie.
    """

    profile_changed = pyqtSignal(object)

    def __init__(self, config_mgr: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config_mgr = config_mgr
        self._ref_image: np.ndarray | None = None
        self._test_image: np.ndarray | None = None
        self._roi: ROI | None = None

        self._build_ui()
        self._connect_signals()
        self._refresh_profile_list()

    # ------------------------------------------------------------------
    # Verejné metódy
    # ------------------------------------------------------------------

    def get_profile(self) -> Profile:
        """Vráti aktuálny profil z formulára."""
        roi = self._roi
        try:
            epsilon = float(self._epsilon_edit.text())
        except ValueError:
            epsilon = 1e-8

        return Profile(
            name=self._name_edit.text().strip() or "unnamed",
            reference_path=self._ref_path_label.text(),
            roi=roi,
            scale_mm_per_px=self._scale_spin.value(),
            algorithm=self._algo_combo.currentText(),
            ecc_max_iter=self._iter_spin.value(),
            ecc_epsilon=epsilon,
        )

    def set_profile(self, profile: Profile) -> None:
        """Naplní formulár hodnotami z profilu."""
        self._name_edit.setText(profile.name)
        self._ref_path_label.setText(profile.reference_path)
        self._scale_spin.setValue(profile.scale_mm_per_px)

        idx = self._algo_combo.findText(profile.algorithm)
        if idx >= 0:
            self._algo_combo.setCurrentIndex(idx)

        self._iter_spin.setValue(profile.ecc_max_iter)
        self._epsilon_edit.setText(str(profile.ecc_epsilon))

        self._roi = profile.roi
        if profile.reference_path and Path(profile.reference_path).exists():
            self._load_image_from_path(profile.reference_path)

        if self._roi is not None and self._ref_image is not None:
            self._viewer.draw_roi(self._roi)
            self._update_roi_spinboxes()

    # ------------------------------------------------------------------
    # Budovanie UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Ľavá strana — image viewer
        self._viewer = ImageViewer()
        self._viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        splitter.addWidget(self._viewer)

        # Pravá strana — scroll area s nastaveniami
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(280)
        right_scroll.setMaximumWidth(360)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)

        right_layout.addWidget(self._build_ref_group())
        right_layout.addWidget(self._build_roi_group())
        right_layout.addWidget(self._build_calib_group())
        right_layout.addWidget(self._build_algo_group())
        right_layout.addWidget(self._build_profile_group())
        right_layout.addWidget(self._build_test_group())
        right_layout.addStretch()

        right_scroll.setWidget(right_widget)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

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

    def _build_algo_group(self) -> QGroupBox:
        grp = QGroupBox("Algoritmus")
        form = QFormLayout(grp)
        self._algo_combo = QComboBox()
        self._algo_combo.addItems(["ECC", "POC"])
        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(100, 10000)
        self._iter_spin.setValue(2000)
        self._epsilon_edit = QLineEdit("1e-08")
        form.addRow("Algoritmus:", self._algo_combo)
        form.addRow("Max iterácií:", self._iter_spin)
        form.addRow("Epsilon:", self._epsilon_edit)
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

    def _build_test_group(self) -> QGroupBox:
        grp = QGroupBox("Test zarovnania")
        layout = QVBoxLayout(grp)

        btn_load = QPushButton("Načítaj testovací obraz…")
        btn_load.clicked.connect(self._on_load_test_image)
        self._test_path_label = QLabel("—")
        self._test_path_label.setWordWrap(True)

        btn_run = QPushButton("Spustiť zarovnanie")
        btn_run.clicked.connect(self._on_run_test)

        self._result_label = QLabel("—")
        self._result_label.setWordWrap(True)
        self._result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout.addWidget(btn_load)
        layout.addWidget(self._test_path_label)
        layout.addWidget(btn_run)
        layout.addWidget(self._result_label)
        return grp

    # ------------------------------------------------------------------
    # Signály
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._viewer.roi_selected.connect(self._on_roi_selected)
        self._draw_roi_btn.toggled.connect(self._viewer.set_roi_mode)
        self._clear_roi_btn.clicked.connect(self._on_clear_roi)

        # Každá zmena nastavenia → profile_changed
        self._scale_spin.valueChanged.connect(self._emit_profile_changed)
        self._algo_combo.currentTextChanged.connect(self._emit_profile_changed)
        self._iter_spin.valueChanged.connect(self._emit_profile_changed)
        self._epsilon_edit.textChanged.connect(self._emit_profile_changed)
        self._name_edit.textChanged.connect(self._emit_profile_changed)

        # ROI spinboxy → aktualizuj ROI objekt + vizualizáciu
        for spin in (self._roi_x0, self._roi_y0, self._roi_x1, self._roi_y1):
            spin.valueChanged.connect(self._on_roi_spinbox_changed)

    # ------------------------------------------------------------------
    # Handlery
    # ------------------------------------------------------------------

    def _on_load_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Vyber referenčný obraz", "",
            "Obrázky (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if path:
            self._load_image_from_path(path)
            self._ref_path_label.setText(path)
            self._emit_profile_changed()

    def _load_image_from_path(self, path: str) -> None:
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Chyba", f"Nemôžem načítať obraz:\n{path}")
            return
        self._ref_image = img
        self._viewer.set_image(img)
        if self._roi is not None:
            self._viewer.draw_roi(self._roi)

    def _on_roi_selected(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Volaný keď používateľ nakreslí ROI v ImageViewer."""
        self._draw_roi_btn.setChecked(False)
        try:
            roi = ROI(x0, y0, x1, y1)
            if not roi.is_valid():
                return
            self._roi = roi
            self._update_roi_spinboxes()
            self._viewer.draw_roi(roi)
            self._emit_profile_changed()
        except Exception:
            pass

    def _on_clear_roi(self) -> None:
        self._roi = None
        self._viewer.clear_roi()
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
                self._viewer.draw_roi(roi)
                self._emit_profile_changed()
            except Exception:
                pass

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

    def _on_load_test_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Vyber testovací obraz", "",
            "Obrázky (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if path:
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, "Chyba", f"Nemôžem načítať obraz:\n{path}")
                return
            self._test_image = img
            self._test_path_label.setText(path)

    def _on_run_test(self) -> None:
        if self._ref_image is None:
            QMessageBox.warning(self, "Chyba", "Najprv načítaj referenčný obraz.")
            return
        if self._test_image is None:
            QMessageBox.warning(self, "Chyba", "Najprv načítaj testovací obraz.")
            return

        profile = self.get_profile()

        ref_pre = preprocess(self._ref_image)
        test_pre = preprocess(self._test_image)

        mask = None
        if self._roi is not None and self._roi.is_valid(ref_pre.shape[:2]):
            mask = self._roi.create_mask(ref_pre.shape[:2])

        try:
            raw = align(
                ref_pre, test_pre,
                max_iter=profile.ecc_max_iter,
                epsilon=profile.ecc_epsilon,
                mask=mask,
                algorithm=profile.algorithm,
            )
        except Exception as e:
            QMessageBox.critical(self, "Chyba zarovnania", str(e))
            return

        cal = Calibration(mm_per_px=profile.scale_mm_per_px)
        result = AlignResult.from_dict(raw, cal)

        self._result_label.setText(
            f"dx = {result.dx_px:.4f} px  ({result.dx_mm:.4f} mm)\n"
            f"dy = {result.dy_px:.4f} px  ({result.dy_mm:.4f} mm)\n"
            f"uhol = {result.angle_deg:.4f}°\n"
            f"spoľahlivosť = {result.confidence:.4f}"
        )

        # Overlay: Canny hrany testovacej snímky + šípka
        edges = cv2.Canny(test_pre, 50, 150)
        self._viewer.set_image(self._ref_image)
        if self._roi is not None:
            self._viewer.draw_roi(self._roi)
        self._viewer.draw_overlay(result.dx_px, result.dy_px, edges)

    def _refresh_profile_list(self) -> None:
        names = self._config_mgr.list_profiles()
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        self._profile_combo.addItems(names)
        self._profile_combo.blockSignals(False)

    def _emit_profile_changed(self) -> None:
        self.profile_changed.emit(self.get_profile())
