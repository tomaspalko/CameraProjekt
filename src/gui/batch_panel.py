"""BatchPanel — záložka Batch: spustenie batch spracovania, tabuľka výsledkov, štatistiky."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QFileDialog, QMessageBox, QSizePolicy,
)

from src.config.config_manager import ConfigManager
from src.config.profile import Profile
from src.batch.batch_processor import process_batch_profile, BatchResult


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class _BatchWorker(QThread):
    """Spúšťa batch spracovanie v samostatnom vlákne."""

    row_ready = pyqtSignal(int, dict)   # (index, row_dict)
    finished = pyqtSignal(object)       # BatchResult
    error = pyqtSignal(str)

    def __init__(
        self,
        profile: Profile,
        folder: str,
        export_csv: str | None,
        export_json: str | None,
    ) -> None:
        super().__init__()
        self._profile = profile
        self._folder = folder
        self._export_csv = export_csv
        self._export_json = export_json

    def run(self) -> None:
        try:
            result = process_batch_profile(
                self._profile,
                self._folder,
                export_csv=self._export_csv,
                export_json=self._export_json,
            )
            # Emituj každý riadok zvlášť (pre live update tabuľky)
            for i, row in enumerate(result.rows):
                self.row_ready.emit(i, row)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# BatchPanel widget
# ---------------------------------------------------------------------------

_COLUMNS = ["Súbor", "Stav", "dx_px", "dy_px", "dx_mm", "dy_mm", "Uhol°", "Spoľahlivosť"]
_STAT_METRICS = [
    ("dx_px",  "dx [px]"),
    ("dy_px",  "dy [px]"),
    ("dx_mm",  "dx [mm]"),
    ("dy_mm",  "dy [mm]"),
    ("angle_deg", "Uhol [°]"),
    ("confidence", "Spoľahlivosť"),
]


class BatchPanel(QWidget):
    """Panel pre dávkové spracovanie obrazov."""

    def __init__(self, config_mgr: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config_mgr = config_mgr
        self._worker: _BatchWorker | None = None
        self._export_csv_path: str | None = None
        self._export_json_path: str | None = None

        self._build_ui()
        self.refresh_profiles()

    # ------------------------------------------------------------------
    # Verejné
    # ------------------------------------------------------------------

    def refresh_profiles(self) -> None:
        """Obnoví zoznam profilov v comboboxe."""
        names = self._config_mgr.list_profiles()
        self._profile_combo.blockSignals(True)
        current = self._profile_combo.currentText()
        self._profile_combo.clear()
        self._profile_combo.addItems(names)
        idx = self._profile_combo.findText(current)
        if idx >= 0:
            self._profile_combo.setCurrentIndex(idx)
        self._profile_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Budovanie UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addWidget(self._build_control_group())
        layout.addWidget(self._build_table())
        layout.addWidget(self._build_stats_group())

    def _build_control_group(self) -> QGroupBox:
        grp = QGroupBox("Nastavenia batch")
        gl = QGridLayout(grp)

        # Priečinok
        gl.addWidget(QLabel("Priečinok:"), 0, 0)
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("cesta k priečinku s obrázkami…")
        gl.addWidget(self._folder_edit, 0, 1)
        btn_folder = QPushButton("Prehľadávať…")
        btn_folder.clicked.connect(self._on_browse_folder)
        gl.addWidget(btn_folder, 0, 2)

        # Profil
        gl.addWidget(QLabel("Profil:"), 1, 0)
        self._profile_combo = QComboBox()
        self._profile_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        gl.addWidget(self._profile_combo, 1, 1)
        btn_refresh = QPushButton("Obnoviť")
        btn_refresh.clicked.connect(self.refresh_profiles)
        gl.addWidget(btn_refresh, 1, 2)

        # Export CSV
        gl.addWidget(QLabel("Export CSV:"), 2, 0)
        self._csv_label = QLabel("—")
        gl.addWidget(self._csv_label, 2, 1)
        btn_csv = QPushButton("Vybrať…")
        btn_csv.clicked.connect(self._on_pick_csv)
        gl.addWidget(btn_csv, 2, 2)

        # Export JSON
        gl.addWidget(QLabel("Export JSON:"), 3, 0)
        self._json_label = QLabel("—")
        gl.addWidget(self._json_label, 3, 1)
        btn_json = QPushButton("Vybrať…")
        btn_json.clicked.connect(self._on_pick_json)
        gl.addWidget(btn_json, 3, 2)

        # Tlačidlo + progress
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Spustiť batch")
        self._run_btn.clicked.connect(self._on_run_batch)
        self._progress = QProgressBar()
        self._progress.setValue(0)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._progress, stretch=1)
        gl.addLayout(btn_row, 4, 0, 1, 3)

        return grp

    def _build_table(self) -> QWidget:
        grp = QGroupBox("Výsledky")
        layout = QVBoxLayout(grp)
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(_COLUMNS)):
            self._table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)
        return grp

    def _build_stats_group(self) -> QGroupBox:
        grp = QGroupBox("Štatistiky")
        gl = QGridLayout(grp)

        self._stat_count_label = QLabel("—")
        gl.addWidget(QLabel("Počet OK / celkom:"), 0, 0)
        gl.addWidget(self._stat_count_label, 0, 1, 1, 3)

        # Hlavička
        for col, header in enumerate(["Metrika", "Priemer ± std", "Min", "Max"]):
            lbl = QLabel(f"<b>{header}</b>")
            gl.addWidget(lbl, 1, col)

        self._stat_labels: dict[str, tuple[QLabel, QLabel, QLabel]] = {}
        for row_idx, (key, display) in enumerate(_STAT_METRICS, start=2):
            mean_std = QLabel("—")
            min_lbl = QLabel("—")
            max_lbl = QLabel("—")
            gl.addWidget(QLabel(display), row_idx, 0)
            gl.addWidget(mean_std, row_idx, 1)
            gl.addWidget(min_lbl, row_idx, 2)
            gl.addWidget(max_lbl, row_idx, 3)
            self._stat_labels[key] = (mean_std, min_lbl, max_lbl)

        return grp

    # ------------------------------------------------------------------
    # Handlery
    # ------------------------------------------------------------------

    def _on_browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Vyber priečinok s obrázkami")
        if folder:
            self._folder_edit.setText(folder)

    def _on_pick_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Uložiť CSV", "", "CSV súbory (*.csv)"
        )
        if path:
            self._export_csv_path = path
            self._csv_label.setText(Path(path).name)

    def _on_pick_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Uložiť JSON", "", "JSON súbory (*.json)"
        )
        if path:
            self._export_json_path = path
            self._json_label.setText(Path(path).name)

    def _on_run_batch(self) -> None:
        folder = self._folder_edit.text().strip()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "Chyba", "Vyber platný priečinok s obrázkami.")
            return

        profile_name = self._profile_combo.currentText()
        if not profile_name:
            QMessageBox.warning(self, "Chyba", "Vyber profil.")
            return

        try:
            profile = self._config_mgr.load_profile(profile_name)
        except Exception as e:
            QMessageBox.critical(self, "Chyba načítania profilu", str(e))
            return

        errors = profile.validate()
        if errors:
            QMessageBox.warning(self, "Neplatný profil", "\n".join(errors))
            return

        # Priprav tabuľku
        self._table.setRowCount(0)
        self._progress.setValue(0)
        self._run_btn.setEnabled(False)
        self._clear_stats()

        self._worker = _BatchWorker(
            profile=profile,
            folder=folder,
            export_csv=self._export_csv_path,
            export_json=self._export_json_path,
        )
        self._worker.row_ready.connect(self._on_row_ready)
        self._worker.finished.connect(self._on_batch_finished)
        self._worker.error.connect(self._on_batch_error)
        self._worker.start()

    def _on_row_ready(self, idx: int, row: dict) -> None:
        self._table.insertRow(idx)
        values = [
            row.get("filename", ""),
            row.get("status", ""),
            self._fmt(row.get("dx_px")),
            self._fmt(row.get("dy_px")),
            self._fmt(row.get("dx_mm")),
            self._fmt(row.get("dy_mm")),
            self._fmt(row.get("angle_deg")),
            self._fmt(row.get("confidence")),
        ]
        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if row.get("status") == "ERROR":
                item.setForeground(Qt.GlobalColor.red)
            self._table.setItem(idx, col, item)
        self._progress.setValue(idx + 1)
        self._table.scrollToBottom()

    def _on_batch_finished(self, result: BatchResult) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setValue(self._table.rowCount())
        self._fill_stats(result)

        msg = f"Hotovo: {result.stats.count_ok} / {result.stats.count_total} OK"
        if result.stats.count_error:
            msg += f"  ({result.stats.count_error} chýb)"
        QMessageBox.information(self, "Batch dokončený", msg)

    def _on_batch_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        QMessageBox.critical(self, "Chyba batch", msg)

    # ------------------------------------------------------------------
    # Štatistiky
    # ------------------------------------------------------------------

    def _fill_stats(self, result: BatchResult) -> None:
        s = result.stats
        self._stat_count_label.setText(
            f"{s.count_ok} / {s.count_total}  (chyby: {s.count_error})"
        )
        self._progress.setMaximum(s.count_total)
        for key, (mean_std_lbl, min_lbl, max_lbl) in self._stat_labels.items():
            mean_val = getattr(s, f"{key}_mean", None)
            std_val  = getattr(s, f"{key}_std",  None)
            min_val  = getattr(s, f"{key}_min",  None)
            max_val  = getattr(s, f"{key}_max",  None)
            if mean_val is None:
                mean_std_lbl.setText("—")
                min_lbl.setText("—")
                max_lbl.setText("—")
            else:
                mean_std_lbl.setText(f"{mean_val:.4f} ± {std_val:.4f}")
                min_lbl.setText(f"{min_val:.4f}")
                max_lbl.setText(f"{max_val:.4f}")

    def _clear_stats(self) -> None:
        self._stat_count_label.setText("—")
        for mean_std_lbl, min_lbl, max_lbl in self._stat_labels.values():
            mean_std_lbl.setText("—")
            min_lbl.setText("—")
            max_lbl.setText("—")

    # ------------------------------------------------------------------
    # Pomocné
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(val) -> str:
        if val is None:
            return "—"
        try:
            return f"{float(val):.4f}"
        except (TypeError, ValueError):
            return str(val)
