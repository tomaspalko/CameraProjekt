"""MainWindow — hlavné okno aplikácie s tab-based layoutom."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QStatusBar,
    QMenuBar, QMenu, QMessageBox,
)
from PyQt6.QtGui import QAction

from src.config.config_manager import ConfigManager
from src.config.profile import Profile
from src.gui.inspection_panel import InspectionPanel
from src.gui.batch_panel import BatchPanel

_APP_NAME = "Weld Inspection Vision System"
_VERSION  = "0.5.0"


class MainWindow(QMainWindow):
    """Hlavné okno aplikácie."""

    def __init__(self, profiles_dir: str = "config/profiles") -> None:
        super().__init__()
        self._config_mgr = ConfigManager(profiles_dir)

        self.setWindowTitle(_APP_NAME)
        self.resize(1400, 800)

        self._build_menu()
        self._build_tabs()
        self._build_status_bar()

        # Prepoj signály medzi panelmi
        self._ref_editor.profile_changed.connect(self._on_profile_changed)

    # ------------------------------------------------------------------
    # Budovanie UI
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        # Súbor
        file_menu: QMenu = menu_bar.addMenu("Súbor")
        exit_action = QAction("Koniec", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Pomoc
        help_menu: QMenu = menu_bar.addMenu("Pomoc")
        about_action = QAction("O aplikácii…", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_tabs(self) -> None:
        self._ref_editor = InspectionPanel(self._config_mgr)
        self._batch_panel = BatchPanel(self._config_mgr)

        tabs = QTabWidget()
        tabs.addTab(self._ref_editor, "Inšpekcia")
        tabs.addTab(self._batch_panel, "Batch")
        self.setCentralWidget(tabs)

    def _build_status_bar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Pripravený")

    # ------------------------------------------------------------------
    # Handlery
    # ------------------------------------------------------------------

    def _on_profile_changed(self, profile: Profile) -> None:
        self._batch_panel.refresh_profiles()
        name = profile.name or "—"
        self._status.showMessage(f"Aktívny profil: {name}")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            f"O aplikácii — {_APP_NAME}",
            f"<b>{_APP_NAME}</b><br>"
            f"Verzia {_VERSION}<br><br>"
            "Systém pre sub-pixelovú registráciu zvárových spojov.<br>"
            "Algoritmy: ECC (Enhanced Correlation Coefficient), POC (Phase-Only Correlation).",
        )
