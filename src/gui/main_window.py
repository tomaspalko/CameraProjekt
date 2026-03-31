"""MainWindow — hlavné okno aplikácie s tab-based layoutom."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QStatusBar, QTabWidget,
    QMenuBar, QMenu, QMessageBox, QDialog,
)
from PyQt6.QtGui import QAction

from src.config.config_manager import ConfigManager
from src.gui.profile_editor_tab import ProfileEditorTab
from src.gui.inspection_tab import InspectionTab
from src.gui.startup_dialog import StartupDialog

_APP_NAME = "Weld Inspection Vision System"
_VERSION  = "0.6.0"


class MainWindow(QMainWindow):
    """Hlavné okno aplikácie — dve záložky + startup dialog."""

    def __init__(self, profiles_dir: str = "config/profiles") -> None:
        super().__init__()
        self._config_mgr = ConfigManager(profiles_dir)

        self.setWindowTitle(_APP_NAME)
        self.resize(1700, 950)

        self._build_menu()
        self._build_tabs()
        self._build_status_bar()

        # Startup dialog — výber profilu
        self._show_startup_dialog()

    # ------------------------------------------------------------------
    # Budovanie UI
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        file_menu: QMenu = menu_bar.addMenu("Súbor")
        new_action = QAction("Nový profil", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._on_new_profile)
        file_menu.addAction(new_action)

        open_action = QAction("Vybrať profil…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._show_startup_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()
        exit_action = QAction("Koniec", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu: QMenu = menu_bar.addMenu("Pomoc")
        about_action = QAction("O aplikácii…", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_tabs(self) -> None:
        self._tabs = QTabWidget()

        self._editor_tab = ProfileEditorTab(self._config_mgr)
        self._insp_tab   = InspectionTab(self._config_mgr)

        self._tabs.addTab(self._editor_tab, "⚙  Konfigurácia profilu")
        self._tabs.addTab(self._insp_tab,   "🔍  Inšpekcia")

        self.setCentralWidget(self._tabs)

        # Keď sa uloží profil v Tab 1 → obnov zoznam profilov v Tab 2
        self._editor_tab.profile_saved.connect(self._on_profile_saved)

    def _build_status_bar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Pripravený")

    # ------------------------------------------------------------------
    # Startup dialog
    # ------------------------------------------------------------------

    def _show_startup_dialog(self) -> None:
        dialog = StartupDialog(self._config_mgr, parent=self)
        result = dialog.exec()
        if result != QDialog.DialogCode.Accepted:
            return
        action, profile_name = dialog.get_selection()
        if action == "new":
            self._on_new_profile()
        elif action == "edit" and profile_name:
            try:
                profile = self._config_mgr.load_profile(profile_name)
                self._editor_tab.set_profile(profile)
            except Exception as e:
                QMessageBox.critical(self, "Chyba", str(e))
            self._tabs.setCurrentIndex(0)
            self._status.showMessage(
                f"Editácia profilu: {profile_name}"
            )
        elif action == "inspect" and profile_name:
            self._insp_tab.set_profile_by_name(profile_name)
            self._tabs.setCurrentIndex(1)
            self._status.showMessage(
                f"Inšpekcia: {profile_name}"
            )

    # ------------------------------------------------------------------
    # Handlery
    # ------------------------------------------------------------------

    def _on_new_profile(self) -> None:
        self._tabs.setCurrentIndex(0)
        self._status.showMessage("Nový profil — nakonfiguruj a ulož.")

    def _on_profile_saved(self, profile) -> None:
        self._insp_tab.refresh_profiles()
        self._status.showMessage(
            f"Profil uložený: [{profile.id}] {profile.name}"
        )

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            f"O aplikácii — {_APP_NAME}",
            f"<b>{_APP_NAME}</b><br>"
            f"Verzia {_VERSION}<br><br>"
            "Systém pre sub-pixelovú registráciu zvárových spojov.<br>"
            "Algoritmy: ECC (Enhanced Correlation Coefficient), POC (Phase-Only Correlation).",
        )
