"""StartupDialog — výber/vytvorenie profilu pri štarte aplikácie."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView,
)

from src.config.config_manager import ConfigManager


class StartupDialog(QDialog):
    """Modálny dialog na štarte — výber alebo vytvorenie profilu.

    Returns:
        exec() == QDialog.DialogCode.Accepted  vždy ak užívateľ zvolí akciu.
        get_selection() → (action, profile_name):
            action = "new"     — vytvoriť nový profil
            action = "edit"    — upraviť vybraný profil
            action = "inspect" — otvoriť inšpekciu s vybraným profilom
    """

    def __init__(self, config_mgr: ConfigManager, parent=None) -> None:
        super().__init__(parent)
        self._config_mgr = config_mgr
        self._action: str = "new"
        self._profile_name: str | None = None

        self.setWindowTitle("Weld Inspection System — výber profilu")
        self.setMinimumSize(480, 360)
        self._build_ui()
        self._refresh_table()

    # ------------------------------------------------------------------
    # Verejné
    # ------------------------------------------------------------------

    def get_selection(self) -> tuple[str, str | None]:
        """Vráti (action, profile_name)."""
        return self._action, self._profile_name

    # ------------------------------------------------------------------
    # Budovanie UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("<h2>Weld Inspection Vision System</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        hint = QLabel("Vyberte profil zo zoznamu alebo vytvorte nový:")
        layout.addWidget(hint)

        # Tabuľka profilov
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["ID", "Názov profilu"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._table)

        # Tlačidlá
        btn_layout = QHBoxLayout()

        self._btn_new = QPushButton("Nový profil")
        self._btn_new.setToolTip("Vytvoriť nový prázdny profil (Tab 1)")
        self._btn_new.clicked.connect(self._on_new)

        self._btn_edit = QPushButton("Upraviť profil")
        self._btn_edit.setToolTip("Editovať vybraný profil (Tab 1)")
        self._btn_edit.setEnabled(False)
        self._btn_edit.clicked.connect(self._on_edit)

        self._btn_inspect = QPushButton("Inšpekcia  ▶")
        self._btn_inspect.setToolTip("Spustiť inšpekciu s vybraným profilom (Tab 2)")
        self._btn_inspect.setEnabled(False)
        self._btn_inspect.setDefault(True)
        self._btn_inspect.clicked.connect(self._on_inspect)

        btn_layout.addWidget(self._btn_new)
        btn_layout.addWidget(self._btn_edit)
        btn_layout.addStretch()
        btn_layout.addWidget(self._btn_inspect)
        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Interné
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        profiles = self._config_mgr.list_profiles_full()
        self._table.setRowCount(len(profiles))
        for row, p in enumerate(profiles):
            id_item = QTableWidgetItem(str(p["id"]) if p["id"] else "—")
            id_item.setTextAlignment(
                int(Qt.AlignmentFlag.AlignRight) | int(Qt.AlignmentFlag.AlignVCenter)
            )
            self._table.setItem(row, 0, id_item)
            name_item = QTableWidgetItem(p["name"])
            name_item.setData(Qt.ItemDataRole.UserRole, p["name"])
            self._table.setItem(row, 1, name_item)

    def _selected_name(self) -> str | None:
        rows = self._table.selectedItems()
        if not rows:
            return None
        row = self._table.currentRow()
        item = self._table.item(row, 1)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _on_selection_changed(self) -> None:
        has = self._selected_name() is not None
        self._btn_edit.setEnabled(has)
        self._btn_inspect.setEnabled(has)

    def _on_double_click(self, _item) -> None:
        self._on_inspect()

    def _on_new(self) -> None:
        self._action = "new"
        self._profile_name = None
        self.accept()

    def _on_edit(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        self._action = "edit"
        self._profile_name = name
        self.accept()

    def _on_inspect(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        self._action = "inspect"
        self._profile_name = name
        self.accept()
