from __future__ import annotations

from typing import List, Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .constants import DEFAULT_COLORS, qcolor_from_hex
from .models import ClassDefinition


class QColorDialogWithParent:
    """Wrapper to shield against missing dialogs when no parent is set."""

    @staticmethod
    def get_color(parent: QWidget, initial: QColor) -> Optional[QColor]:
        from PySide6.QtWidgets import QColorDialog

        return QColorDialog.getColor(initial, parent, "Select Class Color")


class ClassManagerWidget(QWidget):
    """UI surface for managing class definitions."""

    classesChanged = Signal()
    autoPopulateRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._updating = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("<b>Class Management</b>")
        layout.addWidget(title)

        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Name", "Value", "Color"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(self.table)

        button_row = QHBoxLayout()
        self.add_button = QPushButton("Add Class")
        self.remove_button = QPushButton("Remove Selected")
        self.auto_button = QPushButton("Auto Detect From Labels")
        button_row.addWidget(self.add_button)
        button_row.addWidget(self.remove_button)
        button_row.addWidget(self.auto_button)
        layout.addLayout(button_row)

        self.add_button.clicked.connect(self.add_empty_row)
        self.remove_button.clicked.connect(self.remove_selected_row)
        self.auto_button.clicked.connect(self.autoPopulateRequested.emit)
        self.table.cellDoubleClicked.connect(self._handle_cell_double_clicked)
        self.table.itemChanged.connect(self._notify_changed)

    def _notify_changed(self) -> None:
        if not self._updating:
            self.classesChanged.emit()

    def add_empty_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        name_item = QTableWidgetItem(f"Class {row + 1}")
        value_item = QTableWidgetItem("0")
        color_item = QTableWidgetItem("#000000")
        color_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        self.table.setItem(row, 0, name_item)
        self.table.setItem(row, 1, value_item)
        self.table.setItem(row, 2, color_item)
        self._set_color_on_row(row, QColor(0, 0, 0))
        self.classesChanged.emit()

    def remove_selected_row(self) -> None:
        selected = self.table.currentRow()
        if selected >= 0:
            self.table.removeRow(selected)
            self.classesChanged.emit()

    def _handle_cell_double_clicked(self, row: int, column: int) -> None:
        if column != 2:
            return
        existing = self.table.item(row, column)
        start_color = existing.data(Qt.UserRole) if existing else QColor("#000000")
        if not isinstance(start_color, QColor):
            start_color = QColor(str(existing.text()))
        color = QColor(start_color)
        if not color.isValid():
            color = QColor("#000000")
        new_color = QColorDialogWithParent.get_color(self, color)
        if new_color and new_color.isValid():
            self._set_color_on_row(row, new_color)
            self.classesChanged.emit()

    def _set_color_on_row(self, row: int, color: QColor) -> None:
        item = self.table.item(row, 2)
        if item is None:
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.table.setItem(row, 2, item)
        item.setData(Qt.UserRole, color)
        item.setText(color.name())
        item.setBackground(color)
        text_color = QColor(255, 255, 255) if color.lightness() < 128 else QColor(0, 0, 0)
        item.setForeground(text_color)

    def set_classes(self, classes: Sequence[ClassDefinition]) -> None:
        self._updating = True
        try:
            self.table.setRowCount(0)
            for cls_def in classes:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(cls_def.name))
                self.table.setItem(row, 1, QTableWidgetItem(str(cls_def.value)))
                color_item = QTableWidgetItem(cls_def.color.name())
                color_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.table.setItem(row, 2, color_item)
                self._set_color_on_row(row, cls_def.color)
        finally:
            self._updating = False
        self.classesChanged.emit()

    def populate_from_values(self, values: Sequence[int]) -> None:
        classes: List[ClassDefinition] = []
        palette_count = len(DEFAULT_COLORS)
        for idx, value in enumerate(values):
            color = qcolor_from_hex(DEFAULT_COLORS[idx % palette_count])
            classes.append(ClassDefinition(name=f"Class {value}", value=int(value), color=color))
        self.set_classes(classes)

    def get_classes(self) -> List[ClassDefinition]:
        classes: List[ClassDefinition] = []
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            value_item = self.table.item(row, 1)
            color_item = self.table.item(row, 2)
            if name_item is None or value_item is None or color_item is None:
                continue
            try:
                value = int(value_item.text())
            except ValueError:
                continue
            color = color_item.data(Qt.UserRole)
            if not isinstance(color, QColor):
                color = QColor(color_item.text())
            if not color.isValid():
                color = QColor("#000000")
            classes.append(ClassDefinition(name_item.text().strip() or f"Class {value}", value, color))
        return classes
