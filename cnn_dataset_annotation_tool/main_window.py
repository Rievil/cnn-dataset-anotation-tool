from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QObject, QPoint, Qt, QThread, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QImage,
    QKeySequence,
    QPixmap,
    QPalette,
    QShortcut,
    QBrush,
)
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from .about_dialog import AboutDialog
from .class_manager import ClassManagerWidget
from .constants import fallback_color
from .io_utils import (
    load_dataset_from_folders,
    load_entries_from_parquet,
    load_label_image,
    load_rgb_image,
    save_entries_to_parquet,
    save_label_image,
    save_rgb_image,
)
from .label_canvas import LabelCanvas, ToolMode
from .models import ClassDefinition, DatasetEntry, EditOperation


class ExportMode(Enum):
    FULL = "full"
    SUB_IMAGES = "sub_images"


@dataclass
class ExportOptions:
    destination: Path
    mode: ExportMode
    tile_width: int = 416
    tile_height: int = 416


class MainWindow(QMainWindow):
    """Primary window orchestrating the workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CNN Dataset Annotation Tool")
        self.resize(1200, 800)
        self.entries: List[DatasetEntry] = []
        self.current_index: Optional[int] = None
        self._session_dirty = False
        self._session_path: Optional[Path] = None
        self._suppress_class_dirty = False
        self._controls_last_size = 320
        self._show_original_label = False
        self._last_export_dir: Optional[Path] = None
        self._metadata_keys: List[str] = []
        self._updating_description_table = False
        self._create_actions()
        self._build_ui()
        self._create_menus()

    def _create_actions(self) -> None:
        self.load_dataset_action = QAction("Load Dataset...", self)
        self.load_dataset_action.setShortcut(QKeySequence.Open)
        self.load_dataset_action.triggered.connect(self.load_dataset)

        self.add_image_action = QAction("Add Image...", self)
        self.add_image_action.triggered.connect(self._add_single_image)

        self.load_mask_action = QAction("Add Label / Mask...", self)
        self.load_mask_action.triggered.connect(self._load_mask_for_current)

        self.save_session_action = QAction("Save Session", self)
        self.save_session_action.setShortcut(QKeySequence.Save)
        self.save_session_action.triggered.connect(
            lambda checked=False: self.save_session(prompt_for_path=False)
        )

        self.save_session_as_action = QAction("Save Session As...", self)
        self.save_session_as_action.setShortcut(QKeySequence.SaveAs)
        self.save_session_as_action.triggered.connect(
            lambda checked=False: self.save_session(prompt_for_path=True)
        )

        self.revert_label_action = QAction("Revert Current Label", self)
        self.revert_label_action.triggered.connect(self.revert_current_label)

        self.export_action = QAction("Export...", self)
        self.export_action.triggered.connect(self.export_labels)

        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.triggered.connect(self.close)

        self.about_action = QAction("About CNN Dataset Annotation Tool", self)
        self.about_action.triggered.connect(self.show_about_dialog)

    def _create_menus(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.load_dataset_action)
        file_menu.addAction(self.add_image_action)
        file_menu.addAction(self.load_mask_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_session_action)
        file_menu.addAction(self.save_session_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.revert_label_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        about_menu = menu_bar.addMenu("&About")
        about_menu.addAction(self.about_action)

    def show_about_dialog(self) -> None:
        dialog = AboutDialog(self)
        dialog.exec()

    # ----- UI construction --------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # Dataset summary and controls visibility toggle
        dataset_row = QHBoxLayout()
        self.dataset_status = QLabel("No dataset loaded")
        dataset_row.addWidget(self.dataset_status)
        dataset_row.addStretch(1)
        self.controls_toggle_button = QPushButton("Hide Controls")
        self.controls_toggle_button.setCheckable(True)
        self.controls_toggle_button.setChecked(False)
        dataset_row.addWidget(self.controls_toggle_button)
        root_layout.addLayout(dataset_row)

        self.splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(self.splitter, 1)

        # Image list panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(QLabel("<b>Image Pairs</b>"))
        self.image_list = QListWidget()
        self.image_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self._show_image_list_context_menu)
        left_layout.addWidget(self.image_list, 1)
        self.splitter.addWidget(left_panel)

        # Canvas area
        canvas_panel = QWidget()
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)

        self.canvas = LabelCanvas()
        canvas_layout.addWidget(self.canvas, 1)

        self.splitter.addWidget(canvas_panel)

        # Controls panel on the right
        self.controls_container = QWidget()
        controls_outer_layout = QVBoxLayout(self.controls_container)
        controls_outer_layout.setContentsMargins(0, 0, 0, 0)
        controls_outer_layout.setSpacing(0)

        self.controls_tabs = QTabWidget()
        controls_outer_layout.addWidget(self.controls_tabs)

        # Tab 1: editing tools
        tools_tab = QWidget()
        tools_layout = QVBoxLayout(tools_tab)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(10)

        control_panel = QGroupBox("Controls")
        control_layout = QGridLayout(control_panel)
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_layout.setHorizontalSpacing(12)
        control_layout.setVerticalSpacing(8)

        # Overlay alpha
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(60)
        self.alpha_value = QLabel("0.60")
        control_layout.addWidget(QLabel("Overlay Alpha"), 0, 0)
        control_layout.addWidget(self.alpha_slider, 0, 1)
        control_layout.addWidget(self.alpha_value, 0, 2)

        # Brush size
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 200)
        self.brush_slider.setValue(25)
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 200)
        self.brush_spin.setValue(25)
        control_layout.addWidget(QLabel("Brush Size"), 1, 0)
        control_layout.addWidget(self.brush_slider, 1, 1)
        control_layout.addWidget(self.brush_spin, 1, 2)

        self.polyline_thickness_label = QLabel(
            f"Line Thickness: {self.canvas.polyline_width()} px"
        )
        control_layout.addWidget(self.polyline_thickness_label, 2, 0, 1, 3)
        self.polyline_thickness_label.setVisible(False)

        # Brush info
        brush_hint = QLabel(
            "Brush: circular stroke. Hold Ctrl + mouse wheel to zoom, middle mouse to pan.\n"
            "Brush left click: source → target, right click: target → source.\n"
            "Lasso: hold left to trace an area, release to fill. Right click cancels. Magnetic lasso snaps to edges.\n"
            "Polygon: click to place vertices, click the start point to close. Right click closes with swapped classes or cancels.\n"
            "Polygon Line: click to place points along the crack. Mouse wheel adjusts thickness. Close on the start point to apply; right click reverses classes."
        )
        brush_hint.setWordWrap(True)
        control_layout.addWidget(brush_hint, 3, 0, 1, 3)

        # Source / target selection
        self.source_combo = QComboBox()
        self.target_combo = QComboBox()
        control_layout.addWidget(QLabel("Source Class"), 4, 0)
        control_layout.addWidget(self.source_combo, 4, 1, 1, 2)
        control_layout.addWidget(QLabel("Target Class"), 5, 0)
        control_layout.addWidget(self.target_combo, 5, 1, 1, 2)

        self.switch_classes_button = QPushButton("Switch Class Values")
        control_layout.addWidget(self.switch_classes_button, 6, 0, 1, 3)

        # Tool selection
        self.tool_combo = QComboBox()
        self.tool_combo.addItem("Brush", ToolMode.BRUSH)
        self.tool_combo.addItem("Freehand Lasso", ToolMode.LASSO)
        self.tool_combo.addItem("Polygon", ToolMode.POLYGON)
        self.tool_combo.addItem("Polygon Line", ToolMode.POLYLINE)
        self.tool_combo.addItem("Magnetic Lasso", ToolMode.MAGNETIC_LASSO)
        control_layout.addWidget(QLabel("Editing Tool"), 7, 0)
        control_layout.addWidget(self.tool_combo, 7, 1, 1, 2)

        tools_layout.addWidget(control_panel)

        # Class manager block
        self.class_manager = ClassManagerWidget()
        tools_layout.addWidget(self.class_manager, 1)

        self.controls_tabs.addTab(tools_tab, "Tools")

        # Tab 2: label view switching
        label_view_tab = QWidget()
        label_view_layout = QVBoxLayout(label_view_tab)
        label_view_layout.setContentsMargins(12, 12, 12, 12)
        label_view_layout.setSpacing(12)

        self.edited_radio = QRadioButton("Show edited label (current working copy)")
        self.original_radio = QRadioButton("Show original label (read-only preview)")
        self.edited_radio.setChecked(True)

        self.label_view_status = QLabel()
        self.label_view_status.setWordWrap(True)

        label_view_layout.addWidget(self.edited_radio)
        label_view_layout.addWidget(self.original_radio)
        label_view_layout.addWidget(self.label_view_status)
        label_view_layout.addStretch(1)

        self.controls_tabs.addTab(label_view_tab, "Label View")

        # Tab 3: dataset description table
        description_tab = QWidget()
        description_layout = QVBoxLayout(description_tab)
        description_layout.setContentsMargins(8, 8, 8, 8)
        description_layout.setSpacing(8)

        description_layout.addWidget(QLabel("Dataset description key/value pairs:"))
        self.description_table = QTableWidget(0, 2)
        self.description_table.setHorizontalHeaderLabels(["Key", "Value"])
        self.description_table.horizontalHeader().setStretchLastSection(True)
        self.description_table.verticalHeader().setVisible(False)
        self.description_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.description_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.description_table.setEditTriggers(QTableWidget.AllEditTriggers)
        description_layout.addWidget(self.description_table, 1)

        description_buttons = QHBoxLayout()
        description_buttons.addStretch(1)
        self.add_description_row_button = QPushButton("Add Row")
        self.remove_description_row_button = QPushButton("Remove Selected")
        description_buttons.addWidget(self.add_description_row_button)
        description_buttons.addWidget(self.remove_description_row_button)
        description_layout.addLayout(description_buttons)

        self.controls_tabs.addTab(description_tab, "Description")

        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        history_layout.setContentsMargins(8, 8, 8, 8)
        history_layout.setSpacing(8)
        history_layout.addWidget(QLabel("Edit history for the selected image:"))
        self.history_list = QListWidget()
        self.history_list.setSelectionMode(QAbstractItemView.NoSelection)
        history_layout.addWidget(self.history_list, 1)
        self.controls_tabs.addTab(history_tab, "History")

        self.splitter.addWidget(self.controls_container)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([260, 700, self._controls_last_size])

        self.setCentralWidget(central)

        # Signal wiring
        self.image_list.currentRowChanged.connect(self.set_current_index)
        self.alpha_slider.valueChanged.connect(self._handle_alpha_changed)
        self.brush_slider.valueChanged.connect(self._handle_brush_slider_changed)
        self.brush_spin.valueChanged.connect(self._handle_brush_spin_changed)
        self.source_combo.currentIndexChanged.connect(self._update_paint_values)
        self.target_combo.currentIndexChanged.connect(self._update_paint_values)
        self.switch_classes_button.clicked.connect(self._switch_classes)
        self.tool_combo.currentIndexChanged.connect(self._handle_tool_changed)
        self.canvas.labelEdited.connect(self._handle_label_edited)
        self.canvas.operationPerformed.connect(self._record_operation)
        self.canvas.polylineWidthChanged.connect(self._handle_polyline_width_changed)
        self.canvas.brushRadiusChanged.connect(self._handle_canvas_brush_radius_changed)
        self.class_manager.classesChanged.connect(self._handle_classes_changed)
        self.class_manager.autoPopulateRequested.connect(self._auto_populate_classes)
        self.edited_radio.toggled.connect(self._handle_label_view_toggled)
        self.original_radio.toggled.connect(self._handle_label_view_toggled)
        self.add_description_row_button.clicked.connect(self._add_description_row)
        self.remove_description_row_button.clicked.connect(self._remove_description_row)
        self.description_table.itemChanged.connect(self._handle_description_item_changed)
        self.controls_toggle_button.toggled.connect(self._handle_controls_toggled)

        self._undo_shortcut = QShortcut(QKeySequence.Undo, self)
        self._undo_shortcut.activated.connect(self._undo_current)
        self._redo_shortcut = QShortcut(QKeySequence.Redo, self)
        self._redo_shortcut.activated.connect(self._redo_current)
        self._redo_alt_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self._redo_alt_shortcut.activated.connect(self._redo_current)

        self._update_paint_values()
        self._handle_tool_changed(self.tool_combo.currentIndex())
        self._set_controls_visible(True)
        self._update_controls_toggle_text(True)
        self._update_label_view_status()
        self._update_history_view()

    # ----- Dataset handling -------------------------------------------------
    def load_dataset(self) -> None:
        if self._session_dirty and not self._confirm_discard_changes():
            return

        choice = QMessageBox(self)
        choice.setWindowTitle("Load Dataset")
        choice.setText("Select how to load the dataset.")
        parquet_button = choice.addButton("Parquet file", QMessageBox.AcceptRole)
        folders_button = choice.addButton("Image/label folders", QMessageBox.AcceptRole)
        choice.addButton(QMessageBox.Cancel)
        choice.setDefaultButton(folders_button)
        choice.exec()

        clicked = choice.clickedButton()
        if clicked == parquet_button:
            self._load_from_parquet()
        elif clicked == folders_button:
            self._load_from_folders()

    def _load_from_parquet(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select session parquet file",
            str(self._session_path.parent) if self._session_path else "",
            "Parquet Files (*.parquet);;All Files (*.*)",
        )
        if not file_path:
            return
        path = Path(file_path)
        try:
            entries, classes = load_entries_from_parquet(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", f"Failed to load parquet file:\n{exc}")
            return
        if not entries:
            QMessageBox.information(self, "Empty session", "The selected parquet did not contain any entries.")
            return
        self._apply_entries(entries, f"Loaded {len(entries)} entries from {path.name}", classes)
        self._session_path = path

    def _load_from_folders(self) -> None:
        image_dir = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not image_dir:
            return
        label_dir = QFileDialog.getExistingDirectory(self, "Select label folder")
        if not label_dir:
            return

        images_folder = Path(image_dir)
        labels_folder = Path(label_dir)
        entries, errors = load_dataset_from_folders(images_folder, labels_folder)
        if not entries:
            message = "Failed to load any image/label pairs."
            if errors:
                message += "\n" + "\n".join(errors[:5])
            QMessageBox.critical(self, "Load failed", message)
            return

        if errors:
            QMessageBox.information(
                self,
                "Partial load",
                "Some pairs were skipped:\n" + "\n".join(errors[:10]),
            )

        self._session_path = None
        self._apply_entries(entries, f"Loaded {len(entries)} pairs from folders")

    def _apply_entries(
        self,
        entries: List[DatasetEntry],
        status_text: str,
        classes: Optional[Sequence[ClassDefinition]] = None,
    ) -> None:
        self.entries = entries
        self._rebuild_metadata_keys()
        for entry in self.entries:
            self._ensure_entry_metadata_keys(entry)
        self.image_list.clear()
        self.description_table.setRowCount(0)
        for entry in entries:
            item = QListWidgetItem(entry.name)
            self._style_image_list_item(item, entry)
            self.image_list.addItem(item)
        self.dataset_status.setText(status_text)
        self._session_dirty = False
        if entries:
            self.set_current_index(0)
            if classes is not None:
                self._suppress_class_dirty = True
                try:
                    self.class_manager.set_classes(classes)
                finally:
                    self._suppress_class_dirty = False
            else:
                self._suppress_class_dirty = True
                try:
                    self._auto_populate_classes()
                finally:
                    self._suppress_class_dirty = False
        else:
            self.set_current_index(-1)
            self._suppress_class_dirty = True
            try:
                self.class_manager.set_classes([])
            finally:
                self._suppress_class_dirty = False

    def _style_image_list_item(self, item: QListWidgetItem, entry: DatasetEntry) -> None:
        has_image = entry.has_image
        has_label = entry.has_label
        base_foreground = self.image_list.palette().brush(QPalette.Text)
        base_background = self.image_list.palette().brush(QPalette.Base)
        item.setForeground(base_foreground)
        item.setBackground(base_background)
        tooltip_parts: List[str] = []
        if not (has_image and has_label):
            item.setForeground(Qt.red)
            if not has_image and not has_label:
                tooltip_parts.append("Image and mask not loaded")
            elif not has_image:
                tooltip_parts.append("Image not loaded")
            else:
                tooltip_parts.append("Mask not loaded")
        if entry.export_selected:
            item.setBackground(QBrush(QColor(200, 255, 200)))
            if has_image and has_label:
                tooltip_parts.append("Marked for export")
            else:
                tooltip_parts.append("Marked for export (incomplete data)")
        item.setToolTip("\n".join(tooltip_parts))

    def _refresh_image_list_styles(self) -> None:
        for idx in range(min(self.image_list.count(), len(self.entries))):
            item = self.image_list.item(idx)
            if item is not None:
                self._style_image_list_item(item, self.entries[idx])

    def _update_session_status_count(self) -> None:
        if self.entries:
            self.dataset_status.setText(f"{len(self.entries)} item(s) in session")
        else:
            self.dataset_status.setText("No dataset loaded")

    def _show_image_list_context_menu(self, position: QPoint) -> None:
        if not self.entries:
            return
        index = self.image_list.indexAt(position).row()
        if index < 0 or index >= len(self.entries):
            return
        entry = self.entries[index]
        menu = QMenu(self)
        remove_image_action = menu.addAction("Remove Image")
        remove_label_action = menu.addAction("Remove Label")
        menu.addSeparator()
        toggle_export_action = menu.addAction("Mark for Export")
        toggle_export_action.setCheckable(True)
        toggle_export_action.setChecked(entry.export_selected)
        menu.addSeparator()
        remove_item_action = menu.addAction("Remove Item")
        remove_image_action.setEnabled(entry.has_image)
        remove_label_action.setEnabled(entry.has_label)
        action = menu.exec(self.image_list.mapToGlobal(position))
        if action == remove_image_action:
            self._remove_entry_image(index)
        elif action == remove_label_action:
            self._remove_entry_label(index)
        elif action == toggle_export_action:
            self._toggle_export_selection(index, checked=toggle_export_action.isChecked())
        elif action == remove_item_action:
            self._remove_entry_item(index)

    def _remove_entry_image(self, index: int) -> None:
        if index < 0 or index >= len(self.entries):
            return
        entry = self.entries[index]
        if not entry.has_image:
            return
        confirm = QMessageBox.question(
            self,
            "Remove image?",
            (
                f"Remove the image for '{entry.name}' from the session?\n"
                "The label will remain available."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        entry.image = None
        entry.image_path = None
        self._session_dirty = True
        item = self.image_list.item(index)
        if item is not None:
            self._style_image_list_item(item, entry)
        if self.current_index == index:
            self._refresh_canvas()
        self.statusBar().showMessage(f"Removed image for {entry.name}", 4000)

    def _remove_entry_label(self, index: int) -> None:
        if index < 0 or index >= len(self.entries):
            return
        entry = self.entries[index]
        if not entry.has_label:
            return
        confirm = QMessageBox.question(
            self,
            "Remove label?",
            (
                f"Remove the label for '{entry.name}'?\n"
                "This will discard the original and edited masks."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        entry.label_path = None
        entry.original_label = None
        entry.edited_label = None
        entry.undo_stack.clear()
        entry.redo_stack.clear()
        self._session_dirty = True
        item = self.image_list.item(index)
        if item is not None:
            self._style_image_list_item(item, entry)
        if self.current_index == index:
            self._refresh_canvas()
            self._update_history_view()
        self.statusBar().showMessage(f"Removed label for {entry.name}", 4000)

    def _remove_entry_item(self, index: int) -> None:
        if index < 0 or index >= len(self.entries):
            return
        entry = self.entries[index]
        current_row = self.image_list.currentRow()
        confirm = QMessageBox.question(
            self,
            "Remove item?",
            f"Remove '{entry.name}' from the session?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        self.entries.pop(index)
        removed = self.image_list.takeItem(index)
        if removed is not None:
            del removed
        self._session_dirty = True
        if not self.entries:
            self.set_current_index(-1)
        else:
            if index == current_row:
                next_row = min(index, len(self.entries) - 1)
                self.image_list.setCurrentRow(next_row)
            elif current_row >= len(self.entries):
                self.image_list.setCurrentRow(len(self.entries) - 1)
        self._update_session_status_count()
        self._update_history_view()
        self.statusBar().showMessage(f"Removed item {entry.name}", 4000)

    def _toggle_export_selection(self, index: int, *, checked: bool) -> None:
        if index < 0 or index >= len(self.entries):
            return
        entry = self.entries[index]
        entry.export_selected = checked
        self._session_dirty = True
        item = self.image_list.item(index)
        if item is not None:
            self._style_image_list_item(item, entry)
        status = "Marked" if checked else "Unmarked"
        self.statusBar().showMessage(f"{status} {entry.name} for export", 2500)

    def _ensure_unique_entry_name(self, base_name: str) -> str:
        existing = {entry.name for entry in self.entries}
        candidate = base_name
        suffix = 1
        while candidate in existing:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        return candidate

    def _add_single_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image file",
            str(self._session_path.parent) if self._session_path else str(Path.cwd()),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*.*)",
        )
        if not file_path:
            return
        path = Path(file_path)
        try:
            image = load_rgb_image(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Image load failed", f"Could not load image:\n{exc}")
            return

        name = self._ensure_unique_entry_name(path.stem)
        entry = DatasetEntry(
            name=name,
            image_path=path,
            label_path=None,
            image=image,
            original_label=None,
            edited_label=None,
            metadata={},
        )
        self._ensure_entry_metadata_keys(entry)
        self.entries.append(entry)
        item = QListWidgetItem(entry.name)
        self._style_image_list_item(item, entry)
        self.image_list.addItem(item)
        self._update_session_status_count()
        self._session_dirty = True
        self.set_current_index(len(self.entries) - 1)
        self.statusBar().showMessage(f"Added image {entry.name}", 4000)

    def _load_mask_for_current(self) -> None:
        if self.current_index is None or self.current_index < 0 or self.current_index >= len(self.entries):
            QMessageBox.information(self, "No image selected", "Select an image before loading a mask.")
            return

        entry = self.entries[self.current_index]
        if entry.label_path is not None:
            start_dir = str(entry.label_path.parent)
        elif entry.image_path is not None and entry.image_path.exists():
            start_dir = str(entry.image_path.parent)
        else:
            start_dir = str(Path.cwd())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select mask for {entry.name}",
            start_dir,
            "Label Files (*.png *.tif *.tiff *.bmp);;All Files (*.*)",
        )
        if not file_path:
            return
        path = Path(file_path)
        try:
            label = load_label_image(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Mask load failed", f"Could not load mask:\n{exc}")
            return

        if entry.image is not None and label.shape != entry.image.shape[:2]:
            QMessageBox.critical(
                self,
                "Size mismatch",
                (
                    f"Mask dimensions {label.shape[1]}x{label.shape[0]} do not match image dimensions "
                    f"{entry.image.shape[1]}x{entry.image.shape[0]}."
                ),
            )
            return

        if entry.has_label:
            confirm = QMessageBox.question(
                self,
                "Replace existing mask?",
                (
                    "This image already has a mask. Replacing it will discard any edits "
                    "you have made. Continue?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return

        entry.label_path = path
        entry.original_label = label
        entry.edited_label = label.copy()
        entry.undo_stack.clear()
        entry.redo_stack.clear()
        self._session_dirty = True
        item = self.image_list.item(self.current_index)
        if item is not None:
            self._style_image_list_item(item, entry)
        self._refresh_canvas()
        self._update_history_view()
        self.statusBar().showMessage(f"Loaded mask for {entry.name}", 4000)

    def save_session(self, prompt_for_path: bool = True) -> bool:
        if not self.entries:
            QMessageBox.information(self, "Nothing to save", "Load or create a dataset before saving.")
            return False

        target_path: Optional[Path] = None
        if not prompt_for_path and self._session_path is not None:
            target_path = self._session_path
        else:
            directory = str(self._session_path.parent) if self._session_path else ""
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save session as parquet",
                directory,
                "Parquet Files (*.parquet);;All Files (*.*)",
            )
            if not file_path:
                return False
            target_path = Path(file_path)

            if target_path.suffix.lower() != ".parquet":
                target_path = target_path.with_suffix(".parquet")

        try:
            save_entries_to_parquet(self.entries, self.class_manager.get_classes(), target_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", f"Could not write parquet file:\n{exc}")
            return False

        self._session_path = target_path
        self._session_dirty = False
        self.statusBar().showMessage(f"Saved session to {target_path}", 5000)
        return True

    # ----- Image selection --------------------------------------------------
    def set_current_index(self, row: int) -> None:
        if row < 0 or row >= len(self.entries):
            self.current_index = None
            self.canvas.clear()
            self._update_history_view()
            self._load_description_table()
            return
        self.current_index = row
        self.image_list.setCurrentRow(row)
        self._refresh_canvas()
        current_entry = self.entries[row]
        self.statusBar().showMessage(
            f"Viewing {current_entry.name} ({row + 1}/{len(self.entries)})"
        )
        self._update_history_view()
        self._load_description_table()

    # ----- UI state updates -------------------------------------------------
    def _handle_alpha_changed(self, value: int) -> None:
        alpha = value / 100.0
        self.alpha_value.setText(f"{alpha:.2f}")
        self._refresh_canvas()

    def _handle_brush_slider_changed(self, value: int) -> None:
        if self.brush_spin.value() != value:
            self.brush_spin.setValue(value)
        self.canvas.set_brush_radius(value)

    def _handle_brush_spin_changed(self, value: int) -> None:
        if self.brush_slider.value() != value:
            self.brush_slider.setValue(value)
        self.canvas.set_brush_radius(value)

    def _handle_canvas_brush_radius_changed(self, radius: int) -> None:
        if self.brush_slider.value() != radius:
            self.brush_slider.setValue(radius)
        if self.brush_spin.value() != radius:
            self.brush_spin.setValue(radius)

    def _handle_polyline_width_changed(self, width: int) -> None:
        self.polyline_thickness_label.setText(f"Line Thickness: {width} px")

    def _handle_tool_changed(self, index: int) -> None:
        data = self.tool_combo.itemData(index)
        mode = data if isinstance(data, ToolMode) else ToolMode.BRUSH
        self.canvas.set_tool_mode(mode)
        self._update_tool_controls(mode)

    def _update_tool_controls(self, mode: ToolMode) -> None:
        is_brush = mode == ToolMode.BRUSH
        self.brush_slider.setEnabled(is_brush)
        self.brush_spin.setEnabled(is_brush)
        self.polyline_thickness_label.setVisible(mode == ToolMode.POLYLINE)

    def _handle_label_view_toggled(self) -> None:
        show_original = self.original_radio.isChecked()
        if show_original == self._show_original_label:
            return
        self._show_original_label = show_original
        self._update_label_view_status()
        self._refresh_canvas()

    def _update_label_view_status(self) -> None:
        if self._show_original_label:
            text = (
                "Viewing original labels. Editing tools still modify the edited copy; "
                "switch back to review your changes."
            )
        else:
            text = "Viewing edited labels (default working copy)."
        self.label_view_status.setText(text)

    def _handle_controls_toggled(self, checked: bool) -> None:
        visible = not checked
        self._set_controls_visible(visible, from_toggle=True)

    def _set_controls_visible(self, visible: bool, *, from_toggle: bool = False) -> None:
        if visible:
            self.controls_container.show()
            sizes = self.splitter.sizes()
            if len(sizes) < 3:
                sizes = list(sizes) + [self._controls_last_size]
            sizes = list(sizes[:3])
            if len(sizes) < 3:
                sizes.extend([300] * (3 - len(sizes)))
            if sizes[0] <= 0:
                sizes[0] = 260
            if sizes[1] <= 0:
                sizes[1] = max(self.splitter.width() - sizes[0] - self._controls_last_size, 400)
            self._controls_last_size = max(self._controls_last_size, 200)
            sizes[2] = self._controls_last_size
            self.splitter.setSizes(sizes)
        else:
            sizes = self.splitter.sizes()
            if len(sizes) >= 3 and sizes[2] > 0:
                self._controls_last_size = sizes[2]
            if len(sizes) >= 3:
                sizes = list(sizes)
                sizes[2] = 0
                self.splitter.setSizes(sizes)
            self.controls_container.hide()
        if not from_toggle:
            self.controls_toggle_button.blockSignals(True)
            self.controls_toggle_button.setChecked(not visible)
            self.controls_toggle_button.blockSignals(False)
        self._update_controls_toggle_text(visible)

    def _update_controls_toggle_text(self, visible: bool) -> None:
        self.controls_toggle_button.setText("Hide Controls" if visible else "Show Controls")

    def _rebuild_metadata_keys(self) -> None:
        keys: List[str] = []
        for entry in self.entries:
            for key in entry.metadata.keys():
                if key and key not in keys:
                    keys.append(key)
        self._metadata_keys = keys

    def _ensure_entry_metadata_keys(self, entry: DatasetEntry) -> None:
        for key in self._metadata_keys:
            if key and key not in entry.metadata:
                entry.metadata[key] = ""

    def _load_description_table(self) -> None:
        if self.current_index is None or self.current_index < 0 or self.current_index >= len(self.entries):
            self._updating_description_table = True
            self.description_table.setRowCount(0)
            self._updating_description_table = False
            return
        entry = self.entries[self.current_index]
        self._ensure_entry_metadata_keys(entry)
        self._updating_description_table = True
        self.description_table.setRowCount(0)
        for key in self._metadata_keys:
            row = self.description_table.rowCount()
            self.description_table.insertRow(row)
            self.description_table.setItem(row, 0, QTableWidgetItem(key))
            self.description_table.setItem(row, 1, QTableWidgetItem(entry.metadata.get(key, "")))
        self._updating_description_table = False

    def _add_description_row(self) -> None:
        self._metadata_keys.append("")
        self._session_dirty = True
        self._load_description_table()
        last_row = self.description_table.rowCount() - 1
        if last_row >= 0:
            self.description_table.editItem(self.description_table.item(last_row, 0))

    def _remove_description_row(self) -> None:
        selection = self.description_table.selectionModel()
        if selection is None:
            return
        rows = sorted({index.row() for index in selection.selectedRows()}, reverse=True)
        if not rows:
            if self.description_table.rowCount() > 0:
                rows = [self.description_table.rowCount() - 1]
            else:
                return
        for row in rows:
            if 0 <= row < len(self._metadata_keys):
                key = self._metadata_keys.pop(row)
                if key:
                    for entry in self.entries:
                        entry.metadata.pop(key, None)
        self._session_dirty = True
        self._load_description_table()

    def _handle_description_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_description_table or self.current_index is None:
            return
        row = item.row()
        column = item.column()
        if row < 0 or row >= len(self._metadata_keys):
            return
        current_key = self._metadata_keys[row]
        if column == 0:
            new_key = item.text().strip()
            if new_key == current_key:
                return
            if new_key and new_key in self._metadata_keys:
                # Prevent duplicate keys by restoring the old value.
                self._updating_description_table = True
                item.setText(current_key)
                self._updating_description_table = False
                return
            for entry in self.entries:
                previous_value = entry.metadata.pop(current_key, "") if current_key else ""
                if new_key:
                    entry.metadata[new_key] = previous_value
            self._metadata_keys[row] = new_key
            self._session_dirty = True
            self._load_description_table()
        elif column == 1:
            key = self._metadata_keys[row]
            if not key:
                return
            value = item.text()
            entry = self.entries[self.current_index]
            entry.metadata[key] = value
            self._session_dirty = True

    def _handle_classes_changed(self) -> None:
        if not self._suppress_class_dirty:
            self._session_dirty = True
        self._update_class_combos()
        self._refresh_canvas()

    def _update_class_combos(self) -> None:
        classes = self.class_manager.get_classes()
        self.source_combo.blockSignals(True)
        self.target_combo.blockSignals(True)
        source_value = self.source_combo.currentData()
        target_value = self.target_combo.currentData()
        self.source_combo.clear()
        self.target_combo.clear()
        self.source_combo.addItem("Any (all values)", None)
        for cls in classes:
            label = f"{cls.name} ({cls.value})"
            self.source_combo.addItem(label, cls.value)
            self.target_combo.addItem(label, cls.value)
        # Restore previous selections when possible
        for idx in range(self.source_combo.count()):
            if self.source_combo.itemData(idx) == source_value:
                self.source_combo.setCurrentIndex(idx)
                break
        else:
            self.source_combo.setCurrentIndex(0)

        if self.target_combo.count():
            if target_value is None:
                self.target_combo.setCurrentIndex(0)
            else:
                for idx in range(self.target_combo.count()):
                    if self.target_combo.itemData(idx) == target_value:
                        self.target_combo.setCurrentIndex(idx)
                        break
                else:
                    self.target_combo.setCurrentIndex(0)
        self.source_combo.blockSignals(False)
        self.target_combo.blockSignals(False)
        self._update_paint_values()

    def _class_label_for_value(self, value: int) -> str:
        for cls in self.class_manager.get_classes():
            if cls.value == value:
                return f"{cls.name} ({cls.value})"
        return str(value)

    def _update_paint_values(self) -> None:
        source = self.source_combo.currentData()
        target = self.target_combo.currentData()
        self.canvas.set_paint_values(source, target)

    def _switch_classes(self) -> None:
        if self.current_index is None:
            QMessageBox.information(self, "No image selected", "Select an entry before switching classes.")
            return
        source_value = self.source_combo.currentData()
        target_value = self.target_combo.currentData()
        if source_value is None or target_value is None:
            QMessageBox.information(
                self,
                "Selection required",
                "Pick specific source and target classes before switching their values.",
            )
            return
        if source_value == target_value:
            QMessageBox.information(
                self,
                "Same class selected",
                "Choose two different classes to perform a switch.",
            )
            return
        entry = self.entries[self.current_index]
        if not entry.has_label or entry.edited_label is None:
            QMessageBox.information(
                self,
                "Mask required",
                "Load a mask for this image before switching classes.",
            )
            return
        source_label = self._class_label_for_value(source_value)
        target_label = self._class_label_for_value(target_value)
        confirm = QMessageBox.question(
            self,
            "Switch classes",
            (
                "Switch all pixels labeled\n"
                f"- {source_label}\n"
                f"- {target_label}\n"
                f"in {entry.name}? This updates the edited label only."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        label = entry.edited_label
        src_mask = label == int(source_value)
        dst_mask = label == int(target_value)
        if not (np.any(src_mask) or np.any(dst_mask)):
            QMessageBox.information(
                self,
                "Nothing to change",
                "Neither class appears in the current edited label.",
            )
            return
        coords_src = np.argwhere(src_mask)
        coords_dst = np.argwhere(dst_mask)
        coords = np.vstack([coords_src, coords_dst]) if coords_dst.size else coords_src
        prev_values = np.concatenate(
            [
                np.full(coords_src.shape[0], int(source_value), dtype=np.int32),
                np.full(coords_dst.shape[0], int(target_value), dtype=np.int32),
            ]
        ) if coords_dst.size else np.full(coords_src.shape[0], int(source_value), dtype=np.int32)
        new_values = np.concatenate(
            [
                np.full(coords_src.shape[0], int(target_value), dtype=np.int32),
                np.full(coords_dst.shape[0], int(source_value), dtype=np.int32),
            ]
        ) if coords_dst.size else np.full(coords_src.shape[0], int(target_value), dtype=np.int32)
        label[src_mask] = int(target_value)
        label[dst_mask] = int(source_value)
        operation = EditOperation(
            f"Swap {source_label} ↔ {target_label}",
            coords.astype(np.int32, copy=True),
            prev_values.astype(np.int32, copy=False),
            new_values.astype(np.int32, copy=False),
        )
        self._record_operation(operation)
        self._refresh_canvas()
        self.statusBar().showMessage(
            f"Swapped {source_label} and {target_label} in {entry.name}",
            4000,
        )

    def _record_operation(self, operation: EditOperation) -> None:
        if self.current_index is None or self.current_index >= len(self.entries):
            return
        entry = self.entries[self.current_index]
        if entry.edited_label is None:
            return
        entry.undo_stack.append(operation)
        entry.redo_stack.clear()
        self._session_dirty = True
        self._update_history_view()
        count = operation.pixel_count()
        summary = f"{operation.description} ({count} px)" if count else operation.description
        self.statusBar().showMessage(summary, 2500)

    def _apply_operation(self, entry: DatasetEntry, operation: EditOperation, *, forward: bool) -> None:
        if entry.edited_label is None:
            return
        coords = operation.coordinates
        if coords.size == 0:
            return
        rows = coords[:, 0]
        cols = coords[:, 1]
        values = operation.new_values if forward else operation.previous_values
        entry.edited_label[rows, cols] = values

    def _undo_current(self) -> None:
        if self.current_index is None or self.current_index >= len(self.entries):
            return
        entry = self.entries[self.current_index]
        if not entry.undo_stack:
            return
        operation = entry.undo_stack.pop()
        self._apply_operation(entry, operation, forward=False)
        entry.redo_stack.append(operation)
        self._session_dirty = True
        self._refresh_canvas()
        self._update_history_view()
        self.statusBar().showMessage(f"Undid {operation.description}", 2500)

    def _redo_current(self) -> None:
        if self.current_index is None or self.current_index >= len(self.entries):
            return
        entry = self.entries[self.current_index]
        if not entry.redo_stack:
            return
        operation = entry.redo_stack.pop()
        self._apply_operation(entry, operation, forward=True)
        entry.undo_stack.append(operation)
        self._session_dirty = True
        self._refresh_canvas()
        self._update_history_view()
        self.statusBar().showMessage(f"Redid {operation.description}", 2500)

    def _update_history_view(self) -> None:
        self.history_list.clear()
        if self.current_index is None or self.current_index >= len(self.entries):
            item = QListWidgetItem("No image selected.")
            item.setFlags(Qt.ItemIsEnabled)
            self.history_list.addItem(item)
            return
        entry = self.entries[self.current_index]
        if not entry.undo_stack and not entry.redo_stack:
            item = QListWidgetItem("No edits recorded yet.")
            item.setFlags(Qt.ItemIsEnabled)
            self.history_list.addItem(item)
            return
        for index, operation in enumerate(entry.undo_stack, start=1):
            count = operation.pixel_count()
            text = f"{index}. {operation.description} ({count} px)" if count else f"{index}. {operation.description}"
            item = QListWidgetItem(text)
            item.setFlags(Qt.ItemIsEnabled)
            item.setToolTip(f"{count} pixel(s) affected" if count else operation.description)
            self.history_list.addItem(item)
        if entry.redo_stack:
            separator = QListWidgetItem("Redo queue:")
            separator.setFlags(Qt.ItemIsEnabled)
            separator.setForeground(Qt.gray)
            self.history_list.addItem(separator)
            for operation in reversed(entry.redo_stack):
                count = operation.pixel_count()
                text = f"↩ {operation.description} ({count} px)" if count else f"↩ {operation.description}"
                item = QListWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled)
                item.setForeground(Qt.gray)
                item.setToolTip(f"{count} pixel(s) pending redo" if count else operation.description)
                self.history_list.addItem(item)

    def _auto_populate_classes(self) -> None:
        if not self.entries:
            QMessageBox.information(self, "No dataset", "Load a dataset before auto detecting classes.")
            return
        values: List[int] = []
        for entry in self.entries:
            if entry.original_label is None:
                continue
            values.extend(np.unique(entry.original_label).tolist())
        unique_values = sorted(set(values))
        self.class_manager.populate_from_values(unique_values)
        self._update_class_combos()

    # ----- Editing actions --------------------------------------------------
    def _handle_label_edited(self) -> None:
        self._session_dirty = True
        self._refresh_canvas()

    def revert_current_label(self) -> None:
        if self.current_index is None:
            return
        entry = self.entries[self.current_index]
        if entry.original_label is None:
            QMessageBox.information(
                self,
                "No mask",
                "Load a mask for this image before reverting edits.",
            )
            return
        entry.reset_edits()
        self._session_dirty = True
        self._refresh_canvas()
        self._update_history_view()
        self.statusBar().showMessage(f"Reverted edits for {entry.name}", 3000)

    def export_labels(self) -> None:
        if not self.entries:
            QMessageBox.information(self, "Nothing to export", "Load or create entries before exporting.")
            return
        selected_entries = [entry for entry in self.entries if entry.export_selected]
        if not selected_entries:
            QMessageBox.information(
                self,
                "No items selected",
                "Mark at least one entry for export via the list context menu.",
            )
            return
        dialog = ExportOptionsDialog(self, initial_dir=self._last_export_dir)
        if dialog.exec() != QDialog.Accepted:
            return
        options = dialog.options()
        dest_path = options.destination
        try:
            dest_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", f"Could not prepare export folder:\n{exc}")
            return
        self._last_export_dir = dest_path
        manifest_rows: List[Dict[str, object]] = []
        include_subimage_fields = options.mode == ExportMode.SUB_IMAGES
        try:
            if options.mode == ExportMode.FULL:
                summary, manifest_rows = self._export_full_entries(dest_path, selected_entries)
            else:
                summary, manifest_rows = self._export_subimage_entries(
                    dest_path,
                    selected_entries,
                    options.tile_width,
                    options.tile_height,
                )
        except RuntimeError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        try:
            self._write_manifest_csv(dest_path, manifest_rows, include_subimage_fields)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Export warning",
                f"Export succeeded but writing the manifest CSV failed:\n{exc}",
            )
        self.statusBar().showMessage(summary, 5000)

    def _prepare_export_dirs(self, dest_path: Path) -> Tuple[Path, Path]:
        images_dir = dest_path / "Images"
        labels_dir = dest_path / "Labels"
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not prepare export folders:\n{exc}") from exc
        return images_dir, labels_dir

    def _gather_metadata_keys(self) -> List[str]:
        keys: List[str] = []
        for key in self._metadata_keys:
            if key and key not in keys:
                keys.append(key)
        for entry in self.entries:
            for key in entry.metadata.keys():
                if key and key not in keys:
                    keys.append(key)
        return keys

    def _build_manifest_row(
        self,
        dest_path: Path,
        image_output: Path,
        label_output: Path,
        entry: DatasetEntry,
        *,
        subimg_id: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> Dict[str, object]:
        row: Dict[str, object] = {
            "image_path": str(image_output.relative_to(dest_path)),
            "label_path": str(label_output.relative_to(dest_path)),
            "original_name": self._original_entry_name(entry),
            "subimg_id": subimg_id if subimg_id is not None else "",
            "x": x if x is not None else "",
            "y": y if y is not None else "",
        }
        for key in self._gather_metadata_keys():
            row[key] = entry.metadata.get(key, "")
        return row

    def _write_manifest_csv(
        self, dest_path: Path, rows: Sequence[Dict[str, object]], include_subimage_fields: bool
    ) -> None:
        metadata_keys = self._gather_metadata_keys()
        fieldnames: List[str] = ["image_path", "label_path", "original_name"]
        if include_subimage_fields:
            fieldnames.extend(["subimg_id", "x", "y"])
        fieldnames.extend(metadata_keys)
        manifest_path = dest_path / "dataset.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})

    def _original_entry_name(self, entry: DatasetEntry) -> str:
        if entry.image_path is not None and entry.image_path.name:
            return entry.image_path.name
        if entry.label_path is not None and entry.label_path.name:
            return entry.label_path.name
        return entry.name

    def _normalize_label_filename(self, source_name: str) -> str:
        """Remove redundant label suffixes while preserving the extension."""
        path = Path(source_name)
        stem = path.stem
        if stem.endswith("_label"):
            stem = stem[: -len("_label")]
        suffix = path.suffix or ".png"
        return f"{stem}{suffix}"

    def _export_full_entries(
        self, dest_path: Path, entries: Sequence[DatasetEntry]
    ) -> Tuple[str, List[Dict[str, object]]]:
        images_dir, labels_dir = self._prepare_export_dirs(dest_path)
        exported_images = 0
        exported_labels = 0
        skipped_images: List[str] = []
        skipped_labels: List[str] = []
        manifest_rows: List[Dict[str, object]] = []
        for entry in entries:
            image_saved = False
            label_saved = False
            image_output: Optional[Path] = None
            label_output: Optional[Path] = None
            if entry.image is not None:
                image_name = entry.image_path.name if entry.image_path else f"{entry.name}.png"
                image_output = images_dir / image_name
                try:
                    save_rgb_image(entry.image, image_output)
                    exported_images += 1
                    image_saved = True
                except Exception as exc:  # noqa: BLE001
                    skipped_images.append(entry.name)
                    QMessageBox.warning(
                        self,
                        "Export warning",
                        f"Failed to export image for {entry.name}: {exc}",
                    )
            else:
                skipped_images.append(entry.name)

            if entry.edited_label is not None:
                label_source_name = (
                    entry.label_path.name
                    if entry.label_path is not None and entry.label_path.name
                    else f"{entry.name}.png"
                )
                label_name = self._normalize_label_filename(label_source_name)
                label_output = labels_dir / label_name
                try:
                    save_label_image(entry.edited_label, label_output)
                    exported_labels += 1
                    label_saved = True
                except Exception as exc:  # noqa: BLE001
                    skipped_labels.append(entry.name)
                    QMessageBox.warning(
                        self,
                        "Export warning",
                        f"Failed to export label for {entry.name}: {exc}",
                    )
            else:
                skipped_labels.append(entry.name)

            if image_saved and label_saved and image_output is not None and label_output is not None:
                manifest_rows.append(self._build_manifest_row(dest_path, image_output, label_output, entry))

        message = (
            f"Exported {exported_images} image(s) and {exported_labels} label(s) "
            f"from {len(entries)} marked item(s) to {dest_path}"
        )
        if skipped_images:
            message += f"; skipped {len(skipped_images)} image(s)"
        if skipped_labels:
            message += f"; skipped {len(skipped_labels)} label(s)"
        return message, manifest_rows

    def _export_subimage_entries(
        self,
        dest_path: Path,
        entries: Sequence[DatasetEntry],
        tile_width: int,
        tile_height: int,
    ) -> Tuple[str, List[Dict[str, object]]]:
        if tile_width <= 0 or tile_height <= 0:
            raise RuntimeError("Tile dimensions must be positive.")
        images_dir, labels_dir = self._prepare_export_dirs(dest_path)
        total_pairs = 0
        skipped_items: List[str] = []
        manifest_rows: List[Dict[str, object]] = []
        for entry in entries:
            image = entry.image
            label = entry.edited_label
            if image is None or label is None:
                skipped_items.append(entry.name)
                continue
            if image.shape[:2] != label.shape:
                skipped_items.append(entry.name)
                continue
            height, width = label.shape
            if height < tile_height or width < tile_width:
                skipped_items.append(entry.name)
                continue
            tiles_created = 0
            tile_index = 1
            for top in range(0, height - tile_height + 1, tile_height):
                for left in range(0, width - tile_width + 1, tile_width):
                    tile_image = image[top : top + tile_height, left : left + tile_width]
                    tile_label = label[top : top + tile_height, left : left + tile_width]
                    tile_name = f"{entry.name}_sub_img_{tile_index:03d}"
                    image_output = images_dir / f"{tile_name}.png"
                    label_output = labels_dir / self._normalize_label_filename(f"{tile_name}.png")
                    try:
                        save_rgb_image(tile_image, image_output)
                        save_label_image(tile_label, label_output)
                    except Exception as exc:  # noqa: BLE001
                        QMessageBox.warning(
                            self,
                            "Export warning",
                            f"Failed to export {tile_name} for {entry.name}: {exc}",
                        )
                    else:
                        tiles_created += 1
                        total_pairs += 1
                        manifest_rows.append(
                            self._build_manifest_row(
                                dest_path,
                                image_output,
                                label_output,
                                entry,
                                subimg_id=tile_index,
                                x=left,
                                y=top,
                            )
                        )
                    finally:
                        tile_index += 1
            if tiles_created == 0:
                skipped_items.append(entry.name)
        message = (
            f"Exported {total_pairs} sub-image pair(s) of size {tile_width}x{tile_height} "
            f"from {len(entries)} item(s) to {dest_path}"
        )
        if skipped_items:
            message += f"; skipped {len(skipped_items)} item(s) (missing data or size mismatch)"
        return message, manifest_rows

    # ----- Rendering --------------------------------------------------------
    def _refresh_canvas(self) -> None:
        if self.current_index is None:
            self.canvas.clear()
            return
        entry = self.entries[self.current_index]
        label_source = entry.edited_label if entry.edited_label is not None else entry.original_label
        base_image = entry.image if entry.has_image else self._placeholder_from_label(label_source)
        if base_image is None:
            self.canvas.clear()
            return
        self.canvas.set_base_image(base_image)
        self.canvas.set_label_array(entry.edited_label)
        classes = self.class_manager.get_classes()
        overlay_labels = entry.original_label if self._show_original_label else entry.edited_label
        pixmap = self._render_overlay(
            base_image,
            overlay_labels,
            classes,
            self.alpha_slider.value() / 100.0,
        )
        self.canvas.set_pixmap(pixmap)
        self.canvas.viewport().update()

    def _placeholder_from_label(self, label: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if label is None:
            return None
        array = np.asarray(label, dtype=np.float32)
        if array.ndim != 2 or array.size == 0:
            return None
        min_val = float(array.min())
        max_val = float(array.max())
        if max_val > min_val:
            normalized = (array - min_val) / (max_val - min_val)
        elif max_val > 0:
            normalized = array / max_val
        else:
            normalized = np.zeros_like(array, dtype=np.float32)
        grayscale = np.clip(normalized * 255.0, 0, 255).astype(np.uint8, copy=False)
        rgb = np.repeat(grayscale[:, :, np.newaxis], 3, axis=2)
        return np.require(rgb, dtype=np.uint8, requirements=["C"])

    def _render_overlay(
        self,
        image: np.ndarray,
        labels: Optional[np.ndarray],
        classes: Sequence[ClassDefinition],
        alpha: float,
    ) -> QPixmap:
        image_rgb = np.asarray(image, dtype=np.float32)
        if image_rgb.ndim == 2:
            image_rgb = np.stack([image_rgb] * 3, axis=-1)
        blend_alpha = np.clip(alpha, 0.0, 1.0)
        if labels is None or labels.size == 0 or blend_alpha <= 0.0:
            blended = np.clip(image_rgb, 0, 255).astype(np.uint8)
        else:
            color_map: Dict[int, Tuple[int, int, int]] = {cls.value: cls.color_tuple() for cls in classes}
            label_values = np.asarray(labels, dtype=np.int32)
            if label_values.shape != image_rgb.shape[:2]:
                blended = np.clip(image_rgb, 0, 255).astype(np.uint8)
            else:
                unique_values = np.unique(label_values)
                if unique_values.size == 0:
                    color_overlay = np.zeros((*label_values.shape, 3), dtype=np.float32)
                else:
                    colors = np.array(
                        [color_map.get(int(value), fallback_color(int(value))) for value in unique_values],
                        dtype=np.float32,
                    )
                    indices = unique_values.searchsorted(label_values)
                    color_overlay = colors[indices]

                blended = image_rgb * (1.0 - blend_alpha) + color_overlay * blend_alpha
                blended = np.clip(blended, 0, 255).astype(np.uint8)
        height, width, _ = blended.shape
        bytes_per_line = 3 * width
        qimage = QImage(blended.data, width, height, bytes_per_line, QImage.Format_RGB888)
        qimage = qimage.copy()
        return QPixmap.fromImage(qimage)

    # ----- Lifecycle --------------------------------------------------------
    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._session_dirty:
            result = QMessageBox(self)
            result.setWindowTitle("Unsaved changes")
            result.setText("You have unsaved edits. Save them before exiting?")
            save_button = result.addButton("Save", QMessageBox.AcceptRole)
            discard_button = result.addButton("Discard", QMessageBox.DestructiveRole)
            result.addButton("Cancel", QMessageBox.RejectRole)
            result.setDefaultButton(save_button)
            result.exec()
            clicked = result.clickedButton()
            if clicked == save_button:
                if not self.save_session(prompt_for_path=self._session_path is None):
                    event.ignore()
                    return
            elif clicked != discard_button:
                event.ignore()
                return
        super().closeEvent(event)

    def _confirm_discard_changes(self) -> bool:
        confirm = QMessageBox.question(
            self,
            "Discard changes?",
            "You have unsaved edits. Loading will discard them. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return confirm == QMessageBox.Yes


class ExportOptionsDialog(QDialog):
    """Dialog guiding export configuration."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        initial_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Dataset")
        self.resize(420, 320)
        self._initial_dir = str(initial_dir) if initial_dir else ""

        layout = QVBoxLayout(self)

        mode_group_box = QGroupBox("Export Mode")
        mode_layout = QVBoxLayout(mode_group_box)
        self.full_radio = QRadioButton("Export full images && labels")
        self.tiles_radio = QRadioButton("Export sub-images with paired labels")
        self.full_radio.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.full_radio)
        self.mode_group.addButton(self.tiles_radio)
        mode_layout.addWidget(self.full_radio)
        mode_layout.addWidget(self.tiles_radio)
        layout.addWidget(mode_group_box)

        dimension_box = QGroupBox("Sub-image dimensions (pixels)")
        dimension_layout = QFormLayout(dimension_box)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(32, 8192)
        self.width_spin.setValue(416)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(32, 8192)
        self.height_spin.setValue(416)
        dimension_layout.addRow("Width", self.width_spin)
        dimension_layout.addRow("Height", self.height_spin)
        layout.addWidget(dimension_box)

        destination_box = QGroupBox("Destination folder")
        destination_layout = QHBoxLayout(destination_box)
        self.destination_edit = QLineEdit()
        self.destination_edit.setReadOnly(True)
        if self._initial_dir:
            self.destination_edit.setText(self._initial_dir)
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._choose_destination)
        destination_layout.addWidget(self.destination_edit, 1)
        destination_layout.addWidget(browse_button)
        layout.addWidget(destination_box)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.full_radio.toggled.connect(self._update_mode_state)
        self._update_mode_state()

    def _update_mode_state(self) -> None:
        enable_tiles = self.tiles_radio.isChecked()
        self.width_spin.setEnabled(enable_tiles)
        self.height_spin.setEnabled(enable_tiles)

    def _choose_destination(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select export folder",
            self.destination_edit.text() or self._initial_dir,
        )
        if directory:
            self.destination_edit.setText(directory)

    def accept(self) -> None:  # type: ignore[override]
        destination = self.destination_edit.text().strip()
        if not destination:
            QMessageBox.warning(self, "Destination required", "Select a folder to export the dataset.")
            return
        super().accept()

    def options(self) -> ExportOptions:
        mode = ExportMode.SUB_IMAGES if self.tiles_radio.isChecked() else ExportMode.FULL
        destination = Path(self.destination_edit.text().strip())
        return ExportOptions(
            destination=destination,
            mode=mode,
            tile_width=self.width_spin.value(),
            tile_height=self.height_spin.value(),
        )


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
