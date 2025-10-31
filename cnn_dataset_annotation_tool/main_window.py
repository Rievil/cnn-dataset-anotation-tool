from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QPalette, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
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

from .class_manager import ClassManagerWidget
from .constants import fallback_color
from .io_utils import (
    load_dataset_from_folders,
    load_entries_from_parquet,
    load_label_image,
    load_rgb_image,
    save_entries_to_parquet,
    save_label_image,
)
from .label_canvas import LabelCanvas, ToolMode
from .models import ClassDefinition, DatasetEntry, EditOperation


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
        self._build_ui()

    # ----- UI construction --------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # Dataset controls
        dataset_row = QHBoxLayout()
        self.load_button = QPushButton("Load Dataset")
        self.add_image_button = QPushButton("Add Image")
        self.load_mask_button = QPushButton("Load Mask")
        self.save_button = QPushButton("Save Session")
        self.revert_button = QPushButton("Revert Current Label")
        self.export_button = QPushButton("Export Edited Labels")
        dataset_row.addWidget(self.load_button)
        dataset_row.addWidget(self.add_image_button)
        dataset_row.addWidget(self.load_mask_button)
        dataset_row.addWidget(self.save_button)
        dataset_row.addWidget(self.revert_button)
        dataset_row.addWidget(self.export_button)
        dataset_row.addStretch(1)
        self.dataset_status = QLabel("No dataset loaded")
        dataset_row.addWidget(self.dataset_status)
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

        # Brush info
        brush_hint = QLabel(
            "Brush: circular stroke. Hold Ctrl + mouse wheel to zoom, middle mouse to pan.\n"
            "Brush left click: source → target, right click: target → source.\n"
            "Lasso: hold left to trace an area, release to fill. Right click cancels. Magnetic lasso snaps to edges.\n"
            "Polygon: click to place vertices, click the start point to close. Right click closes with swapped classes or cancels."
        )
        brush_hint.setWordWrap(True)
        control_layout.addWidget(brush_hint, 2, 0, 1, 3)

        # Source / target selection
        self.source_combo = QComboBox()
        self.target_combo = QComboBox()
        control_layout.addWidget(QLabel("Source Class"), 3, 0)
        control_layout.addWidget(self.source_combo, 3, 1, 1, 2)
        control_layout.addWidget(QLabel("Target Class"), 4, 0)
        control_layout.addWidget(self.target_combo, 4, 1, 1, 2)

        self.switch_classes_button = QPushButton("Switch Class Values")
        control_layout.addWidget(self.switch_classes_button, 5, 0, 1, 3)

        # Tool selection
        self.tool_combo = QComboBox()
        self.tool_combo.addItem("Brush", ToolMode.BRUSH)
        self.tool_combo.addItem("Freehand Lasso", ToolMode.LASSO)
        self.tool_combo.addItem("Polygon", ToolMode.POLYGON)
        self.tool_combo.addItem("Magnetic Lasso", ToolMode.MAGNETIC_LASSO)
        control_layout.addWidget(QLabel("Editing Tool"), 6, 0)
        control_layout.addWidget(self.tool_combo, 6, 1, 1, 2)

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
        self.load_button.clicked.connect(self.load_dataset)
        self.add_image_button.clicked.connect(self._add_single_image)
        self.load_mask_button.clicked.connect(self._load_mask_for_current)
        self.save_button.clicked.connect(self.save_session)
        self.revert_button.clicked.connect(self.revert_current_label)
        self.export_button.clicked.connect(self.export_labels)
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
        self.class_manager.classesChanged.connect(self._handle_classes_changed)
        self.class_manager.autoPopulateRequested.connect(self._auto_populate_classes)
        self.edited_radio.toggled.connect(self._handle_label_view_toggled)
        self.original_radio.toggled.connect(self._handle_label_view_toggled)
        self.add_description_row_button.clicked.connect(self._add_description_row)
        self.remove_description_row_button.clicked.connect(self._remove_description_row)
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
        if entry.has_label:
            item.setForeground(self.image_list.palette().brush(QPalette.Text))
            item.setToolTip("")
        else:
            item.setForeground(Qt.red)
            item.setToolTip("Mask not loaded")

    def _refresh_image_list_styles(self) -> None:
        for idx in range(min(self.image_list.count(), len(self.entries))):
            item = self.image_list.item(idx)
            if item is not None:
                self._style_image_list_item(item, self.entries[idx])

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
        self.entries.append(entry)
        item = QListWidgetItem(entry.name)
        self._style_image_list_item(item, entry)
        self.image_list.addItem(item)
        self.dataset_status.setText(f"{len(self.entries)} item(s) in session")
        self._session_dirty = True
        self.set_current_index(len(self.entries) - 1)
        self.statusBar().showMessage(f"Added image {entry.name}", 4000)

    def _load_mask_for_current(self) -> None:
        if self.current_index is None or self.current_index < 0 or self.current_index >= len(self.entries):
            QMessageBox.information(self, "No image selected", "Select an image before loading a mask.")
            return

        entry = self.entries[self.current_index]
        start_dir = (
            str(entry.label_path.parent)
            if entry.label_path is not None
            else str(entry.image_path.parent)
            if entry.image_path.exists()
            else str(Path.cwd())
        )
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

        if label.shape != entry.image.shape[:2]:
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
            return
        self.current_index = row
        self.image_list.setCurrentRow(row)
        self._refresh_canvas()
        current_entry = self.entries[row]
        self.statusBar().showMessage(
            f"Viewing {current_entry.name} ({row + 1}/{len(self.entries)})"
        )
        self._update_history_view()

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

    def _handle_tool_changed(self, index: int) -> None:
        data = self.tool_combo.itemData(index)
        mode = data if isinstance(data, ToolMode) else ToolMode.BRUSH
        self.canvas.set_tool_mode(mode)
        self._update_tool_controls(mode)

    def _update_tool_controls(self, mode: ToolMode) -> None:
        is_brush = mode == ToolMode.BRUSH
        self.brush_slider.setEnabled(is_brush)
        self.brush_spin.setEnabled(is_brush)

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

    def _add_description_row(self) -> None:
        row = self.description_table.rowCount()
        self.description_table.insertRow(row)
        self.description_table.setItem(row, 0, QTableWidgetItem(""))
        self.description_table.setItem(row, 1, QTableWidgetItem(""))
        self.description_table.editItem(self.description_table.item(row, 0))

    def _remove_description_row(self) -> None:
        selection = self.description_table.selectionModel()
        if selection is None:
            return
        rows = sorted({index.row() for index in selection.selectedRows()}, reverse=True)
        if not rows:
            if self.description_table.rowCount() > 0:
                self.description_table.removeRow(self.description_table.rowCount() - 1)
            return
        for row in rows:
            self.description_table.removeRow(row)

    def _collect_description_entries(self) -> Dict[str, str]:
        entries: Dict[str, str] = {}
        for row in range(self.description_table.rowCount()):
            key_item = self.description_table.item(row, 0)
            value_item = self.description_table.item(row, 1)
            key = key_item.text().strip() if key_item else ""
            if not key:
                continue
            value = value_item.text().strip() if value_item else ""
            entries[key] = value
        return entries

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
            QMessageBox.information(self, "Nothing to export", "Load and edit labels first.")
            return
        destination = QFileDialog.getExistingDirectory(self, "Select export directory")
        if not destination:
            return
        dest_path = Path(destination)
        exported = 0
        skipped: List[str] = []
        for entry in self.entries:
            if entry.edited_label is None or entry.label_path is None:
                skipped.append(entry.name)
                continue
            output_path = dest_path / entry.label_path.name
            try:
                save_label_image(entry.edited_label, output_path)
                exported += 1
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Export warning",
                    f"Failed to export {entry.name}: {exc}",
                )
        message = f"Exported {exported} label(s) to {dest_path}"
        if skipped:
            message += f"; skipped {len(skipped)} without masks"
        self.statusBar().showMessage(message, 5000)

    # ----- Rendering --------------------------------------------------------
    def _refresh_canvas(self) -> None:
        if self.current_index is None:
            self.canvas.clear()
            return
        entry = self.entries[self.current_index]
        self.canvas.set_base_image(entry.image)
        self.canvas.set_label_array(entry.edited_label)
        classes = self.class_manager.get_classes()
        overlay_labels = entry.original_label if self._show_original_label else entry.edited_label
        pixmap = self._render_overlay(
            entry.image,
            overlay_labels,
            classes,
            self.alpha_slider.value() / 100.0,
        )
        self.canvas.set_pixmap(pixmap)
        self.canvas.viewport().update()

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


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
