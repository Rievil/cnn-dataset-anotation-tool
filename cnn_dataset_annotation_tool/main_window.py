from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
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
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from .class_manager import ClassManagerWidget
from .constants import fallback_color
from .io_utils import (
    load_dataset_from_folders,
    load_entries_from_parquet,
    save_entries_to_parquet,
    save_label_image,
)
from .label_canvas import LabelCanvas
from .models import ClassDefinition, DatasetEntry


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
        self.save_button = QPushButton("Save Session")
        self.revert_button = QPushButton("Revert Current Label")
        self.export_button = QPushButton("Export Edited Labels")
        dataset_row.addWidget(self.load_button)
        dataset_row.addWidget(self.save_button)
        dataset_row.addWidget(self.revert_button)
        dataset_row.addWidget(self.export_button)
        dataset_row.addStretch(1)
        self.dataset_status = QLabel("No dataset loaded")
        dataset_row.addWidget(self.dataset_status)
        root_layout.addLayout(dataset_row)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, 1)

        # Image list panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(QLabel("<b>Image Pairs</b>"))
        self.image_list = QListWidget()
        left_layout.addWidget(self.image_list, 1)
        splitter.addWidget(left_panel)

        # Viewer + controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.canvas = LabelCanvas()
        right_layout.addWidget(self.canvas, 1)

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
            "Brush shape: circle. Hold Ctrl + mouse wheel to zoom, middle mouse to pan.\n"
            "Left click: source → target, right click: target → source."
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

        right_layout.addWidget(control_panel)

        # Class manager block
        self.class_manager = ClassManagerWidget()
        right_layout.addWidget(self.class_manager)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(central)

        # Signal wiring
        self.load_button.clicked.connect(self.load_dataset)
        self.save_button.clicked.connect(self.save_session)
        self.revert_button.clicked.connect(self.revert_current_label)
        self.export_button.clicked.connect(self.export_labels)
        self.image_list.currentRowChanged.connect(self.set_current_index)
        self.alpha_slider.valueChanged.connect(self._handle_alpha_changed)
        self.brush_slider.valueChanged.connect(self._handle_brush_slider_changed)
        self.brush_spin.valueChanged.connect(self._handle_brush_spin_changed)
        self.source_combo.currentIndexChanged.connect(self._update_paint_values)
        self.target_combo.currentIndexChanged.connect(self._update_paint_values)
        self.canvas.labelEdited.connect(self._handle_label_edited)
        self.class_manager.classesChanged.connect(self._handle_classes_changed)
        self.class_manager.autoPopulateRequested.connect(self._auto_populate_classes)

        self._update_paint_values()

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
        for entry in entries:
            item = QListWidgetItem(entry.name)
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
            return
        self.current_index = row
        self.image_list.setCurrentRow(row)
        self._refresh_canvas()
        current_entry = self.entries[row]
        self.statusBar().showMessage(
            f"Viewing {current_entry.name} ({row + 1}/{len(self.entries)})"
        )

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

    def _update_paint_values(self) -> None:
        source = self.source_combo.currentData()
        target = self.target_combo.currentData()
        self.canvas.set_paint_values(source, target)

    def _auto_populate_classes(self) -> None:
        if not self.entries:
            QMessageBox.information(self, "No dataset", "Load a dataset before auto detecting classes.")
            return
        values: List[int] = []
        for entry in self.entries:
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
        entry.reset_edits()
        self._session_dirty = True
        self._refresh_canvas()
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
        for entry in self.entries:
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
        self.statusBar().showMessage(f"Exported {exported} label(s) to {dest_path}", 5000)

    # ----- Rendering --------------------------------------------------------
    def _refresh_canvas(self) -> None:
        if self.current_index is None:
            self.canvas.clear()
            return
        entry = self.entries[self.current_index]
        self.canvas.set_label_array(entry.edited_label)
        classes = self.class_manager.get_classes()
        pixmap = self._render_overlay(
            entry.image,
            entry.edited_label,
            classes,
            self.alpha_slider.value() / 100.0,
        )
        self.canvas.set_pixmap(pixmap)
        self.canvas.viewport().update()

    def _render_overlay(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        classes: Sequence[ClassDefinition],
        alpha: float,
    ) -> QPixmap:
        image_rgb = np.asarray(image, dtype=np.float32)
        if image_rgb.ndim == 2:
            image_rgb = np.stack([image_rgb] * 3, axis=-1)
        color_map: Dict[int, Tuple[int, int, int]] = {cls.value: cls.color_tuple() for cls in classes}
        label_values = np.asarray(labels, dtype=np.int32)
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

        blend_alpha = np.clip(alpha, 0.0, 1.0)
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
