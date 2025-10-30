from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
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
    QGraphicsScene,
    QGraphicsView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
)


DATA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
LABEL_EXTENSIONS = {".png", ".tif", ".tiff"}

DEFAULT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


@dataclass
class DatasetEntry:
    """Bundle holding image, original label, and editable label."""

    name: str
    image_path: Path
    label_path: Path
    image: np.ndarray  # RGB uint8
    original_label: np.ndarray  # int32
    edited_label: np.ndarray  # int32

    def reset_edits(self) -> None:
        self.edited_label = self.original_label.copy()


@dataclass
class ClassDefinition:
    """Semantic class description."""

    name: str
    value: int
    color: QColor

    def color_tuple(self) -> Tuple[int, int, int]:
        return self.color.red(), self.color.green(), self.color.blue()


def load_rgb_image(path: Path) -> np.ndarray:
    """Load an RGB image as numpy array."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.array(rgb, dtype=np.uint8)


def load_label_image(path: Path) -> np.ndarray:
    """Load a single-channel label image."""
    with Image.open(path) as img:
        if img.mode not in ("L", "I;16", "I"):
            img = img.convert("L")
        array = np.array(img)
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.int32, copy=False)


def save_label_image(array: np.ndarray, path: Path) -> None:
    """Persist edited label image to disk."""
    array = np.asarray(array)
    if array.size == 0:
        raise ValueError("Cannot save empty label array")

    min_val = int(array.min())
    max_val = int(array.max())

    if min_val >= 0 and max_val <= 255:
        data = array.astype(np.uint8, copy=False)
    elif min_val >= 0 and max_val <= 65535:
        data = array.astype(np.uint16, copy=False)
    else:
        data = array.astype(np.int32, copy=False)

    Image.fromarray(data).save(path)


def collect_files(folder: Path, extensions: Sequence[str]) -> Dict[str, Path]:
    """Return mapping from base filename to path for allowed extensions."""
    result: Dict[str, Path] = {}
    for item in folder.iterdir():
        if item.is_file() and item.suffix.lower() in extensions:
            result[item.stem] = item
    return result


def fallback_color(value: int) -> Tuple[int, int, int]:
    """Deterministic fallback color when no class color is defined."""
    value = abs(int(value))
    r = (value * 37) % 256
    g = (value * 67 + 89) % 256
    b = (value * 97 + 53) % 256
    return r, g, b


def qcolor_from_hex(code: str) -> QColor:
    return QColor(code)


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


class QColorDialogWithParent:
    """Wrapper to shield against missing dialogs when no parent is set."""

    @staticmethod
    def get_color(parent: QWidget, initial: QColor) -> Optional[QColor]:
        from PySide6.QtWidgets import QColorDialog

        return QColorDialog.getColor(initial, parent, "Select Class Color")


class LabelCanvas(QGraphicsView):
    """Interactive canvas visualizing the image/label overlay."""

    labelEdited = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setBackgroundBrush(QColor("#1e1e1e"))
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = self._scene.addPixmap(QPixmap())
        self._pixmap_item.setZValue(0)
        self._hover_pos: Optional[QPointF] = None
        self._brush_radius = 15
        self._painting = False
        self._last_paint_point: Optional[QPointF] = None
        self._label_array: Optional[np.ndarray] = None
        self._image_size: Tuple[int, int] = (0, 0)
        self._source_value: Optional[int] = None
        self._target_value: Optional[int] = None

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap_item.setPixmap(pixmap)
        rect = QRectF(pixmap.rect())
        self._scene.setSceneRect(rect)
        self._image_size = (pixmap.width(), pixmap.height())
        self.viewport().update()

    def clear(self) -> None:
        self._pixmap_item.setPixmap(QPixmap())
        self._scene.setSceneRect(QRectF())
        self._label_array = None
        self._image_size = (0, 0)
        self.viewport().update()

    def set_label_array(self, array: Optional[np.ndarray]) -> None:
        self._label_array = array

    def set_brush_radius(self, radius: int) -> None:
        self._brush_radius = max(1, int(radius))
        self.viewport().update()

    def set_paint_values(self, source_value: Optional[int], target_value: Optional[int]) -> None:
        self._source_value = source_value
        self._target_value = target_value

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.25 if event.angleDelta().y() > 0 else 0.8
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            if self._target_value is None or self._label_array is None:
                return
            scene_pos = self.mapToScene(event.position().toPoint())
            if not self._within_image(scene_pos):
                return
            changed = self._apply_brush(scene_pos)
            if changed:
                self.labelEdited.emit()
            self._painting = True
            self._last_paint_point = scene_pos
            event.accept()
        elif event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        scene_pos = self.mapToScene(event.position().toPoint())
        self._hover_pos = scene_pos if self._within_image(scene_pos) else None
        if self._painting and self._label_array is not None and self._target_value is not None:
            if self._last_paint_point is not None:
                changed = self._apply_brush_line(self._last_paint_point, scene_pos)
                if changed:
                    self.labelEdited.emit()
            self._last_paint_point = scene_pos
        self.viewport().update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._painting = False
            self._last_paint_point = None
            event.accept()
        elif event.button() == Qt.MiddleButton:
            super().mouseReleaseEvent(event)
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hover_pos = None
        self.viewport().update()
        super().leaveEvent(event)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # type: ignore[override]
        super().drawForeground(painter, rect)
        if self._hover_pos is None or self._brush_radius <= 0:
            return
        if self._image_size == (0, 0):
            return
        painter.save()
        inv_transform = self.transform().inverted()[0]
        unit = inv_transform.mapRect(QRectF(0, 0, 1, 1)).width()
        radius = self._brush_radius
        pen = QPen(QColor(255, 255, 255, 180), max(1.0, unit))
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(self._hover_pos, radius, radius)
        painter.restore()

    def _within_image(self, point: QPointF) -> bool:
        width, height = self._image_size
        return 0 <= point.x() < width and 0 <= point.y() < height

    def _apply_brush_line(self, start: QPointF, end: QPointF) -> bool:
        distance = math.hypot(end.x() - start.x(), end.y() - start.y())
        steps = max(1, int(distance))
        changed = False
        for i in range(steps + 1):
            t = i / steps if steps else 0
            x = start.x() + (end.x() - start.x()) * t
            y = start.y() + (end.y() - start.y()) * t
            if self._apply_brush(QPointF(x, y)):
                changed = True
        return changed

    def _apply_brush(self, point: QPointF) -> bool:
        if self._label_array is None or self._target_value is None:
            return False
        x_center = int(round(point.x()))
        y_center = int(round(point.y()))
        height, width = self._label_array.shape
        radius = self._brush_radius
        x_min = max(0, x_center - radius)
        x_max = min(width - 1, x_center + radius)
        y_min = max(0, y_center - radius)
        y_max = min(height - 1, y_center + radius)
        if x_min > x_max or y_min > y_max:
            return False
        changed = False
        radius_sq = radius * radius
        for y in range(y_min, y_max + 1):
            dy = y - y_center
            for x in range(x_min, x_max + 1):
                dx = x - x_center
                if dx * dx + dy * dy > radius_sq:
                    continue
                current_value = int(self._label_array[y, x])
                if self._source_value is not None and current_value != self._source_value:
                    continue
                if current_value == self._target_value:
                    continue
                self._label_array[y, x] = self._target_value
                changed = True
        return changed


class MainWindow(QMainWindow):
    """Primary window orchestrating the workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CNN Dataset Annotation Tool")
        self.resize(1200, 800)
        self.entries: List[DatasetEntry] = []
        self.current_index: Optional[int] = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # Dataset controls
        dataset_row = QHBoxLayout()
        self.load_button = QPushButton("Load Dataset")
        self.revert_button = QPushButton("Revert Current Label")
        self.export_button = QPushButton("Export Edited Labels")
        dataset_row.addWidget(self.load_button)
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
        brush_hint = QLabel("Brush shape: circle. Hold Ctrl + mouse wheel to zoom, middle mouse to pan.")
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
        self.revert_button.clicked.connect(self.revert_current_label)
        self.export_button.clicked.connect(self.export_labels)
        self.image_list.currentRowChanged.connect(self.set_current_index)
        self.alpha_slider.valueChanged.connect(self._handle_alpha_changed)
        self.brush_slider.valueChanged.connect(self._handle_brush_slider_changed)
        self.brush_spin.valueChanged.connect(self._handle_brush_spin_changed)
        self.source_combo.currentIndexChanged.connect(self._update_paint_values)
        self.target_combo.currentIndexChanged.connect(self._update_paint_values)
        self.canvas.labelEdited.connect(self._refresh_canvas)
        self.class_manager.classesChanged.connect(self._handle_classes_changed)
        self.class_manager.autoPopulateRequested.connect(self._auto_populate_classes)

        self._update_paint_values()

    # ----- Dataset handling -------------------------------------------------
    def load_dataset(self) -> None:
        image_dir = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not image_dir:
            return
        label_dir = QFileDialog.getExistingDirectory(self, "Select label folder")
        if not label_dir:
            return

        images_folder = Path(image_dir)
        labels_folder = Path(label_dir)

        image_files = collect_files(images_folder, DATA_EXTENSIONS)
        label_files = collect_files(labels_folder, LABEL_EXTENSIONS)
        common_names = sorted(set(image_files).intersection(label_files))
        if not common_names:
            QMessageBox.warning(
                self,
                "No pairs found",
                "Could not find matching image/label filenames across the selected folders.",
            )
            return

        entries: List[DatasetEntry] = []
        errors: List[str] = []

        for name in common_names:
            image_path = image_files[name]
            label_path = label_files[name]
            try:
                image = load_rgb_image(image_path)
                label = load_label_image(label_path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
                continue
            if image.shape[:2] != label.shape:
                errors.append(f"{name}: image and label dimensions do not match")
                continue
            entries.append(
                DatasetEntry(
                    name=name,
                    image_path=image_path,
                    label_path=label_path,
                    image=image,
                    original_label=label,
                    edited_label=label.copy(),
                )
            )

        if not entries:
            QMessageBox.critical(
                self,
                "Load failed",
                "Failed to load any image/label pairs.\n" + "\n".join(errors[:5]),
            )
            return

        if errors:
            QMessageBox.information(
                self,
                "Partial load",
                "Some pairs were skipped:\n" + "\n".join(errors[:10]),
            )

        self.entries = entries
        self.image_list.clear()
        for entry in entries:
            item = QListWidgetItem(entry.name)
            self.image_list.addItem(item)
        self.dataset_status.setText(f"Loaded {len(entries)} pairs")
        self.set_current_index(0)
        self._auto_populate_classes()

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
    def revert_current_label(self) -> None:
        if self.current_index is None:
            return
        entry = self.entries[self.current_index]
        entry.reset_edits()
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
        pixmap = self._render_overlay(entry.image, entry.edited_label, classes, self.alpha_slider.value() / 100.0)
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
        label_values = labels.astype(np.int32, copy=False)
        color_overlay = np.zeros((*label_values.shape, 3), dtype=np.float32)
        unique_values = np.unique(label_values)
        for value in unique_values:
            color = color_map.get(int(value), fallback_color(int(value)))
            mask = label_values == value
            color_overlay[mask] = color
        blend_alpha = np.clip(alpha, 0.0, 1.0)
        blended = image_rgb * (1.0 - blend_alpha) + color_overlay * blend_alpha
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        height, width, _ = blended.shape
        bytes_per_line = 3 * width
        qimage = QImage(blended.data, width, height, bytes_per_line, QImage.Format_RGB888)
        qimage = qimage.copy()
        return QPixmap.fromImage(qimage)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
