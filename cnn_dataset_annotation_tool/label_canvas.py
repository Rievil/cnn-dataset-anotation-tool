from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QWidget

from .models import EditOperation

class ToolMode(Enum):
    BRUSH = "brush"
    LASSO = "lasso"
    MAGNETIC_LASSO = "magnetic_lasso"
    POLYGON = "polygon"
    POLYLINE = "polyline"
    MEASURE = "measure"


class LabelCanvas(QGraphicsView):
    """Interactive canvas visualizing the image/label overlay."""

    MAX_ZOOM = 64.0          # hard ceiling: one image pixel spans 64 screen pixels
    MIN_FIT_FRACTION = 0.5   # floor: half of the whole-image fit-to-viewport scale

    # one color per annotator, assigned by sorted name order ("" first keeps the default green)
    MEASURE_COLORS = [
        (0, 230, 130),
        (80, 180, 255),
        (255, 170, 40),
        (255, 90, 220),
        (255, 235, 60),
        (170, 120, 255),
    ]

    labelEdited = Signal()
    operationPerformed = Signal(object)
    polylineWidthChanged = Signal(int)
    brushRadiusChanged = Signal(int)
    measurementsChanged = Signal()

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
        self._active_source: Optional[int] = None
        self._active_target: Optional[int] = None
        self._painting_button: Optional[Qt.MouseButton] = None
        self._last_paint_point: Optional[QPointF] = None
        self._label_array: Optional[np.ndarray] = None
        self._image_size: Tuple[int, int] = (0, 0)
        self._source_value: Optional[int] = None
        self._target_value: Optional[int] = None
        self._tool_mode = ToolMode.BRUSH
        self._lasso_points: List[QPointF] = []
        self._lasso_active = False
        self._base_image: Optional[np.ndarray] = None
        self._gradient_map: Optional[np.ndarray] = None
        self._lasso_snap_radius = 8
        self._lasso_min_distance = 2.0
        self._lasso_start_hover = False
        self._lasso_start_screen_radius = 6.0
        self._lasso_start_hover_margin = 2.0
        self._pending_operation_desc: Optional[str] = None
        self._pending_operation_pixels: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._polyline_width = 5
        self._brush_preview_mask: Optional[np.ndarray] = None
        self._brush_preview_image: Optional[QImage] = None
        self._brush_preview_dirty = False
        self._brush_preview_color = QColor(255, 255, 255, 160)
        self._measurements: List[Dict[str, float]] = []
        self._measure_start: Optional[QPointF] = None
        self._show_measure_labels = True
        self._current_annotator = ""

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
        self._clear_brush_preview()
        self._image_size = (0, 0)
        self._base_image = None
        self._gradient_map = None
        self._cancel_lasso()
        self._measurements = []
        self._measure_start = None
        self.viewport().update()

    def set_measurements(self, measurements: Optional[List[Dict[str, float]]]) -> None:
        """Replace the width measurements shown for the current item (no signal emitted)."""
        self._measurements = [dict(m) for m in (measurements or [])]
        self._measure_start = None
        self.viewport().update()

    def measurements(self) -> List[Dict[str, float]]:
        return [dict(m) for m in self._measurements]

    def set_measure_labels_visible(self, visible: bool) -> None:
        if visible == self._show_measure_labels:
            return
        self._show_measure_labels = visible
        self.viewport().update()

    def set_annotator(self, name: str) -> None:
        self._current_annotator = str(name).strip()
        self.viewport().update()

    def set_label_array(self, array: Optional[np.ndarray]) -> None:
        self._label_array = array
        self._clear_brush_preview()

    def set_base_image(self, image: Optional[np.ndarray]) -> None:
        if image is None:
            self._base_image = None
            self._gradient_map = None
            return
        arr = np.asarray(image, dtype=np.float32, order="C")
        if arr.ndim == 3:
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        else:
            gray = arr
        gx, gy = np.gradient(gray)
        self._base_image = arr
        self._gradient_map = np.hypot(gx, gy)

    def set_brush_radius(self, radius: int) -> None:
        clamped = max(1, min(200, int(radius)))
        if clamped == self._brush_radius:
            return
        self._brush_radius = clamped
        self.brushRadiusChanged.emit(self._brush_radius)
        self.viewport().update()

    def set_polyline_width(self, width: int) -> None:
        clamped = max(1, min(200, int(width)))
        if clamped == self._polyline_width:
            return
        self._polyline_width = clamped
        self.polylineWidthChanged.emit(self._polyline_width)
        self.viewport().update()

    def polyline_width(self) -> int:
        return self._polyline_width

    def set_paint_values(self, source_value: Optional[int], target_value: Optional[int]) -> None:
        self._source_value = source_value
        self._target_value = target_value

    def set_tool_mode(self, mode: ToolMode) -> None:
        if mode == self._tool_mode:
            return
        self._tool_mode = mode
        self._cancel_lasso()
        self._measure_start = None
        self._hover_pos = None
        self._clear_brush_preview()
        if self._tool_mode == ToolMode.POLYLINE:
            self.polylineWidthChanged.emit(self._polyline_width)
        self.viewport().update()

    def _format_value(self, value: Optional[int]) -> str:
        return "any" if value is None else str(int(value))

    def _operation_base_name(self) -> str:
        if self._tool_mode == ToolMode.LASSO:
            return "Freehand Lasso"
        if self._tool_mode == ToolMode.MAGNETIC_LASSO:
            return "Magnetic Lasso"
        if self._tool_mode == ToolMode.POLYGON:
            return "Polygon"
        if self._tool_mode == ToolMode.POLYLINE:
            return "Polygon Line"
        return "Brush"

    def _begin_operation(self, description: str) -> None:
        self._pending_operation_desc = description
        self._pending_operation_pixels = {}

    def _record_pixel_change(self, row: int, col: int, previous: int, new: int) -> None:
        if self._pending_operation_desc is None:
            return
        coord = (int(row), int(col))
        if coord not in self._pending_operation_pixels:
            self._pending_operation_pixels[coord] = (int(previous), int(new))
        else:
            prev = self._pending_operation_pixels[coord][0]
            self._pending_operation_pixels[coord] = (prev, int(new))

    def _emit_operation_from_pending(self) -> None:
        if not self._pending_operation_pixels or self._pending_operation_desc is None:
            self._clear_pending_operation()
            return
        ordered = list(self._pending_operation_pixels.items())
        coords = np.array([coord for coord, _ in ordered], dtype=np.int32)
        previous = np.array([values[0] for _, values in ordered], dtype=np.int32)
        new_values = np.array([values[1] for _, values in ordered], dtype=np.int32)
        operation = EditOperation(self._pending_operation_desc, coords, previous, new_values)
        self.operationPerformed.emit(operation)
        self._clear_pending_operation()

    def _emit_operation_from_arrays(
        self,
        description: str,
        coords: np.ndarray,
        previous: np.ndarray,
        new_values: np.ndarray,
    ) -> None:
        if coords.size == 0:
            return
        operation = EditOperation(
            description,
            np.array(coords, dtype=np.int32, copy=True),
            np.array(previous, dtype=np.int32, copy=True),
            np.array(new_values, dtype=np.int32, copy=True),
        )
        self.operationPerformed.emit(operation)

    def _clear_pending_operation(self) -> None:
        self._pending_operation_desc = None
        self._pending_operation_pixels = {}

    def _lasso_finalize_point(self) -> Optional[QPointF]:
        if not self._lasso_points:
            return None
        if self._tool_mode == ToolMode.POLYLINE:
            return self._lasso_points[-1]
        return self._lasso_points[0]

    def _wheel_steps(self, event) -> int:
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.angleDelta().x()
        steps = delta // 120 if delta else 0
        if steps == 0 and delta:
            steps = 1 if delta > 0 else -1
        return int(steps)

    def _adjust_polyline_width(self, delta: int) -> None:
        if delta == 0:
            return
        self.set_polyline_width(self._polyline_width + delta)

    def _adjust_brush_radius(self, delta: int) -> None:
        if delta == 0:
            return
        self.set_brush_radius(self._brush_radius + delta)

    def _initialize_brush_preview(self, reverse: bool) -> None:
        if self._label_array is None:
            self._clear_brush_preview()
            return
        self._brush_preview_mask = np.zeros(self._label_array.shape, dtype=bool)
        self._brush_preview_image = None
        self._brush_preview_dirty = False
        self._brush_preview_color = QColor(255, 140, 0, 160) if reverse else QColor(255, 255, 255, 160)

    def _clear_brush_preview(self) -> None:
        self._brush_preview_mask = None
        self._brush_preview_image = None
        self._brush_preview_dirty = False

    def _mark_brush_preview_pixel(self, row: int, col: int) -> None:
        if self._brush_preview_mask is None:
            return
        if (
            row < 0
            or col < 0
            or row >= self._brush_preview_mask.shape[0]
            or col >= self._brush_preview_mask.shape[1]
        ):
            return
        self._brush_preview_mask[row, col] = True
        self._brush_preview_dirty = True

    def _current_brush_preview_image(self) -> Optional[QImage]:
        if self._brush_preview_mask is None or not np.any(self._brush_preview_mask):
            return None
        if self._brush_preview_image is None or self._brush_preview_dirty:
            self._rebuild_brush_preview_image()
        return self._brush_preview_image

    def _rebuild_brush_preview_image(self) -> None:
        if (
            self._brush_preview_mask is None
            or self._label_array is None
            or not np.any(self._brush_preview_mask)
        ):
            self._brush_preview_image = None
            self._brush_preview_dirty = False
            return
        mask = self._brush_preview_mask
        height, width = mask.shape
        image = QImage(width, height, QImage.Format_ARGB32)
        image.fill(0)
        ptr = image.bits()
        total_bytes = image.sizeInBytes()
        if hasattr(ptr, "setsize"):
            ptr.setsize(total_bytes)
        buffer = np.frombuffer(ptr, dtype=np.uint8, count=total_bytes)
        array = buffer.reshape((height, image.bytesPerLine()))
        rgba = array[:, : width * 4].reshape((height, width, 4))
        color = self._brush_preview_color if self._brush_preview_color is not None else QColor(255, 255, 255, 160)
        rgba[..., 0][mask] = color.blue()
        rgba[..., 1][mask] = color.green()
        rgba[..., 2][mask] = color.red()
        rgba[..., 3][mask] = color.alpha()
        self._brush_preview_image = image
        self._brush_preview_dirty = False

    def _commit_pending_brush_changes(self) -> bool:
        if self._label_array is None or not self._pending_operation_pixels:
            return False
        ordered = list(self._pending_operation_pixels.items())
        if not ordered:
            return False
        coords = np.array([coord for coord, _ in ordered], dtype=np.int32)
        rows = coords[:, 0]
        cols = coords[:, 1]
        new_values = np.asarray(
            [values[1] for _, values in ordered],
            dtype=self._label_array.dtype,
        )
        self._label_array[rows, cols] = new_values
        return True

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self._tool_mode == ToolMode.BRUSH and not (event.modifiers() & Qt.ControlModifier):
            steps = self._wheel_steps(event)
            if steps:
                self._adjust_brush_radius(int(steps))
                event.accept()
                return
        if self._tool_mode == ToolMode.POLYLINE and self._lasso_active and self._lasso_points:
            steps = self._wheel_steps(event)
            if steps:
                self._adjust_polyline_width(int(steps))
            event.accept()
            return
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.25 if event.angleDelta().y() > 0 else 0.8
            self._apply_zoom(factor)
            return
        super().wheelEvent(event)

    def _fit_scale(self) -> float:
        """Scale at which the whole image fits the viewport."""
        width, height = self._image_size
        viewport = self.viewport()
        if width <= 0 or height <= 0 or viewport.width() <= 0 or viewport.height() <= 0:
            return 1.0
        return min(viewport.width() / width, viewport.height() / height)

    def _apply_zoom(self, factor: float) -> None:
        current = float(self.transform().m11())
        if current <= 0.0 or self._image_size == (0, 0):
            return
        # Never allow zooming out past half the fit-to-view scale (capped at 1:1 so
        # small images always reach 100%), nor zooming in past MAX_ZOOM.
        minimum = min(self._fit_scale() * self.MIN_FIT_FRACTION, 1.0)
        target = max(minimum, min(current * factor, self.MAX_ZOOM))
        effective = target / current
        if abs(effective - 1.0) < 1e-6:
            return
        self.scale(effective, effective)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._tool_mode == ToolMode.BRUSH:
            if event.button() in (Qt.LeftButton, Qt.RightButton):
                if self._label_array is None:
                    return
                reverse = event.button() == Qt.RightButton
                if not reverse:
                    active_source = self._source_value
                    active_target = self._target_value
                else:
                    if self._source_value is None or self._target_value is None:
                        return
                    active_source = self._target_value
                    active_target = self._source_value
                if active_target is None:
                    return
                scene_pos = self.mapToScene(event.position().toPoint())
                if not self._within_image(scene_pos):
                    return
                self._initialize_brush_preview(reverse)
                description = (
                    f"Brush {self._format_value(active_source)}→{self._format_value(active_target)}"
                )
                self._begin_operation(description)
                changed = self._apply_brush(scene_pos, active_source, active_target)
                if changed:
                    self.viewport().update()
                self._painting = True
                self._active_source = active_source
                self._active_target = active_target
                self._painting_button = event.button()
                self._last_paint_point = scene_pos
                event.accept()
                return
        elif self._tool_mode == ToolMode.POLYGON:
            if self._handle_polygon_press(event):
                return
        elif self._tool_mode == ToolMode.POLYLINE:
            if self._handle_polyline_press(event):
                return
        elif self._tool_mode == ToolMode.MEASURE:
            if self._handle_measure_press(event):
                return
        else:
            scene_pos = self.mapToScene(event.position().toPoint())
            if self._lasso_active:
                if event.button() in (Qt.LeftButton, Qt.RightButton):
                    if self._is_near_lasso_start(scene_pos):
                        if event.button() == Qt.LeftButton:
                            if self._target_value is None:
                                event.accept()
                                return
                            self._active_source = self._source_value
                            self._active_target = self._target_value
                        else:
                            if self._source_value is None or self._target_value is None:
                                event.accept()
                                return
                            self._active_source = self._target_value
                            self._active_target = self._source_value
                        source_display = self._format_value(self._active_source)
                        target_display = self._format_value(self._active_target)
                        is_reverse = event.button() == Qt.RightButton
                        result = self._finish_lasso()
                        if result is not None:
                            coords, previous_values, new_values = result
                            description = (
                                f"{self._operation_base_name()} {source_display}→{target_display}"
                            )
                            if is_reverse:
                                description += " (reverse)"
                            self.labelEdited.emit()
                            self._emit_operation_from_arrays(description, coords, previous_values, new_values)
                        event.accept()
                        return
                    self._cancel_lasso()
                    self.viewport().update()
                    event.accept()
                    return
                if event.button() == Qt.RightButton:
                    self._cancel_lasso()
                    self.viewport().update()
                    event.accept()
                    return
            if event.button() == Qt.LeftButton:
                if self._label_array is None or self._target_value is None:
                    return
                if not self._within_image(scene_pos):
                    return
                self._begin_lasso(scene_pos)
                event.accept()
                return
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def _handle_polygon_press(self, event) -> bool:
        if event.button() == Qt.MiddleButton:
            return False
        scene_pos = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.LeftButton:
            if self._label_array is None or self._target_value is None:
                event.accept()
                return True
            if not self._within_image(scene_pos):
                event.accept()
                return True
            if not self._lasso_active:
                self._begin_lasso(scene_pos)
                event.accept()
                return True
            if self._is_near_lasso_start(scene_pos):
                self._active_source = self._source_value
                self._active_target = self._target_value
                if self._active_target is None:
                    self._cancel_lasso()
                    self.viewport().update()
                    event.accept()
                    return True
                changed = self._finish_lasso()
                if changed:
                    self.labelEdited.emit()
                event.accept()
                return True
            self._append_lasso_point(scene_pos)
            self.viewport().update()
            event.accept()
            return True
        if event.button() == Qt.RightButton:
            if not self._lasso_active:
                event.accept()
                return True
            if self._is_near_lasso_start(scene_pos):
                if self._source_value is None or self._target_value is None:
                    self._cancel_lasso()
                    self.viewport().update()
                    event.accept()
                    return True
                self._active_source = self._target_value
                self._active_target = self._source_value
                changed = self._finish_lasso()
                if changed:
                    self.labelEdited.emit()
                event.accept()
                return True
            self._cancel_lasso()
            self.viewport().update()
            event.accept()
            return True
        return False

    def _handle_polyline_press(self, event) -> bool:
        if event.button() == Qt.MiddleButton:
            return False
        scene_pos = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.LeftButton:
            if self._label_array is None or self._target_value is None:
                event.accept()
                return True
            if not self._within_image(scene_pos):
                event.accept()
                return True
            if not self._lasso_active:
                self._begin_lasso(scene_pos)
                event.accept()
                return True
            if self._is_near_lasso_start(scene_pos) and len(self._lasso_points) >= 2:
                self._active_source = self._source_value
                self._active_target = self._target_value
                source_value = self._active_source
                target_value = self._active_target
                result = self._finish_lasso()
                if result is not None and target_value is not None:
                    coords, previous_values, new_values = result
                    description = (
                        f"{self._operation_base_name()} "
                        f"{self._format_value(source_value)}→{self._format_value(target_value)}"
                    )
                    self.labelEdited.emit()
                    self._emit_operation_from_arrays(description, coords, previous_values, new_values)
                event.accept()
                return True
            self._append_lasso_point(scene_pos)
            self.viewport().update()
            event.accept()
            return True
        if event.button() == Qt.RightButton:
            if not self._lasso_active:
                event.accept()
                return True
            if self._is_near_lasso_start(scene_pos) and len(self._lasso_points) >= 2:
                if self._source_value is None or self._target_value is None:
                    self._cancel_lasso()
                    self.viewport().update()
                    event.accept()
                    return True
                self._active_source = self._target_value
                self._active_target = self._source_value
                source_value = self._active_source
                target_value = self._active_target
                result = self._finish_lasso()
                if result is not None and target_value is not None:
                    coords, previous_values, new_values = result
                    description = (
                        f"{self._operation_base_name()} "
                        f"{self._format_value(source_value)}→{self._format_value(target_value)} (reverse)"
                    )
                    self.labelEdited.emit()
                    self._emit_operation_from_arrays(description, coords, previous_values, new_values)
                event.accept()
                return True
            self._cancel_lasso()
            self.viewport().update()
            event.accept()
            return True
        return False

    def _handle_measure_press(self, event) -> bool:
        if event.button() == Qt.MiddleButton:
            return False
        scene_pos = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.LeftButton:
            if not self._within_image(scene_pos):
                event.accept()
                return True
            if self._measure_start is None:
                self._measure_start = QPointF(scene_pos)
            else:
                start = self._measure_start
                length = math.hypot(scene_pos.x() - start.x(), scene_pos.y() - start.y())
                if length >= 0.5:
                    item: Dict[str, object] = {
                        "x1": float(start.x()),
                        "y1": float(start.y()),
                        "x2": float(scene_pos.x()),
                        "y2": float(scene_pos.y()),
                    }
                    if self._current_annotator:
                        item["annotator"] = self._current_annotator
                    self._measurements.append(item)
                    self.measurementsChanged.emit()
                self._measure_start = None
            self.viewport().update()
            event.accept()
            return True
        if event.button() == Qt.RightButton:
            if self._measure_start is not None:
                self._measure_start = None
            else:
                index = self._measurement_at(scene_pos)
                if index is not None:
                    del self._measurements[index]
                    self.measurementsChanged.emit()
            self.viewport().update()
            event.accept()
            return True
        return False

    def _measurement_at(self, point: QPointF) -> Optional[int]:
        radius = max(6.0 * self._scene_unit(), 2.0)
        best_index: Optional[int] = None
        best_distance = radius
        for index, item in enumerate(self._measurements):
            distance = self._point_segment_distance(
                point.x(), point.y(), item["x1"], item["y1"], item["x2"], item["y2"]
            )
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index

    @staticmethod
    def _point_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq <= 0.0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        scene_pos = self.mapToScene(event.position().toPoint())
        self._hover_pos = scene_pos if self._within_image(scene_pos) else None
        if self._tool_mode == ToolMode.BRUSH:
            if self._painting and self._label_array is not None and self._active_target is not None:
                if self._last_paint_point is not None:
                    self._apply_brush_line(
                        self._last_paint_point,
                        scene_pos,
                        self._active_source,
                        self._active_target,
                    )
                self._last_paint_point = scene_pos
        elif (
            self._tool_mode in (ToolMode.LASSO, ToolMode.MAGNETIC_LASSO)
            and self._lasso_active
            and (event.buttons() & Qt.LeftButton)
        ):
            self._append_lasso_point(scene_pos)
            event.accept()
        self._update_lasso_start_hover()
        self.viewport().update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._tool_mode == ToolMode.BRUSH:
            if event.button() in (Qt.LeftButton, Qt.RightButton):
                committed = self._commit_pending_brush_changes()
                self._emit_operation_from_pending()
                if committed:
                    self.labelEdited.emit()
                self._painting = False
                self._painting_button = None
                self._active_source = None
                self._active_target = None
                self._last_paint_point = None
                self._clear_brush_preview()
                self.viewport().update()
                event.accept()
                return
        else:
            if self._lasso_active and event.button() in (Qt.LeftButton, Qt.RightButton):
                event.accept()
                return
        if event.button() == Qt.MiddleButton:
            super().mouseReleaseEvent(event)
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hover_pos = None
        self._update_lasso_start_hover()
        self.viewport().update()
        super().leaveEvent(event)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # type: ignore[override]
        super().drawForeground(painter, rect)
        if self._image_size == (0, 0):
            return
        inv_transform = self.transform().inverted()[0]
        unit = inv_transform.mapRect(QRectF(0, 0, 1, 1)).width()
        if (
            self._tool_mode
            in (ToolMode.LASSO, ToolMode.MAGNETIC_LASSO, ToolMode.POLYGON, ToolMode.POLYLINE)
            and self._lasso_points
        ):
            painter.save()
            polyline_pen_width: Optional[float] = None
            if self._tool_mode == ToolMode.POLYLINE:
                pen_width = max(unit, self._polyline_width * unit)
                polyline_pen_width = pen_width
                stroke_pen = QPen(
                    QColor(255, 255, 255, 200),
                    pen_width,
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin,
                )
                painter.setPen(stroke_pen)
                painter.setBrush(Qt.NoBrush)
                if len(self._lasso_points) == 1:
                    radius = max(pen_width * 0.5, unit)
                    painter.drawEllipse(self._lasso_points[0], radius, radius)
                else:
                    painter.drawPolyline(QPolygonF(self._lasso_points))
                if self._lasso_active and self._hover_pos is not None:
                    preview_pen = QPen(
                        QColor(200, 200, 200, 160),
                        pen_width,
                        Qt.DashLine,
                        Qt.RoundCap,
                        Qt.RoundJoin,
                    )
                    painter.setPen(preview_pen)
                    painter.drawLine(self._lasso_points[-1], self._hover_pos)
            else:
                path = QPolygonF(self._lasso_points)
                pen = QPen(QColor(255, 255, 255, 200), max(1.0, unit))
                painter.setPen(pen)
                painter.setBrush(QColor(255, 255, 255, 40) if self._lasso_active else Qt.NoBrush)
                painter.drawPolygon(path)
                if self._tool_mode == ToolMode.POLYGON and self._lasso_active and self._hover_pos is not None:
                    preview_pen = QPen(QColor(200, 200, 200, 160), max(1.0, unit), Qt.DashLine)
                    painter.setPen(preview_pen)
                    painter.drawLine(self._lasso_points[-1], self._hover_pos)
            if self._lasso_active:
                finalize_point = self._lasso_finalize_point()
                if finalize_point is not None:
                    start_radius = max(self._lasso_start_screen_radius * unit, 3.0 * unit)
                    if polyline_pen_width is not None:
                        start_radius = max(start_radius, polyline_pen_width * 0.6)
                    hover_color = QColor(160, 160, 160, 230)
                    base_color = QColor(255, 255, 255, 230)
                    marker_color = hover_color if self._lasso_start_hover else base_color
                    marker_pen = QPen(QColor(30, 30, 30, 220), max(1.0, unit))
                    painter.setPen(marker_pen)
                    painter.setBrush(marker_color)
                    painter.drawEllipse(finalize_point, start_radius, start_radius)
            painter.restore()
        if self._tool_mode == ToolMode.BRUSH:
            preview_image = self._current_brush_preview_image()
            if preview_image is not None:
                painter.save()
                painter.drawImage(QPointF(0, 0), preview_image)
                painter.restore()
        if self._tool_mode == ToolMode.BRUSH and self._hover_pos is not None and self._brush_radius > 0:
            painter.save()
            radius = self._brush_radius
            pen = QPen(QColor(255, 255, 255, 180), max(1.0, unit))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self._hover_pos, radius, radius)
            painter.restore()
        if self._tool_mode == ToolMode.MEASURE:
            self._draw_measurements(painter, unit)

    def _annotator_colors(self) -> Dict[str, Tuple[int, int, int]]:
        names = {str(item.get("annotator", "")) for item in self._measurements}
        names.add(self._current_annotator)
        return {
            name: self.MEASURE_COLORS[index % len(self.MEASURE_COLORS)]
            for index, name in enumerate(sorted(names))
        }

    def _draw_measurements(self, painter: QPainter, unit: float) -> None:
        painter.save()
        colors = self._annotator_colors()
        pen_width = max(1.2 * unit, unit)
        tick_half = max(3.0 * unit, 1.5)
        for item in self._measurements:
            rgb = colors[str(item.get("annotator", ""))]
            painter.setPen(QPen(QColor(*rgb, 235), pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            start = QPointF(item["x1"], item["y1"])
            end = QPointF(item["x2"], item["y2"])
            painter.drawLine(start, end)
            self._draw_measure_ticks(painter, start, end, tick_half)
            if self._show_measure_labels:
                length = math.hypot(end.x() - start.x(), end.y() - start.y())
                mid = QPointF((start.x() + end.x()) * 0.5, (start.y() + end.y()) * 0.5)
                self._draw_measure_label(painter, mid, f"{length:.1f} px", unit, rgb)
        if self._measure_start is not None:
            active_rgb = colors[self._current_annotator]
            marker_pen = QPen(QColor(*active_rgb, 235), pen_width)
            painter.setPen(marker_pen)
            cross = max(4.0 * unit, 2.0)
            painter.drawLine(
                QPointF(self._measure_start.x() - cross, self._measure_start.y()),
                QPointF(self._measure_start.x() + cross, self._measure_start.y()),
            )
            painter.drawLine(
                QPointF(self._measure_start.x(), self._measure_start.y() - cross),
                QPointF(self._measure_start.x(), self._measure_start.y() + cross),
            )
            if self._hover_pos is not None:
                preview_pen = QPen(QColor(*active_rgb, 170), pen_width, Qt.DashLine)
                painter.setPen(preview_pen)
                painter.drawLine(self._measure_start, self._hover_pos)
                length = math.hypot(
                    self._hover_pos.x() - self._measure_start.x(),
                    self._hover_pos.y() - self._measure_start.y(),
                )
                self._draw_measure_label(painter, self._hover_pos, f"{length:.1f} px", unit, active_rgb)
        painter.restore()

    @staticmethod
    def _draw_measure_ticks(painter: QPainter, start: QPointF, end: QPointF, half_length: float) -> None:
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        norm = math.hypot(dx, dy)
        if norm <= 0.0:
            return
        # Perpendicular unit vector marks the two crack edges the user clicked.
        ux = -dy / norm
        uy = dx / norm
        for point in (start, end):
            painter.drawLine(
                QPointF(point.x() - ux * half_length, point.y() - uy * half_length),
                QPointF(point.x() + ux * half_length, point.y() + uy * half_length),
            )

    def _draw_measure_label(
        self,
        painter: QPainter,
        anchor: QPointF,
        text: str,
        unit: float,
        rgb: Tuple[int, int, int] = (0, 255, 150),
    ) -> None:
        font = QFont(painter.font())
        font.setPixelSize(max(int(round(11.0 * unit)), 2))
        font.setBold(True)
        painter.setFont(font)
        offset = 5.0 * unit
        position = QPointF(anchor.x() + offset, anchor.y() - offset)
        halo = max(1.0 * unit, 0.5)
        painter.setPen(QPen(QColor(20, 20, 20, 220)))
        for dx, dy in ((-halo, 0.0), (halo, 0.0), (0.0, -halo), (0.0, halo)):
            painter.drawText(QPointF(position.x() + dx, position.y() + dy), text)
        painter.setPen(QPen(QColor(*rgb, 255)))
        painter.drawText(position, text)

    def _within_image(self, point: QPointF) -> bool:
        width, height = self._image_size
        return 0 <= point.x() < width and 0 <= point.y() < height

    def _apply_brush_line(
        self,
        start: QPointF,
        end: QPointF,
        source_value: Optional[int],
        target_value: Optional[int],
    ) -> bool:
        distance = math.hypot(end.x() - start.x(), end.y() - start.y())
        steps = max(1, int(distance))
        changed = False
        for i in range(steps + 1):
            t = i / steps if steps else 0
            x = start.x() + (end.x() - start.x()) * t
            y = start.y() + (end.y() - start.y()) * t
            if self._apply_brush(QPointF(x, y), source_value, target_value):
                changed = True
        return changed

    def _begin_lasso(self, point: QPointF) -> None:
        snapped = self._snap_to_edge(point)
        self._lasso_points = [snapped]
        self._lasso_active = True
        self._active_source = self._source_value
        self._active_target = self._target_value
        self._lasso_start_hover = False

    def _append_lasso_point(self, point: QPointF) -> None:
        snapped = self._snap_to_edge(point)
        if not self._within_image(snapped):
            return
        if not self._lasso_points:
            self._lasso_points.append(snapped)
            return
        last = self._lasso_points[-1]
        dx = snapped.x() - last.x()
        dy = snapped.y() - last.y()
        if dx * dx + dy * dy >= self._lasso_min_distance * self._lasso_min_distance:
            self._lasso_points.append(snapped)
        else:
            self._lasso_points[-1] = snapped

    def _finish_lasso(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if (
            not self._lasso_active
            or not self._lasso_points
            or self._active_target is None
            or self._label_array is None
        ):
            self._cancel_lasso()
            return None
        min_points = 2 if self._tool_mode == ToolMode.POLYLINE else 3
        if len(self._lasso_points) < min_points:
            self._cancel_lasso()
            return None
        if self._tool_mode == ToolMode.POLYLINE:
            result = self._apply_polyline(
                self._lasso_points,
                self._polyline_width,
                self._active_source,
                self._active_target,
            )
        else:
            result = self._apply_polygon(
                self._lasso_points,
                self._active_source,
                self._active_target,
            )
        self._cancel_lasso()
        self.viewport().update()
        return result

    def _scene_unit(self) -> float:
        transform = self.transform()
        inverted, invertible = transform.inverted()
        if not invertible:
            return 1.0
        return inverted.mapRect(QRectF(0, 0, 1, 1)).width()

    def _lasso_hover_radius_scene(self) -> float:
        unit = self._scene_unit()
        base_radius = (self._lasso_start_screen_radius + self._lasso_start_hover_margin) * unit
        if self._tool_mode == ToolMode.POLYLINE:
            base_radius = max(base_radius, (self._polyline_width * 0.6 + self._lasso_start_hover_margin) * unit)
        return max(base_radius, 3.0 * unit)

    def _is_near_lasso_start(self, point: QPointF) -> bool:
        if (
            self._tool_mode
            not in (ToolMode.LASSO, ToolMode.MAGNETIC_LASSO, ToolMode.POLYGON, ToolMode.POLYLINE)
            or not self._lasso_points
        ):
            return False
        finalize_point = self._lasso_finalize_point()
        if finalize_point is None:
            return False
        radius = self._lasso_hover_radius_scene()
        dx = point.x() - finalize_point.x()
        dy = point.y() - finalize_point.y()
        return dx * dx + dy * dy <= radius * radius

    def _update_lasso_start_hover(self) -> None:
        hovering = False
        if (
            self._tool_mode
            in (ToolMode.LASSO, ToolMode.MAGNETIC_LASSO, ToolMode.POLYGON, ToolMode.POLYLINE)
            and self._lasso_points
            and self._hover_pos is not None
        ):
            finalize_point = self._lasso_finalize_point()
            if finalize_point is not None:
                start = finalize_point
                radius = self._lasso_hover_radius_scene()
                dx = self._hover_pos.x() - start.x()
                dy = self._hover_pos.y() - start.y()
                hovering = dx * dx + dy * dy <= radius * radius
        if hovering != self._lasso_start_hover:
            self._lasso_start_hover = hovering
            self.viewport().update()

    def _apply_polygon(
        self,
        points: List[QPointF],
        source_value: Optional[int],
        target_value: Optional[int],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._label_array is None or target_value is None or not points:
            return None
        height, width = self._label_array.shape
        if width == 0 or height == 0:
            return None
        mask_image = QImage(width, height, QImage.Format_Grayscale8)
        mask_image.fill(0)
        painter = QPainter(mask_image)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 255))
        painter.drawPolygon(QPolygonF(points))
        painter.end()
        ptr = mask_image.bits()
        total_bytes = mask_image.height() * mask_image.bytesPerLine()
        if hasattr(ptr, "setsize"):
            ptr.setsize(total_bytes)
            buffer = np.frombuffer(ptr, dtype=np.uint8)
        else:
            buffer = np.frombuffer(ptr, dtype=np.uint8, count=total_bytes)
        mask_buffer = buffer.reshape(mask_image.height(), mask_image.bytesPerLine())
        mask = mask_buffer[:, :width] > 0
        if source_value is not None:
            selection = np.logical_and(mask, self._label_array == source_value)
        else:
            selection = mask
        if not np.any(selection):
            return None
        different = np.logical_and(selection, self._label_array != target_value)
        if not np.any(different):
            return None
        coords = np.argwhere(different)
        previous_values = self._label_array[different].astype(np.int32, copy=True)
        self._label_array[different] = target_value
        new_values = np.full(previous_values.shape, int(target_value), dtype=np.int32)
        return coords, previous_values, new_values

    def _apply_polyline(
        self,
        points: List[QPointF],
        thickness: int,
        source_value: Optional[int],
        target_value: Optional[int],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._label_array is None or target_value is None or len(points) < 2:
            return None
        height, width = self._label_array.shape
        if width == 0 or height == 0:
            return None
        mask_image = QImage(width, height, QImage.Format_Grayscale8)
        mask_image.fill(0)
        painter = QPainter(mask_image)
        pen = QPen(QColor(255, 255, 255, 255), float(max(1, thickness)), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPolyline(QPolygonF(points))
        painter.end()
        ptr = mask_image.bits()
        total_bytes = mask_image.height() * mask_image.bytesPerLine()
        if hasattr(ptr, "setsize"):
            ptr.setsize(total_bytes)
            buffer = np.frombuffer(ptr, dtype=np.uint8)
        else:
            buffer = np.frombuffer(ptr, dtype=np.uint8, count=total_bytes)
        mask_buffer = buffer.reshape(mask_image.height(), mask_image.bytesPerLine())
        mask = mask_buffer[:, :width] > 0
        if source_value is not None:
            selection = np.logical_and(mask, self._label_array == source_value)
        else:
            selection = mask
        if not np.any(selection):
            return None
        different = np.logical_and(selection, self._label_array != target_value)
        if not np.any(different):
            return None
        coords = np.argwhere(different)
        previous_values = self._label_array[different].astype(np.int32, copy=True)
        self._label_array[different] = target_value
        new_values = np.full(previous_values.shape, int(target_value), dtype=np.int32)
        return coords, previous_values, new_values

    def _snap_to_edge(self, point: QPointF) -> QPointF:
        if self._tool_mode != ToolMode.MAGNETIC_LASSO or self._gradient_map is None:
            return QPointF(point)
        x = int(round(point.x()))
        y = int(round(point.y()))
        height, width = self._gradient_map.shape
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        radius = self._lasso_snap_radius
        x_min = max(0, x - radius)
        x_max = min(width - 1, x + radius)
        y_min = max(0, y - radius)
        y_max = min(height - 1, y + radius)
        region = self._gradient_map[y_min : y_max + 1, x_min : x_max + 1]
        if region.size == 0:
            return QPointF(point)
        max_index = int(np.argmax(region))
        rel_y, rel_x = divmod(max_index, region.shape[1])
        snapped_x = x_min + rel_x
        snapped_y = y_min + rel_y
        return QPointF(float(snapped_x), float(snapped_y))

    def _cancel_lasso(self) -> None:
        self._lasso_points = []
        self._lasso_active = False
        self._active_source = None
        self._active_target = None
        self._painting_button = None
        self._last_paint_point = None
        self._painting = False
        self._lasso_start_hover = False
        self._clear_pending_operation()

    def _apply_brush(
        self,
        point: QPointF,
        source_value: Optional[int],
        target_value: Optional[int],
    ) -> bool:
        if self._label_array is None or target_value is None:
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
                if source_value is not None and current_value != source_value:
                    continue
                if current_value == target_value:
                    continue
                self._record_pixel_change(y, x, current_value, int(target_value))
                self._mark_brush_preview_pixel(y, x)
                changed = True
        return changed
