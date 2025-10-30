from __future__ import annotations

import math
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QWidget

class ToolMode(Enum):
    BRUSH = "brush"
    LASSO = "lasso"
    MAGNETIC_LASSO = "magnetic_lasso"


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
        self._base_image = None
        self._gradient_map = None
        self._cancel_lasso()
        self.viewport().update()

    def set_label_array(self, array: Optional[np.ndarray]) -> None:
        self._label_array = array

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
        self._brush_radius = max(1, int(radius))
        self.viewport().update()

    def set_paint_values(self, source_value: Optional[int], target_value: Optional[int]) -> None:
        self._source_value = source_value
        self._target_value = target_value

    def set_tool_mode(self, mode: ToolMode) -> None:
        if mode == self._tool_mode:
            return
        self._tool_mode = mode
        self._cancel_lasso()
        self._hover_pos = None
        self.viewport().update()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.25 if event.angleDelta().y() > 0 else 0.8
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._tool_mode == ToolMode.BRUSH:
            if event.button() in (Qt.LeftButton, Qt.RightButton):
                if self._label_array is None:
                    return
                if event.button() == Qt.LeftButton:
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
                changed = self._apply_brush(scene_pos, active_source, active_target)
                if changed:
                    self.labelEdited.emit()
                self._painting = True
                self._active_source = active_source
                self._active_target = active_target
                self._painting_button = event.button()
                self._last_paint_point = scene_pos
                event.accept()
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
                        changed = self._finish_lasso()
                        if changed:
                            self.labelEdited.emit()
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

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        scene_pos = self.mapToScene(event.position().toPoint())
        self._hover_pos = scene_pos if self._within_image(scene_pos) else None
        if self._tool_mode == ToolMode.BRUSH:
            if self._painting and self._label_array is not None and self._active_target is not None:
                if self._last_paint_point is not None:
                    changed = self._apply_brush_line(
                        self._last_paint_point,
                        scene_pos,
                        self._active_source,
                        self._active_target,
                    )
                    if changed:
                        self.labelEdited.emit()
                self._last_paint_point = scene_pos
        elif self._lasso_active and (event.buttons() & Qt.LeftButton):
            self._append_lasso_point(scene_pos)
            event.accept()
        self._update_lasso_start_hover()
        self.viewport().update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._tool_mode == ToolMode.BRUSH:
            if event.button() in (Qt.LeftButton, Qt.RightButton):
                self._painting = False
                self._painting_button = None
                self._active_source = None
                self._active_target = None
                self._last_paint_point = None
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
        if self._tool_mode in (ToolMode.LASSO, ToolMode.MAGNETIC_LASSO) and self._lasso_points:
            painter.save()
            path = QPolygonF(self._lasso_points)
            pen = QPen(QColor(255, 255, 255, 200), max(1.0, unit))
            painter.setPen(pen)
            painter.setBrush(QColor(255, 255, 255, 40) if self._lasso_active else Qt.NoBrush)
            painter.drawPolygon(path)
            if self._lasso_active:
                start_point = self._lasso_points[0]
                start_radius = max(self._lasso_start_screen_radius * unit, 3.0 * unit)
                hover_color = QColor(160, 160, 160, 230)
                base_color = QColor(255, 255, 255, 230)
                marker_color = hover_color if self._lasso_start_hover else base_color
                marker_pen = QPen(QColor(30, 30, 30, 220), max(1.0, unit))
                painter.setPen(marker_pen)
                painter.setBrush(marker_color)
                painter.drawEllipse(start_point, start_radius, start_radius)
            painter.restore()
        if self._tool_mode == ToolMode.BRUSH and self._hover_pos is not None and self._brush_radius > 0:
            painter.save()
            radius = self._brush_radius
            pen = QPen(QColor(255, 255, 255, 180), max(1.0, unit))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self._hover_pos, radius, radius)
            painter.restore()

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

    def _finish_lasso(self) -> bool:
        if (
            not self._lasso_active
            or not self._lasso_points
            or self._active_target is None
            or self._label_array is None
        ):
            self._cancel_lasso()
            return False
        if len(self._lasso_points) < 3:
            self._cancel_lasso()
            return False
        changed = self._apply_polygon(
            self._lasso_points,
            self._active_source,
            self._active_target,
        )
        self._cancel_lasso()
        self.viewport().update()
        return changed

    def _scene_unit(self) -> float:
        transform = self.transform()
        inverted, invertible = transform.inverted()
        if not invertible:
            return 1.0
        return inverted.mapRect(QRectF(0, 0, 1, 1)).width()

    def _lasso_hover_radius_scene(self) -> float:
        unit = self._scene_unit()
        base_radius = (self._lasso_start_screen_radius + self._lasso_start_hover_margin) * unit
        return max(base_radius, 3.0 * unit)

    def _is_near_lasso_start(self, point: QPointF) -> bool:
        if (
            self._tool_mode not in (ToolMode.LASSO, ToolMode.MAGNETIC_LASSO)
            or not self._lasso_points
        ):
            return False
        start = self._lasso_points[0]
        radius = self._lasso_hover_radius_scene()
        dx = point.x() - start.x()
        dy = point.y() - start.y()
        return dx * dx + dy * dy <= radius * radius

    def _update_lasso_start_hover(self) -> None:
        hovering = False
        if (
            self._tool_mode in (ToolMode.LASSO, ToolMode.MAGNETIC_LASSO)
            and self._lasso_points
            and self._hover_pos is not None
        ):
            start = self._lasso_points[0]
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
    ) -> bool:
        if self._label_array is None or target_value is None or not points:
            return False
        height, width = self._label_array.shape
        if width == 0 or height == 0:
            return False
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
            return False
        self._label_array[selection] = target_value
        return True

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
                self._label_array[y, x] = target_value
                changed = True
        return changed
