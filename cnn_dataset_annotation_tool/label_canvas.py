from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QWidget


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
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        scene_pos = self.mapToScene(event.position().toPoint())
        self._hover_pos = scene_pos if self._within_image(scene_pos) else None
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
        self.viewport().update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._painting = False
            self._painting_button = None
            self._active_source = None
            self._active_target = None
            self._last_paint_point = None
            event.accept()
            return
        if event.button() == Qt.MiddleButton:
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
