from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtGui import QColor


@dataclass
class EditOperation:
    """Record of a single editing action on a label array."""

    description: str
    coordinates: np.ndarray  # shape (N, 2) with (row, col) indices
    previous_values: np.ndarray  # shape (N,)
    new_values: np.ndarray  # shape (N,)

    def pixel_count(self) -> int:
        return int(self.coordinates.shape[0]) if self.coordinates.ndim == 2 else 0


@dataclass
class DatasetEntry:
    """Bundle holding image data, original label, and editable label."""

    name: str
    image_path: Path
    label_path: Optional[Path]
    image: np.ndarray  # RGB uint8
    original_label: Optional[np.ndarray]  # int32
    edited_label: Optional[np.ndarray]  # int32
    metadata: Dict[str, str] = field(default_factory=dict)
    undo_stack: List[EditOperation] = field(default_factory=list, repr=False)
    redo_stack: List[EditOperation] = field(default_factory=list, repr=False)

    def reset_edits(self) -> None:
        """Restore edited label to original."""
        if self.original_label is not None:
            self.edited_label = self.original_label.copy()
        self.undo_stack.clear()
        self.redo_stack.clear()

    @property
    def has_label(self) -> bool:
        return self.edited_label is not None


@dataclass
class ClassDefinition:
    """Semantic class description."""

    name: str
    value: int
    color: QColor

    def color_tuple(self) -> Tuple[int, int, int]:
        return self.color.red(), self.color.green(), self.color.blue()
