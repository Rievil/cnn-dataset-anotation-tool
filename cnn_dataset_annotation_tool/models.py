from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PySide6.QtGui import QColor


@dataclass
class DatasetEntry:
    """Bundle holding image data, original label, and editable label."""

    name: str
    image_path: Path
    label_path: Path
    image: np.ndarray  # RGB uint8
    original_label: np.ndarray  # int32
    edited_label: np.ndarray  # int32

    def reset_edits(self) -> None:
        """Restore edited label to original."""
        self.edited_label = self.original_label.copy()


@dataclass
class ClassDefinition:
    """Semantic class description."""

    name: str
    value: int
    color: QColor

    def color_tuple(self) -> Tuple[int, int, int]:
        return self.color.red(), self.color.green(), self.color.blue()
