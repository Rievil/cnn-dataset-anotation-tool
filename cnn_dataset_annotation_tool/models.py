from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PySide6.QtGui import QColor


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

    def reset_edits(self) -> None:
        """Restore edited label to original."""
        if self.original_label is not None:
            self.edited_label = self.original_label.copy()

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
