from __future__ import annotations

from typing import Tuple

from PySide6.QtGui import QColor


DATA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
LABEL_EXTENSIONS = {".png", ".tif", ".tiff"}

# Metadata key holding manual crack-width measurements as a JSON list of
# {"x1", "y1", "x2", "y2"} pixel coordinates (stored per entry, persisted to parquet).
WIDTH_MEASUREMENTS_KEY = "width_measurements"

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


def fallback_color(value: int) -> Tuple[int, int, int]:
    """Return deterministic fallback color when no class color is defined."""
    value = abs(int(value))
    r = (value * 37) % 256
    g = (value * 67 + 89) % 256
    b = (value * 97 + 53) % 256
    return r, g, b


def qcolor_from_hex(code: str) -> QColor:
    """Construct a QColor from a hex code."""
    return QColor(code)
