from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .constants import DATA_EXTENSIONS, LABEL_EXTENSIONS, qcolor_from_hex
from .models import ClassDefinition, DatasetEntry


def collect_files(folder: Path, extensions: Sequence[str]) -> Dict[str, Path]:
    """Return mapping from base filename to path for allowed extensions."""
    result: Dict[str, Path] = {}
    for item in folder.iterdir():
        if item.is_file() and item.suffix.lower() in extensions:
            result[item.stem] = item
    return result


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


def load_dataset_from_folders(image_dir: Path, label_dir: Path) -> Tuple[List[DatasetEntry], List[str]]:
    """Load dataset entries from folder pairs, returning entries and any error messages."""
    image_files = collect_files(image_dir, DATA_EXTENSIONS)
    label_files = collect_files(label_dir, LABEL_EXTENSIONS)
    common_names = sorted(set(image_files).intersection(label_files))
    if not common_names:
        return [], ["Could not find matching image/label filenames across the selected folders."]

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
                metadata={},
            )
        )

    return entries, errors


def _flatten_array(array: np.ndarray) -> Tuple[bytes, Tuple[int, ...], str]:
    """Convert numpy array to portable components."""
    return array.tobytes(), array.shape, str(array.dtype)


def _restore_array(data: bytes, shape: Iterable[int], dtype: str) -> np.ndarray:
    """Reconstruct numpy array from portable components."""
    return np.frombuffer(data, dtype=dtype).reshape(tuple(shape))


def _flatten_optional(array: Optional[np.ndarray]) -> Tuple[Optional[bytes], Optional[List[int]], Optional[str]]:
    if array is None:
        return None, None, None
    data, shape, dtype = _flatten_array(array)
    return data, list(shape), dtype


def _coerce_array_bytes(data: object) -> Optional[bytes]:
    if data is None:
        return None
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, np.ndarray):
        return data.tobytes()
    return None


def _coerce_shape(shape: object) -> Optional[Tuple[int, ...]]:
    if shape is None:
        return None
    if isinstance(shape, (list, tuple)):
        try:
            return tuple(int(dim) for dim in shape)
        except (TypeError, ValueError):
            return None
    if isinstance(shape, np.ndarray):
        try:
            return tuple(int(dim) for dim in shape.tolist())
        except (TypeError, ValueError):
            return None
    return None


def _coerce_dtype(dtype: object) -> Optional[str]:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        dtype = dtype.strip()
        return dtype or None
    if isinstance(dtype, np.dtype):
        return str(dtype)
    return None


def _restore_optional_array(data: object, shape: object, dtype: object) -> Optional[np.ndarray]:
    bytes_data = _coerce_array_bytes(data)
    norm_shape = _coerce_shape(shape)
    norm_dtype = _coerce_dtype(dtype)
    if bytes_data is None or norm_shape is None or norm_dtype is None:
        return None
    return _restore_array(bytes_data, norm_shape, norm_dtype).copy()


def save_entries_to_parquet(
    entries: Sequence[DatasetEntry],
    classes: Sequence[ClassDefinition],
    path: Path,
) -> None:
    """Serialize dataset entries to a parquet file."""
    records: List[Dict[str, object]] = []
    classes_payload = json.dumps(
        [
            {"name": cls.name, "value": cls.value, "color": cls.color.name()}
            for cls in classes
        ]
    )
    for entry in entries:
        img_bytes, img_shape, img_dtype = _flatten_array(entry.image)
        orig_bytes, orig_shape, orig_dtype = _flatten_optional(entry.original_label)
        edit_bytes, edit_shape, edit_dtype = _flatten_optional(entry.edited_label)
        records.append(
            {
                "name": entry.name,
                "image_filename": entry.image_path.name,
                "label_filename": entry.label_path.name if entry.label_path else None,
                "image_bytes": img_bytes,
                "image_shape": list(img_shape),
                "image_dtype": img_dtype,
                "original_bytes": orig_bytes,
                "original_shape": orig_shape,
                "original_dtype": orig_dtype,
                "edited_bytes": edit_bytes,
                "edited_shape": edit_shape,
                "edited_dtype": edit_dtype,
                "classes_json": classes_payload,
                "metadata_json": json.dumps({str(k): str(v) for k, v in (entry.metadata or {}).items()}),
            }
        )
    df = pd.DataFrame.from_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_entries_from_parquet(path: Path) -> Tuple[List[DatasetEntry], Optional[List[ClassDefinition]]]:
    """Rehydrate dataset entries from a parquet file."""
    df = pd.read_parquet(path)
    entries: List[DatasetEntry] = []
    classes: Optional[List[ClassDefinition]] = None
    class_json = None
    if "classes_json" in df.columns and not df.empty:
        class_json = df.iloc[0]["classes_json"]
    if isinstance(class_json, (str, bytes)):
        try:
            raw_classes = json.loads(class_json)
            classes = []
            for cls in raw_classes:
                name = str(cls.get("name", "")).strip()
                value = int(cls.get("value", 0))
                color_code = str(cls.get("color", "#000000"))
                color = qcolor_from_hex(color_code)
                classes.append(ClassDefinition(name or f"Class {value}", value, color))
        except (ValueError, TypeError, json.JSONDecodeError):
            classes = None
    for _, row in df.iterrows():
        image = _restore_array(row["image_bytes"], row["image_shape"], row["image_dtype"]).copy()
        original = _restore_optional_array(
            row.get("original_bytes") if hasattr(row, "get") else None,
            row.get("original_shape") if hasattr(row, "get") else None,
            row.get("original_dtype") if hasattr(row, "get") else None,
        )
        edited = _restore_optional_array(
            row.get("edited_bytes") if hasattr(row, "get") else None,
            row.get("edited_shape") if hasattr(row, "get") else None,
            row.get("edited_dtype") if hasattr(row, "get") else None,
        )
        metadata: Dict[str, str] = {}
        metadata_raw = row.get("metadata_json") if hasattr(row, "get") else None
        if isinstance(metadata_raw, (str, bytes)) and metadata_raw:
            try:
                decoded = json.loads(metadata_raw)
                if isinstance(decoded, dict):
                    metadata = {str(k): str(v) for k, v in decoded.items()}
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
        label_filename = row.get("label_filename") if hasattr(row, "get") else None
        if isinstance(label_filename, float) and np.isnan(label_filename):
            label_filename = None
        if label_filename in ("", None):
            label_path = None
        else:
            label_path = Path(str(label_filename))
        entries.append(
            DatasetEntry(
                name=str(row["name"]),
                image_path=Path(str(row["image_filename"])),
                label_path=label_path,
                image=image,
                original_label=original,
                edited_label=edited,
                metadata=metadata,
            )
        )
    return entries, classes
