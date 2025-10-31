from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from .io_utils import (
    load_entries_from_parquet,
    load_label_image,
    load_rgb_image,
    save_entries_to_parquet,
)
from .models import ClassDefinition, DatasetEntry


@dataclass
class DatasetStore:
    """Manage dataset entries stored inside a parquet file."""

    path: Path
    entries: Dict[str, DatasetEntry]
    classes: List[ClassDefinition]

    @classmethod
    def load(cls, path: Path) -> "DatasetStore":
        """Load dataset entries from an existing parquet file or initialise an empty store."""
        path = Path(path)
        if path.exists():
            entries, classes = load_entries_from_parquet(path)
        else:
            entries, classes = [], []
        entry_map = {entry.name: entry for entry in entries}
        return cls(path=path, entries=entry_map, classes=list(classes or []))

    def list_entries(self) -> List[DatasetEntry]:
        """Return dataset entries sorted by name for deterministic output."""
        return [self.entries[name] for name in sorted(self.entries)]

    def save(self) -> None:
        """Persist current state to the configured parquet path."""
        save_entries_to_parquet(list(self.entries.values()), self.classes, self.path)

    def add_entry(
        self,
        name: str,
        image_path: Path,
        label_path: Path,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> DatasetEntry:
        """Add a new dataset entry, replacing any existing entry with the same name."""
        image_path = Path(image_path)
        label_path = Path(label_path)
        image = load_rgb_image(image_path)
        label = load_label_image(label_path)
        if image.shape[:2] != label.shape:
            raise ValueError("Image and label dimensions do not match")
        entry = DatasetEntry(
            name=name,
            image_path=image_path,
            label_path=label_path,
            image=image,
            original_label=label,
            edited_label=label.copy(),
            metadata={str(k): str(v) for k, v in (metadata or {}).items()},
        )
        self.entries[name] = entry
        return entry

    def remove_entry(self, name: str) -> None:
        """Remove an entry by name, raising if it does not exist."""
        try:
            del self.entries[name]
        except KeyError as exc:  # noqa: BLE001
            raise KeyError(f"Entry '{name}' not found") from exc

    def update_entry(
        self,
        name: str,
        *,
        image_path: Optional[Path] = None,
        label_path: Optional[Path] = None,
        metadata: Optional[Mapping[str, str]] = None,
        replace_metadata: bool = False,
    ) -> DatasetEntry:
        """Modify an existing entry."""

        if name not in self.entries:
            raise KeyError(f"Entry '{name}' not found")
        entry = self.entries[name]

        if image_path is not None:
            image_path = Path(image_path)
            entry.image = load_rgb_image(image_path)
            entry.image_path = image_path

        if label_path is not None:
            label_path = Path(label_path)
            new_label = load_label_image(label_path)
            if entry.image.shape[:2] != new_label.shape:
                raise ValueError("Updated label dimensions do not match the image")
            entry.original_label = new_label
            entry.edited_label = new_label.copy()
            entry.label_path = label_path

        if metadata is not None:
            if replace_metadata:
                entry.metadata = {str(k): str(v) for k, v in metadata.items()}
            else:
                updates = {str(k): str(v) for k, v in metadata.items()}
                entry.metadata.update(updates)

        return entry

    def merge_entries(self, entries: Iterable[DatasetEntry]) -> None:
        """Bulk merge entries, keyed by name."""
        for entry in entries:
            self.entries[entry.name] = entry
