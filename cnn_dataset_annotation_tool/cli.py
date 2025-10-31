from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .dataset_store import DatasetStore


def _parse_metadata(items: Optional[Iterable[str]]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    if not items:
        return metadata
    for raw in items:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(
                f"Invalid metadata '{raw}'. Use KEY=VALUE format."
            )
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError("Metadata keys cannot be empty")
        metadata[key] = value.strip()
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage CNN annotation parquet datasets from the command line.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        type=Path,
        help="Path to the parquet dataset file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List dataset entries and their metadata.")

    add_parser = subparsers.add_parser("add", help="Add a new image/label pair to the dataset.")
    add_parser.add_argument("name", help="Unique name for the dataset entry.")
    add_parser.add_argument("image", type=Path, help="Path to the RGB image file.")
    add_parser.add_argument("label", type=Path, help="Path to the label image file.")
    add_parser.add_argument(
        "--metadata",
        "-m",
        action="append",
        help="Entry metadata specified as KEY=VALUE (repeat for multiple entries).",
    )
    add_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing entry with the same name if present.",
    )

    remove_parser = subparsers.add_parser("remove", help="Remove an entry by name.")
    remove_parser.add_argument("name", help="Name of the entry to remove.")

    update_parser = subparsers.add_parser("update", help="Modify images, labels, or metadata for an entry.")
    update_parser.add_argument("name", help="Name of the entry to update.")
    update_parser.add_argument(
        "--image",
        type=Path,
        help="Path to a replacement RGB image file.",
    )
    update_parser.add_argument(
        "--label",
        type=Path,
        help="Path to a replacement label image file.",
    )
    update_parser.add_argument(
        "--metadata",
        "-m",
        action="append",
        help="Metadata updates as KEY=VALUE (repeat to specify multiple pairs).",
    )
    update_parser.add_argument(
        "--replace-metadata",
        action="store_true",
        help="Replace existing metadata instead of merging with it.",
    )
    update_parser.add_argument(
        "--clear-metadata",
        action="store_true",
        help="Remove all metadata from the entry.",
    )

    return parser


def _handle_list(store: DatasetStore) -> None:
    entries = store.list_entries()
    if not entries:
        print("No entries found.")
        return
    for entry in entries:
        print(f"{entry.name}")
        print(f"  image: {entry.image_path}")
        print(f"  label: {entry.label_path}")
        if entry.metadata:
            metadata_json = json.dumps(entry.metadata, ensure_ascii=True)
        else:
            metadata_json = "{}"
        print(f"  metadata: {metadata_json}")


def _handle_add(store: DatasetStore, args: argparse.Namespace) -> None:
    if not args.overwrite and args.name in store.entries:
        raise SystemExit(
            f"Entry '{args.name}' already exists. Use --overwrite to replace it."
        )
    metadata = _parse_metadata(args.metadata)
    store.add_entry(args.name, args.image, args.label, metadata=metadata)
    store.save()
    print(f"Entry '{args.name}' added to {store.path}.")


def _handle_remove(store: DatasetStore, args: argparse.Namespace) -> None:
    try:
        store.remove_entry(args.name)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    store.save()
    print(f"Entry '{args.name}' removed from {store.path}.")


def _handle_update(store: DatasetStore, args: argparse.Namespace) -> None:
    metadata_updates: Optional[Dict[str, str]] = None
    replace_metadata = args.replace_metadata
    if args.clear_metadata:
        metadata_updates = {}
        replace_metadata = True
        if args.metadata:
            raise SystemExit("--clear-metadata cannot be combined with --metadata.")
    elif args.metadata:
        metadata_updates = _parse_metadata(args.metadata)

    try:
        store.update_entry(
            args.name,
            image_path=args.image,
            label_path=args.label,
            metadata=metadata_updates,
            replace_metadata=replace_metadata,
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    store.save()
    print(f"Entry '{args.name}' updated in {store.path}.")


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    store = DatasetStore.load(args.dataset)

    if args.command == "list":
        _handle_list(store)
    elif args.command == "add":
        _handle_add(store, args)
    elif args.command == "remove":
        _handle_remove(store, args)
    elif args.command == "update":
        _handle_update(store, args)
    else:  # pragma: no cover - defensive guard
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
