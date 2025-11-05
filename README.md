# CNN Dataset Annotation Tool

## Overview
The CNN Dataset Annotation Tool is a PySide6 desktop application for reviewing and repairing pixel-wise segmentation results. It loads paired images and label masks, overlays color-coded classes, and gives you interactive tools to correct mistakes before exporting a cleaned dataset.

## Project Status
Since the last README update the project has moved from specification to a working annotator. Highlights:

- **Implemented** – Load sessions from paired folders or parquet files (with class definitions), edit masks with brush/lasso/polygon/polyline tools, undo or redo every change with a per-item history view, mark items for export and batch-write images plus labels, and manage the same parquet datasets through the shared CLI.
- **In progress** – The dataset description tab is UI-only; saving those key/value notes back into parquet is still pending. Editing per-entry metadata from the GUI and adding richer QA helpers remain on the roadmap.

## Desktop Application

### Getting Started
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Launch the GUI via `python main.py`.

A smoke-test pair lives in `datasets/images` and `datasets/labels`.

### Windows Standalone Build
- On a Windows machine with Python 3.9+ installed, open PowerShell and run `powershell -ExecutionPolicy Bypass -File build_windows_exe.ps1` from the repository root.
- The script creates a local `.venv-build`, installs PyInstaller plus the app requirements, and emits `dist/cnn-dataset-annotation-tool.exe`.
- The packaged build includes the `cnn_dataset_annotation_tool` package and, when present, the demo `datasets` directory for quick validation.
- Re-run the script any time the code changes; existing dependencies are reused to keep builds fast.
- Delete the `.venv-build` folder if you need to force a clean environment on the next build.

### Key Capabilities
- Load datasets from folders (auto-matching common filenames) or resume saved parquet sessions, with clear feedback when pairs are skipped.
- Append standalone images with `Add Image` and attach masks later through `Load Mask`.
- Maintain class definitions: rename classes, reassign pixel values, pick colors, or auto-detect values from the current labels.
- Toggle between edited and original labels, adjust overlay alpha, and hide or show the controls column to maximise canvas space.
- Choose among multiple editing modes: brush, freehand lasso, magnetic lasso, polygon fill, and variable-width polygon line strokes. Left click applies source→target, right click swaps direction.
- Track every modification with undo/redo shortcuts, inspect the history tab, and revert an item back to its original mask.
- Mark items for export from the list context menu, then export selected images and labels into organised `Images/` and `Labels/` directories.
- Save all work—including images, masks, class definitions, and per-entry metadata—to parquet for later resumption or CLI automation.

### Typical Workflow
1. Click `Load Dataset` and choose between a parquet session or paired folders.
2. Use `Add Image` / `Load Mask` to fill in missing pairs if needed.
3. Auto-detect classes or configure them manually, then review overlay alpha and source/target selections.
4. Pick an editing tool; left click applies the configured direction, right click reverses it.
5. Zoom with `Ctrl` + mouse wheel, pan with the middle mouse button, and use undo/redo as you refine the mask.
6. Toggle between edited and original labels or reset the current item when you need a clean slate.
7. Mark finished items for export via the list context menu.
8. Save the session to parquet or run `Export Images & Labels` to write the marked items to disk.

### Editing Tool Tips
- Freehand and polygon lassos close when you return to the start point; right click cancels the trace.
- Magnetic lasso follows image gradients—well-defined edges produce the best results.
- Polygon line thickness adjusts with the mouse wheel while tracing; the current value appears beside the tool controls.

## Command Line Dataset Management
The CLI shares the same `DatasetStore` as the GUI. Install dependencies, then manage parquet datasets from the terminal:

```
python -m cnn_dataset_annotation_tool.cli --dataset work.parquet list
```

Available subcommands:
- `list` – display every entry with its image, label, and metadata.
- `add NAME IMAGE LABEL [-m KEY=VALUE ...] [--overwrite]` – append a new item (or replace an existing one).
- `remove NAME` – delete an entry.
- `update NAME [--image IMAGE] [--label LABEL] [-m KEY=VALUE ...] [--replace-metadata|--clear-metadata]` – modify assets or metadata in place.

## Sample Data
`work.parquet` demonstrates the saved-session format, and the `datasets/images` plus `datasets/labels` folders supply a minimal pair for quick validation.

## Roadmap
- Persist the dataset description tab into saved parquet files.
- Surface per-entry metadata editing inside the GUI.
- Add automated QA helpers and additional shortcuts as the workflow evolves.
