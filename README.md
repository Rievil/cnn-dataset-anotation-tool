# CNN Dataset Annotation Tool

## Overview
The CNN Dataset Annotation Tool is a PySide6 desktop application for reviewing and repairing pixel-wise segmentation results. It loads paired images and label masks, overlays color-coded classes, and gives you interactive tools to correct mistakes before exporting a cleaned dataset.

## Project Status
Since the last README update the project has moved from specification to a working annotator. Highlights:

- **Implemented** – Load sessions from paired folders or parquet files (with class definitions), edit masks with brush/lasso/polygon/polyline tools, undo or redo every change with a per-item history view, mark items for export and batch-write images plus labels, and manage the same parquet datasets through the shared CLI.
- **In progress** – The dataset description tab is UI-only; saving those key/value notes back into parquet is still pending. Editing per-entry metadata from the GUI and adding richer QA helpers remain on the roadmap.

## Documentation
- [Application Design & Workflow](docs/app_design.md) – high-level UI layout, tool behavior (including the “finalizing dot” interactions), recoloring guidance, export modes, and CLI automation tips with embedded script examples.

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
- Visualize any class as a **distance map + skeleton** (Tools tab → "Distance Map / Skeleton View"): pick a class (defaults to Crack), and the overlay colors each class pixel by its Euclidean distance to the class edge (dark blue = edge, yellow = center) with the medial skeleton drawn in red — the same construction as the CNN crack-width metric (width = 2 × distance at the skeleton). Use it to pick representative crack locations before measuring widths. Requires `scipy` + `scikit-image`; the result is cached and recomputed only when the mask changes.
- Measure crack widths manually with the **Measure Width** tool: left click the two opposite crack edges to record a width (shown in px on the canvas), right click to cancel the first point or delete the nearest measurement. A "Show measured values" checkbox in the controls hides the numeric labels when many measurements would clutter the view (lines and end ticks stay visible). An **Annotator** field tags every new measurement with the person's name — measurements are drawn in a distinct color per annotator, the name is stored in the metadata and exported in the CSV (`annotator` column), enabling per-person analysis (e.g., inter-annotator agreement).
- Merge measurements from another session or person with **File → Import Width Measurements (CSV)...**: rows are matched to entries by `entry` name (or `image_file`), exact duplicates are skipped, and rows without an annotator get the current Annotator field value (or `imported`). The import summary reports imported/duplicate/unmatched counts. Measurements are stored in the per-entry metadata (key `width_measurements`), survive session save/load, and export to CSV via `File → Export Width Measurements (CSV)...` — including the label class value under each measurement midpoint for QA. Used to validate the CNN-derived crack-width metric against independent manual readings.

### Typical Workflow
1. Click `Load Dataset` and choose between a parquet session or paired folders.
2. Use `Add Image` / `Load Mask` to fill in missing pairs if needed.
3. Auto-detect classes or configure them manually, then review overlay alpha and source/target selections.
4. Pick an editing tool; left click applies the configured direction, right click reverses it.
5. Zoom with `Ctrl` + mouse wheel (clamped between half the fit-to-view scale and 64× so the image can never be zoomed out of reach), pan with the middle mouse button, and use undo/redo as you refine the mask.
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

## Assets
- Placeholder application icons live in `resources/app_icon.png` and `resources/app_icon.ico`. Replace them with branded artwork before distributing installers.

## Roadmap
- Persist the dataset description tab into saved parquet files.
- Surface per-entry metadata editing inside the GUI.
- Add automated QA helpers and additional shortcuts as the workflow evolves.
