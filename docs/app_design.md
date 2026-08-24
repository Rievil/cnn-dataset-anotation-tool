# Application Design & Workflow

This document sketches the CNN Dataset Annotation Tool from a product and UX perspective so designers, QA, and contributors can reason about the full experience without diving into the source first.

## Layout Overview
- **Image list (left):** Shows every image/label pair with export selection highlights. Context menu shortcuts let you remove assets, mark items for export, or delete entire entries.
- **Canvas (center):** Displays the RGB image with the currently selected label overlay. Zoom with `Ctrl + mouse wheel`, pan with the middle mouse button, and switch between edited/original labels from the Label View tab.
- **Controls (right):** Tabs expose tool configuration, class/color management, dataset description metadata, and per-image history. The controls column can be collapsed via the “Hide Controls” toggle when you need more canvas space.

## Editing Tools & Finalizing Gestures
- **Brush:** Adjustable via the slider/spinbox or the mouse wheel while the brush is active. Hold the left mouse button to apply the configured source→target mapping, hold the right button to swap. Brushing now previews the stroke until you release, mirroring Ilastik’s UX.
- **Freehand & Magnetic Lasso:** Hold the left mouse button to trace the boundary. When the cursor snaps back to the glowing start dot, a left click fills using the target class, and a right click fills using the reversed mapping. Right-clicking elsewhere cancels the trace.
- **Polygon Tool:** Single left clicks drop vertices. When the cursor is near the start dot, a left click closes/fills with the current mapping; a right click closes/fills with swapped classes. The finalizing dot ensures you never accidentally close while sketching mid-edge.
- **Polygon Line Tool:** Tailored for crack-like annotations. Left clicks drop control points; the mouse wheel adjusts thickness on the fly. Click the start dot to confirm (reversed when you right click). Use the undo stack to back out of intermediate mistakes.
- **Measure Width Tool:** Records manual crack-width readings instead of editing the mask. The first left click marks one crack edge (shown as a cross with a live dashed preview), the second click on the opposite edge finalizes the measurement — drawn as a line with perpendicular end ticks and its length in pixels. Right click cancels an open first point, or deletes the measurement nearest to the cursor. Measurements are per-entry, live in the metadata key `width_measurements` (JSON), persist through parquet save/load, and export to CSV via **File → Export Width Measurements (CSV)…** together with the label class value under each midpoint. Zoom in (Ctrl + wheel) before clicking for sub-pixel precision on fine cracks; measure perpendicular to the local crack direction. When many measurements crowd the view, untick **Show measured values** in the controls to hide the numeric labels — the lines and end ticks remain, and the live preview while measuring always shows its running length. The **Annotator** field (visible in Measure mode) tags each new measurement with the person taking it; each annotator's measurements render in their own color (the live preview uses the active annotator's color), the tag round-trips through parquet sessions and the CSV export, and **File → Import Width Measurements (CSV)…** merges another person's exported CSV into the current session — matching rows by entry name or image filename, skipping exact duplicates, and defaulting untagged rows to the current Annotator value (or `imported`). This keeps each labeler's data separable for inter-annotator agreement analysis.
- **Distance Map / Skeleton View (Tools tab):** Companion to the Measure Width tool. Select a class and tick *Show distance map + skeleton*: the class is recolored by its Euclidean distance transform (dark blue at the edges, yellow at the center line) and the medial skeleton is drawn in red — exactly the construction behind the automated width metric (width = 2 × EDT sampled on the skeleton). This shows where the algorithm "reads" the width, so manual measurements can target the same locations, and makes label defects (breaks, blobs, rims) obvious. The computation runs once per mask state (busy cursor + timing in the status bar, can take seconds on full-resolution images) and is cached until the mask or class changes; the normal class overlay returns when the box is unticked or the class has no pixels. Needs `scipy` and `scikit-image` (in `requirements.txt`); if missing, the app explains instead of crashing.

## Class Management & Re-Colouring Labels
- The Classes panel lists every semantic class with an editable name, numeric value, and color swatch. Changing the color updates the overlay instantly so you can tune palettes for clarity.
- Clicking “Auto Populate” scans the current labels for unique values and seeds new rows (preserving any custom colors that already exist).
- Use “Switch Class Values” next to the tool selectors when you need to flip source/target values globally, and keep the “Any” option selected whenever you want the brush to recolor every value inside its radius.

## Exporting Datasets
- Choose **File → Export…** to open the export control panel.
- **Full images & labels:** Writes every selected item into parallel `Images/` and `Labels/` subfolders inside the chosen destination. Existing files are overwritten.
- **Labels only:** Toggle the content option when you only want mask exports. The Labels folder and manifest still update, but image files are skipped.
- **Sub-images for training:** Specify the tile width/height (default 416×416). The exporter walks each selected entry in tile-sized strides, generating filenames like `road_scene_sub_img_001.png` in both `Images/` and `Labels/` so basenames match for downstream pairing. Entries that lack either an image or label—or that are smaller than the requested tile—are reported in the status bar summary.
- **Class remapping:** The export dialog includes a mapping table so you can merge classes (e.g., map “pores” and “matrix” into “background”) before writing labels. Final pixel values are auto-reindexed to a compact range and previewed in the dialog.
- Every export also writes `dataset.csv` in the chosen destination, listing image/label paths, the original filename, and (for tiled exports) the tile id plus top-left x/y offsets. Columns for description metadata appear automatically when you add keys in the Description panel. Per-class pixel counts use the class names (e.g., `road_surface_px_sum`), reflecting the final values after any remapping.

## Command-Line Interface (CLI)
The same dataset format is accessible via the CLI (`python -m cnn_dataset_annotation_tool.cli`). Command usage details live in the repository README and `cnn_dataset_annotation_tool/cli.py` docstrings.

Here is a simple Python helper that shells out to the CLI to list entries, then refresh the metadata for a specific item:

```python
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATASET = REPO_ROOT / "work.parquet"
ENTRY = "demo-entry"

# List the current entries (mirrors `cli --dataset work.parquet list`).
subprocess.run(
    [sys.executable, "-m", "cnn_dataset_annotation_tool.cli", "--dataset", str(DATASET), "list"],
    check=True,
)

# Update metadata for one entry via the CLI.
subprocess.run(
    [
        sys.executable,
        "-m",
        "cnn_dataset_annotation_tool.cli",
        "--dataset",
        str(DATASET),
        "update",
        ENTRY,
        "--metadata",
        "stage=qa",
        "--metadata",
        "reviewer=ana",
    ],
    check=True,
)
```

> Replace `demo-entry` with an actual entry name in your dataset. The pattern above is convenient for CI pipelines that need to keep metadata in sync without opening the GUI.

## Assets & Iconography
- Placeholder application icons live in `resources/app_icon.png` and `resources/app_icon.ico`. Swap these files with branded artwork when you are ready to ship customized builds.
