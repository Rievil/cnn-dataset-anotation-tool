# CNN Dataset Annotation Tool

## Overview
The CNN Dataset Annotation Tool is a desktop application built with PySide that streamlines the review and correction of pixel-wise labels for convolutional neural network (CNN) datasets. The tool loads paired images and label masks, lets you visualize segmentation results with customizable overlays, and offers intuitive editing utilities to repair mislabeled pixels before exporting an updated label set.

## Key Features
- **Dataset Loader** – Select separate folders for source images and their corresponding label masks. The tool handles dataset initialization and maintains the link between each image-label pair.
- **Class Management** – Define the semantic classes present in the dataset, assign the integer pixel value each class represents (for example `0 = background`, `1 = matrix`, `2 = crack`, `3 = pore`), and configure per-class display colors.
- **Label Visualization** – Display images with an adjustable overlay of their labels using a Jet colormap. Control the alpha value to fine-tune the transparency of the mask and make label discrepancies easy to spot.
- **Interactive Editing** – Pick a source class (e.g., `pore`) and a target class (e.g., `matrix`) to repaint false positives. Editing is applied with a brush whose size and shape are visualized in real time, ensuring precise corrections.
- **Navigation Tools** – Zoom in/out to inspect details, reposition the cursor accurately, and pan across the canvas by holding the middle mouse button.
- **Revision Control** – Every label mask is duplicated before edits begin so that you can revert to the original state at any time. When edits are complete, export the modified masks as a new label set without overwriting the originals.

## Typical Workflow
1. **Launch the application** and choose the directories that contain your images and the associated labels.
2. **Configure class mappings** by entering each class name, its pixel value, and the desired display color.
3. **Tune the visualization** by adjusting the overlay alpha until the areas needing attention stand out.
4. **Select a repaint operation** by picking the source class you want to correct and the target class you want those pixels to become.
5. **Use the brush tool** to apply corrections, leveraging zoom and pan to work on fine details. The on-screen brush preview helps confirm the brush footprint before committing changes.
6. **Review and revert if needed** using the preserved originals to undo accidental edits.
7. **Export the edited labels** to generate a clean set of masks ready for downstream training or evaluation workflows.

## Getting Started
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the tool with `python main.py`.

The included `datasets/images` and `datasets/labels` folders contain a tiny sample pair you can use for a smoke test.

## Command Line Dataset Management
Install the dependencies and run the CLI to work with parquet datasets:

```
python -m cnn_dataset_annotation_tool.cli --dataset work.parquet list
```

Available subcommands:
- `list` – show every entry with its image, label, and metadata.
- `add NAME IMAGE LABEL [-m KEY=VALUE ...] [--overwrite]` – append a new item (optionally replacing an existing one).
- `remove NAME` – delete an entry.
- `update NAME [--image IMAGE] [--label LABEL] [-m KEY=VALUE ...] [--replace-metadata|--clear-metadata]` – modify data or metadata.

Metadata is stored as JSON key-value pairs. Provide `-m key=value` multiple times to set or update fields.

## Programmatic Usage
Use the `DatasetStore` helper for Python workflows:

```python
from pathlib import Path
from cnn_dataset_annotation_tool import DatasetStore

store = DatasetStore.load(Path("work.parquet"))
store.add_entry(
    name="sample",
    image_path=Path("datasets/images/sample.png"),
    label_path=Path("datasets/labels/sample.png"),
    metadata={"split": "train", "notes": "new capture"},
)
store.save()

for entry in store.list_entries():
    print(entry.name, entry.metadata)
```

## Current Implementation Highlights
- Load image and label folders independently; matching filenames are paired automatically.
- Auto-detect classes from loaded label masks, with controls to rename, reassign values, and choose display colors.
- Adjustable overlay alpha for inspecting the segmentation mask on top of the source image.
- Circular brush with configurable size and live preview for repainting pixels from a chosen source class to a target class.
- Zoom with the mouse wheel (hold `Ctrl`), pan with the middle mouse button, and revert label edits at any time.
- Export edited masks to a user-selected directory without modifying the originals.

Contributions and feedback are welcome as the tool evolves from this specification into a full annotation workflow.
