"""Create a balanced, reproducible 500-image OCTMNIST subset."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from assignment_utils import (
    CLASS_NAMES,
    CLASS_SLUGS,
    RANDOM_SEED,
    SPLIT_CLASS_COUNTS,
    SUBSET_PATH,
    download_octmnist,
    save_json,
)


def select_indices(labels, count_per_class, rng):
    selected = []
    labels = labels.reshape(-1)
    for class_id in sorted(CLASS_NAMES):
        candidates = np.flatnonzero(labels == class_id)
        if len(candidates) < count_per_class:
            raise ValueError(
                f"Class {class_id} has {len(candidates)} images; "
                f"cannot select {count_per_class}."
            )
        selected.extend(rng.choice(candidates, count_per_class, replace=False))
    return np.asarray(selected, dtype=np.int64)


def export_pngs(subset, export_root: Path):
    rows = []
    for split in ("train", "val", "test"):
        images = subset[f"{split}_images"]
        labels = subset[f"{split}_labels"].reshape(-1)
        indices = subset[f"{split}_indices"]

        class_counters = {class_id: 0 for class_id in CLASS_NAMES}
        for image, label, original_index in zip(images, labels, indices):
            class_id = int(label)
            sequence = class_counters[class_id]
            class_counters[class_id] += 1

            folder = export_root / split / f"{class_id}_{CLASS_SLUGS[class_id]}"
            folder.mkdir(parents=True, exist_ok=True)
            filename = f"{sequence:03d}_source_{int(original_index):05d}.png"
            path = folder / filename
            Image.fromarray(image).save(path)

            rows.append(
                {
                    "split": split,
                    "source_index": int(original_index),
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "file": str(path.relative_to(export_root.parent.parent)),
                }
            )
    return rows


def main():
    raw_path = download_octmnist()
    rng = np.random.default_rng(RANDOM_SEED)
    subset = {}

    with np.load(raw_path) as source:
        for split, count_per_class in SPLIT_CLASS_COUNTS.items():
            images = source[f"{split}_images"]
            labels = source[f"{split}_labels"]
            indices = select_indices(labels, count_per_class, rng)
            subset[f"{split}_images"] = images[indices]
            subset[f"{split}_labels"] = labels[indices]
            subset[f"{split}_indices"] = indices

    SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SUBSET_PATH, **subset)

    export_root = SUBSET_PATH.parent / "images"
    rows = export_pngs(subset, export_root)
    manifest_path = SUBSET_PATH.parent / "selection_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "dataset": "OCTMNIST",
        "resolution": [28, 28],
        "random_seed": RANDOM_SEED,
        "classes": {str(key): value for key, value in CLASS_NAMES.items()},
        "images_per_class": sum(SPLIT_CLASS_COUNTS.values()),
        "split_images_per_class": SPLIT_CLASS_COUNTS,
        "total_images": len(rows),
    }
    save_json(SUBSET_PATH.parent / "metadata.json", metadata)

    print(f"Saved balanced subset: {SUBSET_PATH}")
    print(f"Saved PNG images under: {export_root}")
    print(f"Saved selection manifest: {manifest_path}")
    print("500 images total = 125 per class (80 train + 20 val + 25 test).")


if __name__ == "__main__":
    main()

