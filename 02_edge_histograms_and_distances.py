"""Assignment Item 2: edge histograms and histogram distances."""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances

from assignment_utils import (
    CLASS_NAMES,
    CLASS_SLUGS,
    N_BINS,
    OUTPUT_DIR,
    combine_subset,
    edge_histogram_matrix,
    load_subset,
    representative_indices,
    save_json,
    to_grayscale,
)


COMPARISON_CLASSES = (0, 3)  # choroidal neovascularization vs. normal


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images, labels, splits, source_indices = combine_subset(load_subset())
    histograms = edge_histogram_matrix(images, N_BINS)
    representatives = representative_indices(histograms, labels)

    fig, axes = plt.subplots(4, 2, figsize=(12, 15))
    representative_metadata = {}

    for row, class_id in enumerate(sorted(CLASS_NAMES)):
        index = representatives[class_id]
        image = to_grayscale(images[index])
        histogram = histograms[index]

        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(CLASS_NAMES[class_id].title())
        axes[row, 0].axis("off")

        axes[row, 1].bar(np.arange(1, N_BINS + 1), histogram, width=0.85)
        axes[row, 1].set_title(f"{N_BINS}-bin Sobel edge-angle histogram")
        axes[row, 1].set_xlabel("Bins")
        axes[row, 1].set_ylabel("Pixel Count")
        axes[row, 1].set_xlim(0.25, N_BINS + 0.75)

        representative_metadata[str(class_id)] = {
            "class_name": CLASS_NAMES[class_id],
            "combined_subset_index": index,
            "split": str(splits[index]),
            "source_index": int(source_indices[index]),
        }

    fig.suptitle("Representative OCTMNIST Images and Edge Histograms", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    figure_path = OUTPUT_DIR / "item2_edge_histograms.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    first_class, second_class = COMPARISON_CLASSES
    first_histogram = histograms[representatives[first_class]].reshape(1, -1)
    second_histogram = histograms[representatives[second_class]].reshape(1, -1)
    metric_names = {
        "euclidean": "Euclidean distance",
        "manhattan": "Manhattan distance",
        "cosine": "Cosine distance",
    }
    distance_rows = []
    for metric, display_name in metric_names.items():
        value = float(pairwise_distances(first_histogram, second_histogram, metric=metric)[0, 0])
        distance_rows.append(
            {
                "class_1": CLASS_NAMES[first_class],
                "class_2": CLASS_NAMES[second_class],
                "metric": display_name,
                "distance": value,
            }
        )

    distance_path = OUTPUT_DIR / "item2_histogram_distances.csv"
    with distance_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=distance_rows[0].keys())
        writer.writeheader()
        writer.writerows(distance_rows)

    save_json(OUTPUT_DIR / "representative_images.json", representative_metadata)
    np.savetxt(
        OUTPUT_DIR / "edge_histograms_500x36.csv",
        histograms,
        delimiter=",",
        fmt="%.0f",
        header=",".join(f"bin_{i:02d}" for i in range(1, N_BINS + 1)),
        comments="",
    )

    print(f"Saved: {figure_path}")
    print(f"Saved: {distance_path}")
    for row in distance_rows:
        print(f"{row['metric']}: {row['distance']:.6f}")


if __name__ == "__main__":
    main()

