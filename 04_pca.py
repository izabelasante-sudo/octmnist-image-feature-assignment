"""Assignment Item 4: reduce 36-bin edge histograms to two PCA dimensions."""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from assignment_utils import (
    CLASS_COLORS,
    CLASS_NAMES,
    OUTPUT_DIR,
    combine_subset,
    edge_histogram_matrix,
    load_subset,
    save_json,
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images, labels, splits, source_indices = combine_subset(load_subset())
    edge_histograms = edge_histogram_matrix(images)

    # Labels are intentionally not provided to PCA.
    pca = PCA(n_components=2)
    points = pca.fit_transform(edge_histograms)

    fig, ax = plt.subplots(figsize=(9, 7))
    for class_id in sorted(CLASS_NAMES):
        mask = labels == class_id
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=30,
            alpha=0.65,
            color=CLASS_COLORS[class_id],
            edgecolors="none",
            label=f"{CLASS_NAMES[class_id]} (n={int(mask.sum())})",
        )

    ax.set_title("PCA of 36-bin Edge Histograms (500 OCTMNIST Images)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(alpha=0.2)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "item4_pca_edge_histograms.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    point_rows = []
    for i in range(len(points)):
        point_rows.append(
            {
                "split": str(splits[i]),
                "source_index": int(source_indices[i]),
                "class_id": int(labels[i]),
                "class_name": CLASS_NAMES[int(labels[i])],
                "pc1": float(points[i, 0]),
                "pc2": float(points[i, 1]),
            }
        )
    points_path = OUTPUT_DIR / "item4_pca_points.csv"
    with points_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=point_rows[0].keys())
        writer.writeheader()
        writer.writerows(point_rows)

    summary = {
        "input_shape": list(edge_histograms.shape),
        "output_shape": list(points.shape),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "note": "PCA was fit without class labels; labels were used only for plot colors.",
    }
    save_json(OUTPUT_DIR / "item4_pca_summary.json", summary)

    print(f"PCA input shape: {edge_histograms.shape}")
    print(f"PCA output shape: {points.shape}")
    print(
        "Explained variance (PC1 + PC2): "
        f"{100 * pca.explained_variance_ratio_.sum():.2f}%"
    )
    print(f"Saved: {plot_path}")
    print(f"Saved: {points_path}")


if __name__ == "__main__":
    main()

