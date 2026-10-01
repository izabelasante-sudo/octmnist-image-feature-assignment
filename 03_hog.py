"""Assignment Item 3: compute and visualize a HOG descriptor."""

from __future__ import annotations

import matplotlib.pyplot as plt
from skimage import exposure, feature

from assignment_utils import (
    CLASS_NAMES,
    OUTPUT_DIR,
    combine_subset,
    edge_histogram_matrix,
    load_subset,
    representative_indices,
    to_grayscale,
)


HOG_CLASS = 3  # normal


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images, labels, _, _ = combine_subset(load_subset())
    edge_features = edge_histogram_matrix(images)
    image_index = representative_indices(edge_features, labels)[HOG_CLASS]
    image = to_grayscale(images[image_index])

    descriptor, hog_image = feature.hog(
        image,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    hog_display = exposure.rescale_intensity(hog_image, in_range=(0, hog_image.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original: {CLASS_NAMES[HOG_CLASS].title()}")
    axes[0].axis("off")
    axes[1].imshow(hog_display, cmap="gray")
    axes[1].set_title(f"HOG visualization ({descriptor.size} values)")
    axes[1].axis("off")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "item3_hog_normal.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"HOG descriptor dimensionality: {descriptor.size}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

