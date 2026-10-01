"""Shared utilities for the OCTMNIST image-mining assignment."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
from skimage import color, exposure, filters, util


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "octmnist.npz"
SUBSET_PATH = PROJECT_DIR / "data" / "selected" / "octmnist_subset_500.npz"
OUTPUT_DIR = PROJECT_DIR / "output"

OCTMNIST_URL = (
    "https://zenodo.org/records/10519652/files/octmnist.npz?download=1"
)
OCTMNIST_MD5 = "c68d92d5b585d8d81f7112f81e2d0842"

CLASS_NAMES = {
    0: "choroidal neovascularization",
    1: "diabetic macular edema",
    2: "drusen",
    3: "normal",
}

CLASS_SLUGS = {
    0: "cnv",
    1: "dme",
    2: "drusen",
    3: "normal",
}

CLASS_COLORS = {
    0: "#d62728",
    1: "#ff7f0e",
    2: "#9467bd",
    3: "#1f77b4",
}

SPLIT_CLASS_COUNTS = {
    "train": 80,
    "val": 20,
    "test": 25,
}

RANDOM_SEED = 42
N_BINS = 36


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_octmnist() -> Path:
    """Download and verify the official 28x28 OCTMNIST archive if needed."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists() and md5sum(RAW_DATA_PATH) == OCTMNIST_MD5:
        return RAW_DATA_PATH

    print(f"Downloading OCTMNIST from {OCTMNIST_URL}")
    temporary_path = RAW_DATA_PATH.with_suffix(".npz.part")
    urllib.request.urlretrieve(OCTMNIST_URL, temporary_path)

    actual_md5 = md5sum(temporary_path)
    if actual_md5 != OCTMNIST_MD5:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"OCTMNIST MD5 mismatch: expected {OCTMNIST_MD5}, got {actual_md5}"
        )

    temporary_path.replace(RAW_DATA_PATH)
    return RAW_DATA_PATH


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a floating-point grayscale image in the range [0, 1]."""
    if image.ndim == 3:
        return color.rgb2gray(image)
    return util.img_as_float(image)


def angle(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Calculate angles between the horizontal and vertical Sobel operators."""
    return np.mod(np.arctan2(dy, dx), np.pi)


def edge_angle_image(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    return angle(filters.sobel_h(gray), filters.sobel_v(gray))


def edge_histogram(image: np.ndarray, nbins: int = N_BINS):
    """Compute the assignment's Sobel-angle histogram."""
    angle_sobel = edge_angle_image(image)
    counts, bin_centers = exposure.histogram(angle_sobel, nbins=nbins)
    return counts.astype(np.float64), bin_centers


def edge_histogram_matrix(images: np.ndarray, nbins: int = N_BINS) -> np.ndarray:
    return np.vstack([edge_histogram(image, nbins)[0] for image in images])


def load_subset():
    if not SUBSET_PATH.exists():
        raise FileNotFoundError(
            f"Subset not found at {SUBSET_PATH}. Run 01_prepare_subset.py first."
        )

    with np.load(SUBSET_PATH) as data:
        result = {key: data[key].copy() for key in data.files}
    return result


def combine_subset(subset):
    """Combine the preserved train/val/test portions for assignment Items 2-4."""
    images = np.concatenate(
        [subset[f"{split}_images"] for split in ("train", "val", "test")]
    )
    labels = np.concatenate(
        [subset[f"{split}_labels"].reshape(-1) for split in ("train", "val", "test")]
    )
    splits = np.concatenate(
        [
            np.repeat(split, len(subset[f"{split}_labels"]))
            for split in ("train", "val", "test")
        ]
    )
    original_indices = np.concatenate(
        [subset[f"{split}_indices"] for split in ("train", "val", "test")]
    )
    return images, labels.astype(int), splits, original_indices


def representative_indices(features: np.ndarray, labels: np.ndarray):
    """Select one reproducible, central example from each class."""
    representatives = {}
    for class_id in sorted(CLASS_NAMES):
        indices = np.flatnonzero(labels == class_id)
        class_features = features[indices]
        class_mean = class_features.mean(axis=0)
        distance_to_mean = np.linalg.norm(class_features - class_mean, axis=1)
        representatives[class_id] = int(indices[np.argmin(distance_to_mean)])
    return representatives


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")

