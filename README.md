# OCTMNIST image feature assignment

This project completes Assignment Items 2-4 with a reproducible, balanced
500-image subset of **OCTMNIST**.

## Dataset choice

OCTMNIST is a good fit because it is already a four-class, single-label image
dataset. The four classes are:

1. choroidal neovascularization (CNV)
2. diabetic macular edema (DME)
3. drusen
4. normal

The images are already grayscale, so no RGB-to-grayscale conversion is needed.
The official MedMNIST metadata reports 109,309 images and official train,
validation, and test splits. This project uses the standard 28 x 28 MedMNIST
v2 archive.

Official sources:

- MedMNIST website: <https://medmnist.com/v2>
- Official metadata and label definitions:
  <https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py>
- Official dataset archive: <https://zenodo.org/records/10519652>

## Selected 500-image dataset

The scripts use random seed 42 and select 125 images per class while preserving
the official split structure:

| Split | Images per class | Total images |
|---|---:|---:|
| Train | 80 | 320 |
| Validation | 20 | 80 |
| Test | 25 | 100 |
| **Total** | **125** | **500** |

The selected subset is stored in:

- `data/selected/octmnist_subset_500.npz`
- `data/selected/selection_manifest.csv`

Running `01_prepare_subset.py` also exports 500 convenient PNG copies under
`data/selected/images/`. Those generated copies are not tracked because all
500 images and labels are already contained in the much smaller subset NPZ.

## Installation and execution

From this directory, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The selected 500-image dataset is included in this repository. The full
109,309-image raw OCTMNIST archive is intentionally not tracked because it can
be reproduced from the official source. If the raw archive is absent, the
preparation script downloads it from Zenodo and checks its MD5 checksum.

## Scripts

- `01_prepare_subset.py`: selects and exports the balanced 500-image subset.
- `02_edge_histograms_and_distances.py`: selects one representative image per
  class, computes the required 36-bin Sobel-angle histograms, and calculates
  Euclidean, Manhattan, and cosine distances.
- `03_hog.py`: computes and visualizes the HOG descriptor for the representative
  normal image.
- `04_pca.py`: computes all 500 edge histograms, applies PCA from 36 dimensions
  to 2 without using labels, and colors the final points by class.
- `run_all.py`: runs all four scripts in sequence.

## Reproducible selection decisions

The representative image for each class is not chosen subjectively. The script
computes all 36-bin histograms in a class, calculates the class mean histogram,
and selects the image whose histogram is closest to that mean.

For the distance comparison, the script uses the representative CNV and normal
images. Class labels are never supplied to PCA; they are used only after PCA to
color the scatter plot.

## Main outputs

- `output/item2_edge_histograms.png`
- `output/item2_histogram_distances.csv`
- `output/item3_hog_normal.png`
- `output/item4_pca_edge_histograms.png`
- `output/item4_pca_points.csv`
- `output/item4_pca_summary.json`

The ready-to-submit results and interpretation are summarized in `REPORT.md`.
