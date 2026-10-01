"""Run every assignment step in order."""

from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTS = (
    "01_prepare_subset.py",
    "02_edge_histograms_and_distances.py",
    "03_hog.py",
    "04_pca.py",
)


def main():
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===", flush=True)
        subprocess.run([sys.executable, str(PROJECT_DIR / script)], check=True)
    print("\nAll assignment outputs were generated successfully.")


if __name__ == "__main__":
    main()

