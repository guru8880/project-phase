import os
import sys
import argparse
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import shutil

def download_kaggle_dataset(kaggle_ref, out_dir):
    """
    kaggle_ref: e.g. 'paultimothymooney/chest-xray-pneumonia' or 'deepfake-detection-challenge'
    out_dir: directory to extract dataset into
    """
    api = KaggleApi()
    api.authenticate()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {kaggle_ref} ...")
    # this downloads the dataset as dataset.zip into current dir
    api.dataset_download_files(kaggle_ref, path=str(out_dir), unzip=False, quiet=False)

    # find the zip(s) and unzip
    for f in out_dir.glob("*.zip"):
        print(f"Extracting {f} ...")
        with zipfile.ZipFile(f, 'r') as z:
            z.extractall(path=str(out_dir))
        f.unlink()  # remove zip after extraction
    print("Download & extraction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Kaggle dataset reference (owner/dataset-name)")
    parser.add_argument("--out", default="../data", help="Output directory (relative to script)")
    args = parser.parse_args()
    download_kaggle_dataset(args.ref, args.out)