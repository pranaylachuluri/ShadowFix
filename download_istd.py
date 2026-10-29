"""Download and extract the ISTD shadow-removal dataset.

ISTD (Wang et al., CVPR 2018) is distributed by its authors as a single RAR archive.
Pass a direct download link or a Google Drive file id/link, or a local copy of the archive:

    python download_istd.py --output-dir data --url "<link to ISTD_Dataset.rar>"
    python download_istd.py --output-dir data --archive ~/Downloads/ISTD_Dataset.rar

Extraction needs `unrar` (Colab: apt-get install -y unrar). Afterwards the data root contains
ISTD_Dataset/train/train_{A,B,C} and ISTD_Dataset/test/test_{A,B,C}.
"""
import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def download(url, dest):
    if "drive.google.com" in url or ("/" not in url and len(url) > 20):
        try:
            import gdown
        except ImportError:
            sys.exit("Google Drive links need gdown: pip install gdown")
        gdown.download(url if "/" in url else None, str(dest), id=None if "/" in url else url, quiet=False, fuzzy=True)
    else:
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)


def extract(archive, output_dir):
    archive = Path(archive)
    if archive.suffix.lower() == ".zip":
        shutil.unpack_archive(str(archive), str(output_dir))
        return
    if shutil.which("unrar") is None:
        sys.exit("unrar is not installed. On Colab/Ubuntu: apt-get install -y unrar")
    subprocess.run(["unrar", "x", "-o+", str(archive), str(output_dir) + "/"], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="direct link, Google Drive link or Google Drive file id")
    src.add_argument("--archive", help="local ISTD .rar or .zip")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive) if args.archive else out / "ISTD_Dataset.rar"
    if args.url and not archive.exists():
        download(args.url, archive)
    extract(archive, out)
    print(f"Extraction complete. Inspect: {out}")
    print("Expected folders include train/train_A, train/train_B, train/train_C, test/test_A, test/test_B, and test/test_C.")


if __name__ == "__main__":
    main()
