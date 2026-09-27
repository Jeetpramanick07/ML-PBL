"""Download and extract the TORGO dysarthric speech database.

TORGO is distributed by the University of Toronto Computational Linguistics
group as four archives, split by gender and dysarthria status:

    F.tar.bz2   female speakers with dysarthria   (~1.1 GB)
    FC.tar.bz2  female control speakers           (~2.5 GB)
    M.tar.bz2   male speakers with dysarthria     (~2.5 GB)
    MC.tar.bz2  male control speakers             (~3.4 GB)

Source page (manual download, if this script ever stops working because the
host or paths changed):
    http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
Direct archive links (as of the last time this script was verified):
    http://www.cs.toronto.edu/~complingweb/data/TORGO/F.tar.bz2
    http://www.cs.toronto.edu/~complingweb/data/TORGO/FC.tar.bz2
    http://www.cs.toronto.edu/~complingweb/data/TORGO/M.tar.bz2
    http://www.cs.toronto.edu/~complingweb/data/TORGO/MC.tar.bz2

If the script fails (host down, moved files, HTML page changed), download the
four archives manually from the page above and extract each one directly
into data/raw/torgo/ (e.g. `tar xjf F.tar.bz2 -C data/raw/torgo/`), then
re-run scripts/build_dataset.py — everything downstream only depends on
data/raw/torgo/ existing with per-speaker folders in it, not on how it got
there.

Usage:
    python scripts/download_torgo.py                # all 4 archives
    python scripts/download_torgo.py --only F FC     # just a subset
    python scripts/download_torgo.py --keep-archives # don't delete .tar.bz2 after extracting

Archives are downloaded and extracted one at a time, and deleted immediately
after successful extraction (unless --keep-archives is passed), since the
extracted corpus plus all four compressed archives held simultaneously can
exceed available disk space on a typical dev machine.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "http://www.cs.toronto.edu/~complingweb/data/TORGO"
ARCHIVES = ["F", "FC", "M", "MC"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "torgo"


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download_file(url: str, dest: Path) -> None:
    """Stream-download a URL to dest, printing periodic progress."""
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB
        last_report_mb = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded // (25 * 1024 * 1024) > last_report_mb:
                    last_report_mb = downloaded // (25 * 1024 * 1024)
                    pct = f" ({downloaded / total:.0%})" if total else ""
                    print(f"  ... {_format_bytes(downloaded)}{pct}", flush=True)
    print(f"  done: {_format_bytes(dest.stat().st_size)}")


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    print(f"  extracting {archive_path.name} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:bz2") as tf:
        tf.extractall(dest_dir, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=ARCHIVES, default=ARCHIVES,
                         help="Subset of archives to fetch (default: all four).")
    parser.add_argument("--keep-archives", action="store_true",
                         help="Do not delete .tar.bz2 files after extracting them.")
    parser.add_argument("--dest", default=str(RAW_DIR),
                         help=f"Extraction target directory (default: {RAW_DIR}).")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    total, used, free = shutil.disk_usage(dest_dir)
    print(f"Disk free at destination: {_format_bytes(free)}")

    for name in args.only:
        url = f"{BASE_URL}/{name}.tar.bz2"
        archive_path = dest_dir / f"{name}.tar.bz2"
        marker = dest_dir / f".{name}.extracted"

        if marker.exists():
            print(f"[{name}] already extracted, skipping (remove {marker} to force).")
            continue

        print(f"[{name}] downloading {url}")
        try:
            download_file(url, archive_path)
        except Exception as exc:  # noqa: BLE001 - report and let user fall back to manual download
            print(f"[{name}] download FAILED: {exc}", file=sys.stderr)
            print("See the manual-download instructions in this script's docstring / README.md.",
                  file=sys.stderr)
            sys.exit(1)

        extract_archive(archive_path, dest_dir)
        marker.touch()

        if not args.keep_archives:
            archive_path.unlink()
            print(f"[{name}] removed archive to save disk space.")

    print("TORGO download/extraction complete.")


if __name__ == "__main__":
    main()
