"""download a small selection of webhose financial-news ZIP releases.

usage: python scripts/download_webhose_dataset.py --latest 4
"""
from __future__ import annotations

import argparse
from pathlib import Path
import requests

API = "https://api.github.com/repos/Webhose/financial-news-dataset/contents/Datasets"
RAW = "https://raw.githubusercontent.com/Webhose/financial-news-dataset/master/Datasets"
DEST = Path("data/webhose")


def main() -> None:
    """find the newest webhose zip files on github and download any we don't have yet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest", type=int, default=4, help="number of newest archives to download (default: 4)")
    parser.add_argument("--sentiment", choices=("positive", "negative", "both"), default="both")
    args = parser.parse_args()
    if args.latest < 1:
        parser.error("--latest must be at least 1")

    # list the files in the webhose github repository and keep the zip archives.
    response = requests.get(API, timeout=30)
    response.raise_for_status()
    files = [x["name"] for x in response.json() if x.get("name", "").lower().endswith(".zip")]
    if args.sentiment != "both":
        files = [name for name in files if f"_{args.sentiment}_" in name.lower()]
    # newest first (the file names start with the date), then keep the newest few.
    files.sort(reverse=True)
    selected = files[: args.latest]
    if not selected:
        raise SystemExit("No matching ZIP archives found in the Webhose dataset repository")

    DEST.mkdir(parents=True, exist_ok=True)
    # download each archive unless we already have it.
    for name in selected:
        target = DEST / name
        if target.exists() and target.stat().st_size:
            print(f"cached {target}")
            continue
        download = requests.get(f"{RAW}/{name}", timeout=120)
        download.raise_for_status()
        target.write_bytes(download.content)
        print(f"saved {target} ({len(download.content):,} bytes)")


if __name__ == "__main__":
    main()
