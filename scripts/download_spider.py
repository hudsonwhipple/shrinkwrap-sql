"""Download Spider (questions, gold SQL and SQLite databases) and verify the archive.

Source: the official Google Drive file linked from https://yale-lily.github.io/spider.
Spider is CC BY-SA 4.0 (Yu et al., EMNLP 2018).
"""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

DRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
URL = f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t"
SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "spider_data.zip"
    if not archive.exists():
        print(f"Downloading Spider from Google Drive -> {archive}")
        download(URL, archive)

    digest = sha256(archive)
    if digest != SHA256:
        sys.exit(f"Checksum mismatch for {archive}: got {digest}, expected {SHA256}. Delete it and retry.")

    if not (args.out / "spider_data" / "dev.json").exists():
        print(f"Extracting -> {args.out / 'spider_data'}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(args.out, members=[m for m in zf.namelist() if not m.startswith("__MACOSX")])
    print("Spider ready.")


if __name__ == "__main__":
    main()
