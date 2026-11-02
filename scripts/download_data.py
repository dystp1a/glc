"""Download and extract the official CrossCodeEval data (no git needed; works on Windows/Kaggle).

    python scripts/download_data.py            # downloads ~42 MB, extracts ~850 MB (all languages)
    python scripts/download_data.py --tar path/to/crosscodeeval_data.tar.xz   # use a local copy
"""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

URL = "https://github.com/amazon-science/cceval/raw/main/data/crosscodeeval_data.tar.xz"
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--tar", default=None)
    ap.add_argument("--all_languages", action="store_true", help="extract every language (default: python only)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tar_path = Path(a.tar) if a.tar else out / "crosscodeeval_data.tar.xz"
    if not tar_path.exists():
        print(f"downloading {URL} -> {tar_path}")
        urllib.request.urlretrieve(URL, tar_path)
    print("extracting ...")
    with tarfile.open(tar_path, "r:xz") as t:
        members = [m for m in t.getmembers() if a.all_languages or m.name.split("/")[0] == "python"
                   or "/python/" in m.name or m.name.startswith("python")]
        dest = out / "crosscodeeval_data"
        dest.mkdir(exist_ok=True)
        try:
            t.extractall(dest, members=members, filter="data")
        except TypeError:  # older Python without extraction filters
            t.extractall(dest, members=members)
    py = dest / "python"
    files = sorted(p.name for p in py.glob("*.jsonl"))
    print(f"python files in {py}:")
    for f in files:
        print("  ", f)
    if not files:
        raise SystemExit("no python jsonl files found after extraction - check the tarball layout with `ls`")


if __name__ == "__main__":
    main()
