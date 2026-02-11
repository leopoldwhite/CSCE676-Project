#!/usr/bin/env python3
"""Download candidate datasets for checkpoint 1.

Datasets:
1) GRBench (HuggingFace JSON files)
2) OGBN-Arxiv (zip from SNAP OGB mirror)
3) SNAP com-Amazon graph (gz edge list)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

HF_TREE_URL = "https://huggingface.co/api/datasets/PeterJinGo/GRBench/tree/main?recursive=true"
HF_FILE_BASE = "https://huggingface.co/datasets/PeterJinGo/GRBench/resolve/main"
OGBN_ARXIV_URL = "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"
SNAP_AMAZON_URL = "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz"


def safe_relpath(path: Path, project_root: Path) -> str:
    """Return project-relative path when possible, absolute as fallback."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, out_path.open("wb") as out:
        shutil.copyfileobj(resp, out)


def download_grbench(root: Path, log: dict, project_root: Path) -> None:
    gr_dir = root / "grbench"
    gr_dir.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(HF_TREE_URL) as resp:
        tree = json.load(resp)

    json_files = sorted(item["path"] for item in tree if item.get("path", "").endswith(".json"))
    readme_files = [item["path"] for item in tree if item.get("path") == "README.md"]

    downloaded = []
    for rel in json_files + readme_files:
        url = f"{HF_FILE_BASE}/{urllib.parse.quote(rel)}"
        out = gr_dir / rel
        download_file(url, out)
        downloaded.append(
            {
                "path": safe_relpath(out, project_root),
                "bytes": out.stat().st_size,
                "sha256": sha256sum(out),
            }
        )

    log["grbench"] = {
        "source": HF_FILE_BASE,
        "files": downloaded,
    }


def download_ogbn_arxiv(root: Path, log: dict, project_root: Path) -> None:
    ogb_dir = root / "ogbn_arxiv"
    ogb_dir.mkdir(parents=True, exist_ok=True)

    zip_path = ogb_dir / "arxiv.zip"
    download_file(OGBN_ARXIV_URL, zip_path)

    extract_dir = ogb_dir / "arxiv"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    log["ogbn_arxiv"] = {
        "source": OGBN_ARXIV_URL,
        "zip": {
            "path": safe_relpath(zip_path, project_root),
            "bytes": zip_path.stat().st_size,
            "sha256": sha256sum(zip_path),
        },
        "extracted_root": safe_relpath(extract_dir, project_root),
    }


def download_snap_amazon(root: Path, log: dict, project_root: Path) -> None:
    snap_dir = root / "snap_com_amazon"
    snap_dir.mkdir(parents=True, exist_ok=True)

    gz_path = snap_dir / "com-amazon.ungraph.txt.gz"
    txt_path = snap_dir / "com-amazon.ungraph.txt"

    download_file(SNAP_AMAZON_URL, gz_path)

    with gzip.open(gz_path, "rb") as src, txt_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)

    log["snap_com_amazon"] = {
        "source": SNAP_AMAZON_URL,
        "gz": {
            "path": safe_relpath(gz_path, project_root),
            "bytes": gz_path.stat().st_size,
            "sha256": sha256sum(gz_path),
        },
        "txt": {
            "path": safe_relpath(txt_path, project_root),
            "bytes": txt_path.stat().st_size,
            "sha256": sha256sum(txt_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download datasets for checkpoint 1")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Directory where raw datasets will be downloaded",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("reports/dataset_download_log.json"),
        help="JSON file to write download metadata",
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    data_root = (project_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root.resolve()
    log_path = (project_root / args.log_path).resolve() if not args.log_path.is_absolute() else args.log_path.resolve()

    log: dict[str, object] = {
        "data_root": safe_relpath(data_root, project_root),
    }

    try:
        download_grbench(data_root, log, project_root)
        download_ogbn_arxiv(data_root, log, project_root)
        download_snap_amazon(data_root, log, project_root)
    except urllib.error.URLError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"Completed. Log saved to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
