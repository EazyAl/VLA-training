#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MechaLabs dataset snapshot.")
    parser.add_argument("--repo-id", default="CRPlab/MechaLabs", help="HF repo id.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/mechalabs"), help="Destination dir.")
    parser.add_argument("--allow-patterns", nargs="*", default=None, help="Optional patterns to limit download.")
    parser.add_argument("--exclude", nargs="*", default=None, help="Exclude patterns.")
    args = parser.parse_args()

    files = list_repo_files(args.repo_id)
    if len(files) <= 2:  # only README + gitattributes
        print(f"Repo {args.repo_id} appears empty (files: {files}). Nothing to download.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.out_dir,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.exclude,
    )
    print(f"Downloaded snapshot to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

