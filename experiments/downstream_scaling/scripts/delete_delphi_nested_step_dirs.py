# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delete duplicate nested ``step-N/step-N/`` directories in Delphi checkpoints.

The levanter HF export for Delphi writes each checkpoint as ``hf/step-N/<files>`` AND
``hf/step-N/step-N/<files>`` — the inner directory is a byte-exact duplicate of the
outer. ``discover_hf_checkpoints`` picks the deepest match, so vLLM ends up loading
the inner copy. Removing the inner copy halves storage and lets vLLM resolve to the
outer directory instead.

Before deleting, this script confirms the inner directory is a strict duplicate
(same file names, same sizes). Anything that doesn't match is skipped with a
warning instead of removed.

Usage:
    uv run python experiments/downstream_scaling/scripts/delete_delphi_nested_step_dirs.py \
        --region eu-west4
    uv run python experiments/downstream_scaling/scripts/delete_delphi_nested_step_dirs.py \
        --region eu-west4 --apply
"""

from __future__ import annotations

import argparse

import fsspec

from experiments.downstream_scaling.models.delphi import DELPHI_CHECKPOINTS


def _file_sizes(fs: fsspec.AbstractFileSystem, path: str) -> dict[str, int]:
    return {
        e["name"].rsplit("/", 1)[-1]: e["size"]
        for e in fs.ls(path, detail=True)
        if e["type"] == "file"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, help="GCS region suffix (e.g. eu-west4, us-east5)")
    parser.add_argument("--apply", action="store_true", help="Apply deletes (default: dry-run)")
    args = parser.parse_args()

    bucket = f"marin-{args.region}"
    fs = fsspec.filesystem("gs")

    for slug, rel_path in DELPHI_CHECKPOINTS.items():
        hf_dir = f"{bucket}/{rel_path}"
        outer_dirs = [
            e["name"]
            for e in fs.ls(hf_dir, detail=True)
            if e["type"] == "directory" and e["name"].rstrip("/").rsplit("/", 1)[-1].startswith("step-")
        ]
        if not outer_dirs:
            print(f"NONE   {slug}: no step-* directories under gs://{hf_dir}")
            continue
        for outer in outer_dirs:
            outer = outer.rstrip("/")
            step_name = outer.rsplit("/", 1)[-1]
            inner = f"{outer}/{step_name}"
            if not fs.isdir(inner):
                print(f"OK     no inner: gs://{outer}")
                continue
            outer_files = _file_sizes(fs, outer)
            inner_files = _file_sizes(fs, inner)
            if outer_files != inner_files:
                print(f"SKIP   not a strict duplicate: gs://{inner}")
                print(f"  outer files: {outer_files}")
                print(f"  inner files: {inner_files}")
                continue
            print(f"DELETE gs://{inner}/  ({len(inner_files)} files)")
            if not args.apply:
                continue
            fs.rm(inner, recursive=True)


if __name__ == "__main__":
    main()
