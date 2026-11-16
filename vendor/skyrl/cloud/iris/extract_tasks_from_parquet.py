#!/usr/bin/env python3
"""Extract Harbor-compatible task folders from a task parquet snapshot.

Downloads a parquet (or raw task dir) from a HuggingFace repo (or reads a local
path) and materializes the canonical ``task-xxxx`` directory structure. Invoked as
a subprocess by ``cloud.iris.rl_data.resolve_rl_train_data`` to stage RL task data.

Supports two dataset formats (auto-detected):
1. Parquet with a ``task_binary`` column (tar archives, needs extraction).
2. Raw task directories (already have ``instruction.md`` files).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from cloud.iris import tasks_parquet as tpc
from cloud.iris.hf_datasets import download_hf_dataset, is_raw_tasks_directory


def copy_raw_tasks(source_dir: Path, output_dir: Path, on_exist: str = "skip") -> int:
    """Copy raw task directories to the output directory.

    Returns the number of tasks copied.
    """
    task_dirs = tpc.find_tasks(source_dir, recursive=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for task_dir in task_dirs:
        rel_path = task_dir.relative_to(source_dir)
        dest_dir = output_dir / rel_path

        if dest_dir.exists():
            if on_exist == "skip":
                continue
            elif on_exist == "error":
                raise FileExistsError(f"Target exists: {dest_dir}")
            elif on_exist == "overwrite":
                shutil.rmtree(dest_dir)

        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(task_dir, dest_dir)
        copied += 1

    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Harbor tasks from a parquet file.")
    parser.add_argument(
        "--parquet",
        required=True,
        help=(
            "Path to the task parquet file to extract, a directory containing parquet files, "
            "or a Hugging Face dataset repo ID. If a repo ID is provided, the script downloads "
            "it before extraction. Also supports raw task directories (auto-detected)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where extracted task folders will be written.",
    )
    parser.add_argument(
        "--on_exist",
        choices=("skip", "overwrite", "error"),
        default="error",
        help="How to handle existing task folders (default: error).",
    )
    parser.add_argument(
        "--tasks_revision",
        default=None,
        help="Optional revision/commit to download when --parquet references a Hugging Face repo.",
    )
    parser.add_argument(
        "--parquet_name",
        default=None,
        help="Optional parquet filename to select when the source contains multiple parquet files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print actions without extracting anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parquet_input = args.parquet
    source_dir: Path | None = None  # For raw task directories
    parquet_paths: list[Path] = []  # For parquet extraction (multi-shard support)

    candidate_path = Path(parquet_input).expanduser()
    if candidate_path.exists():
        if candidate_path.is_dir():
            if is_raw_tasks_directory(candidate_path):
                print(f"[extract] Detected raw task directory: {candidate_path}")
                source_dir = candidate_path
            else:
                parquet_files = sorted(candidate_path.rglob("*.parquet"))
                if not parquet_files:
                    raise FileNotFoundError(f"No parquet files or raw tasks found under directory: {candidate_path}")
                if args.parquet_name:
                    matching = [p for p in parquet_files if p.name == args.parquet_name]
                    if not matching:
                        raise FileNotFoundError(
                            f"Could not find parquet named '{args.parquet_name}' under {candidate_path}"
                        )
                    parquet_paths = [matching[0]]
                else:
                    parquet_paths = list(parquet_files)
        else:
            parquet_paths = [candidate_path.resolve()]
    else:
        # Not a local path - treat as a HuggingFace repo.
        print(f"[extract] Treating '{parquet_input}' as a Hugging Face dataset repo; downloading snapshot...")
        snapshot_dir = Path(download_hf_dataset(parquet_input, revision=args.tasks_revision))

        if is_raw_tasks_directory(snapshot_dir):
            print(f"[extract] Detected raw task directory (no extraction needed): {snapshot_dir}")
            source_dir = snapshot_dir
        else:
            parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(
                    f"No parquet files or raw tasks found under downloaded repo '{parquet_input}' "
                    f"(path: {snapshot_dir})"
                )
            if args.parquet_name:
                matching = [p for p in parquet_files if p.name == args.parquet_name]
                if not matching:
                    available = "\n  - ".join(str(p.relative_to(snapshot_dir)) for p in parquet_files[:20])
                    raise FileNotFoundError(
                        f"Could not find parquet named '{args.parquet_name}' in repo '{parquet_input}'. "
                        f"Available examples:\n  - {available}"
                    )
                parquet_paths = [matching[0]]
            else:
                parquet_paths = list(parquet_files)
            if len(parquet_paths) == 1:
                print(f"[extract] Using parquet file: {parquet_paths[0]}")
            else:
                print(f"[extract] Found {len(parquet_paths)} parquet shards; will extract all of them.")

    output_dir = Path(args.output_dir).expanduser().resolve()

    # Early exit if output already exists.
    if args.on_exist == "error" and output_dir.exists() and any(output_dir.iterdir()):
        print(
            "[extract] Output directory already contains data and on_exist=error; "
            "assuming tasks have already been extracted. Exiting gracefully."
        )
        return

    if args.dry_run:
        if source_dir:
            print(f"[extract] DRY RUN: Would copy raw tasks from {source_dir} to {output_dir}")
        else:
            for p in parquet_paths:
                print(f"[extract] DRY RUN: Would extract parquet {p} to {output_dir}")
        return

    if source_dir:
        print(f"[extract] Copying raw tasks from: {source_dir}")
        print(f"[extract] Output directory: {output_dir} (on_exist={args.on_exist})")
        num_copied = copy_raw_tasks(source_dir, output_dir, on_exist=args.on_exist)
        print(f"[extract] Done. Copied {num_copied} task directories.")
    else:
        assert parquet_paths, "No parquet files resolved"
        # For multi-shard runs, the first shard uses --on_exist as-is; subsequent
        # shards treat 'error' as 'skip' so an empty starting dir doesn't blow up
        # on the 2nd shard. 'overwrite' / 'skip' pass through unchanged.
        for i, parquet_path in enumerate(parquet_paths):
            label = f"shard {i + 1}/{len(parquet_paths)}: " if len(parquet_paths) > 1 else ""
            print(f"[extract] Extracting parquet ({label}{parquet_path})")
            print(f"[extract] Output directory: {output_dir} (on_exist={args.on_exist})")
            eff = args.on_exist if (i == 0 or args.on_exist != "error") else "skip"
            tpc.from_parquet(str(parquet_path), str(output_dir), on_exist=eff)
        print(f"[extract] Done. Extracted {len(parquet_paths)} parquet file(s).")


if __name__ == "__main__":
    main()
