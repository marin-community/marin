#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A/B benchmark: local dedup pipeline on main vs arrow-scatter-reduce.

Generates shared input files once, then runs benchmark_dedup_pipeline.py
in both the current branch and a main worktree, comparing throughput.

Usage:
    uv run python experiments/dedup/bench_local_pipeline.py
    uv run python experiments/dedup/bench_local_pipeline.py --num-docs 500000 --backends threadpool sync
"""

import os
import shutil
import subprocess
import tempfile

import click

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKTREE_DIR = os.path.join(tempfile.gettempdir(), "marin-main-bench-local")
BENCHMARK_SCRIPT = "lib/zephyr/tests/benchmark_dedup_pipeline.py"


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def run_live(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def setup_main_worktree() -> str:
    print("\n=== Setting up main worktree ===")
    if os.path.exists(WORKTREE_DIR):
        print(f"  Removing existing worktree at {WORKTREE_DIR}")
        run(["git", "worktree", "remove", "--force", WORKTREE_DIR], cwd=REPO_ROOT, check=False)
        if os.path.exists(WORKTREE_DIR):
            shutil.rmtree(WORKTREE_DIR)

    # Create a new branch from main so we don't pin main itself to the worktree
    BENCH_BRANCH = "arrow-scatter-test"
    run(["git", "branch", "-D", BENCH_BRANCH], cwd=REPO_ROOT, check=False)
    run(["git", "worktree", "add", "-b", BENCH_BRANCH, WORKTREE_DIR, "main"], cwd=REPO_ROOT)
    print(f"  Worktree created at {WORKTREE_DIR} (branch {BENCH_BRANCH})")
    return WORKTREE_DIR


def generate_input(
    cwd: str,
    input_dir: str,
    num_docs: int,
    words_per_doc: int,
    num_input_files: int,
) -> None:
    """Generate shared input files using the benchmark script's write-input command."""
    print(f"\n=== Generating input ({num_docs:,} docs) ===")
    cmd = [
        "uv",
        "run",
        "python",
        BENCHMARK_SCRIPT,
        "write-input",
        "--output-dir",
        input_dir,
        "--num-docs",
        str(num_docs),
        "--words-per-doc",
        str(words_per_doc),
        "--num-input-files",
        str(num_input_files),
    ]
    run_live(cmd, cwd=cwd)


def run_benchmark(
    cwd: str,
    label: str,
    input_dir: str,
    num_docs: int,
    words_per_doc: int,
    num_input_files: int,
    backends: list[str],
) -> None:
    """Run the benchmark in the given directory."""
    print(f"\n{'=' * 60}")
    print(f"Running benchmark: {label}")
    print(f"  cwd: {cwd}")
    print(f"{'=' * 60}")

    cmd = [
        "uv",
        "run",
        "python",
        BENCHMARK_SCRIPT,
        "benchmark",
        "--input-dir",
        input_dir,
        "--num-docs",
        str(num_docs),
        "--words-per-doc",
        str(words_per_doc),
        "--num-input-files",
        str(num_input_files),
    ]
    for backend in backends:
        cmd.extend(["--backends", backend])

    run_live(cmd, cwd=cwd)


@click.command()
@click.option("--num-docs", type=int, default=1_000_000, help="Number of documents to generate")
@click.option("--words-per-doc", type=int, default=1000, help="Words per document")
@click.option("--num-input-files", type=int, default=10, help="Number of input parquet files")
@click.option(
    "--backends",
    multiple=True,
    type=click.Choice(["sync", "threadpool", "ray"]),
    default=["threadpool"],
    help="Backends to benchmark",
)
def main(
    num_docs: int,
    words_per_doc: int,
    num_input_files: int,
    backends: tuple[str, ...],
) -> None:
    """A/B benchmark: local dedup pipeline on main vs current branch."""
    backends_list = list(backends)

    print("=" * 60)
    print("A/B Local Benchmark: main vs arrow-scatter-reduce")
    print(f"  docs={num_docs:,}  words/doc={words_per_doc}  files={num_input_files}")
    print(f"  backends: {', '.join(backends_list)}")
    print("=" * 60)

    worktree = setup_main_worktree()
    input_dir = tempfile.mkdtemp(prefix="zephyr_ab_input_")

    try:
        # Generate input once from current branch
        generate_input(REPO_ROOT, input_dir, num_docs, words_per_doc, num_input_files)

        # Run on main
        run_benchmark(worktree, "main", input_dir, num_docs, words_per_doc, num_input_files, backends_list)

        # Run on current branch
        run_benchmark(
            REPO_ROOT, "arrow-scatter-reduce", input_dir, num_docs, words_per_doc, num_input_files, backends_list
        )

        print("\n" + "=" * 60)
        print("A/B benchmark complete. Compare results above.")
        print("=" * 60)

    finally:
        print(f"\nCleaning up input directory {input_dir}...")
        shutil.rmtree(input_dir, ignore_errors=True)
        print(f"Cleaning up worktree {worktree}...")
        run(["git", "worktree", "remove", "--force", worktree], cwd=REPO_ROOT, check=False)
        if os.path.exists(worktree):
            shutil.rmtree(worktree, ignore_errors=True)


if __name__ == "__main__":
    main()
