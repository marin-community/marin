#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A/B benchmark: exact paragraph dedup on main vs arrow-scatter-reduce.

Submits 4 jobs total:
  - main  @ 10% files  (prefix: exact-bench-main-10pct)
  - branch @ 10% files (prefix: exact-bench-fast-10pct)
  - main  @ full       (prefix: exact-bench-main-full)
  - branch @ full      (prefix: exact-bench-fast-full)

Creates a worktree for main, patches both branches with ZEPHYR_FORCE_EXTERNAL_MERGE,
deletes stale output buckets, and submits all jobs via iris.
"""

import os
import shutil
import subprocess
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKTREE_DIR = os.path.join(tempfile.gettempdir(), "marin-main-bench")
EXPERIMENT = "experiments/dedup/nemotron_1split_exact.py"

# 10% of ~11108 files ≈ 1111
TEN_PCT_FILES = 1111

VARIANTS = [
    {"label": "10pct", "max_files": str(TEN_PCT_FILES)},
]

MAIN_PREFIX_FMT = "exact-bench-main-{label}"
BRANCH_PREFIX_FMT = "exact-bench-fast-{label}"

IRIS_BASE_CMD = [
    "uv",
    "run",
    "iris",
    "--config=lib/iris/examples/marin-dev.yaml",
    "job",
    "run",
    "--no-wait",
    "--memory=4g",
    "--cpu=0",
    "--region=europe-west4",
]

FORCE_ENV = ("ZEPHYR_FORCE_EXTERNAL_MERGE", "1")


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def run_live(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def delete_old_outputs() -> None:
    print("\n=== Deleting previous benchmark outputs ===")
    for variant in VARIANTS:
        for fmt in (MAIN_PREFIX_FMT, BRANCH_PREFIX_FMT):
            prefix = fmt.format(**variant)
            bucket_path = f"gs://marin-tmp-eu-west4/ttl=1d/{prefix}"
            result = subprocess.run(
                ["gcloud", "storage", "rm", "-r", bucket_path],
                text=True,
                capture_output=True,
            )
            if result.returncode == 0:
                print(f"  Deleted {bucket_path}")
            else:
                print(f"  Nothing to delete at {bucket_path} (or already gone)")


def setup_main_worktree() -> str:
    print("\n=== Setting up main worktree ===")
    if os.path.exists(WORKTREE_DIR):
        print(f"  Removing existing worktree at {WORKTREE_DIR}")
        run(["git", "worktree", "remove", "--force", WORKTREE_DIR], cwd=REPO_ROOT, check=False)
        if os.path.exists(WORKTREE_DIR):
            shutil.rmtree(WORKTREE_DIR)

    BENCH_BRANCH = "arrow-scatter-test"
    run(["git", "branch", "-D", BENCH_BRANCH], cwd=REPO_ROOT, check=False)
    run(["git", "worktree", "add", "-b", BENCH_BRANCH, WORKTREE_DIR, "main"], cwd=REPO_ROOT)
    print(f"  Worktree created at {WORKTREE_DIR} (branch {BENCH_BRANCH})")
    return WORKTREE_DIR


def patch_main_worktree(worktree: str) -> None:
    print("\n=== Patching main worktree plan.py ===")
    plan_py = os.path.join(worktree, "lib/zephyr/src/zephyr/plan.py")

    with open(plan_py) as f:
        content = f.read()

    old = (
        "    use_external = (\n"
        "        external_sort_dir is not None\n"
        "        and isinstance(shard, ScatterShard)\n"
        "        and shard.needs_external_sort(_TaskResources.from_environment().memory_bytes)\n"
        "    )"
    )
    new = (
        '    force_external = os.environ.get("ZEPHYR_FORCE_EXTERNAL_MERGE", "").lower() in ("1", "true", "yes")\n'
        "    use_external = (\n"
        "        external_sort_dir is not None\n"
        "        and isinstance(shard, ScatterShard)\n"
        "        and (force_external or shard.needs_external_sort(_TaskResources.from_environment().memory_bytes))\n"
        "    )"
    )

    if old not in content:
        if "ZEPHYR_FORCE_EXTERNAL_MERGE" in content:
            print("  Already patched, skipping")
            return
        raise RuntimeError("Cannot patch main's plan.py — expected code not found")

    content = content.replace(old, new)
    with open(plan_py, "w") as f:
        f.write(content)
    print("  Patched plan.py with ZEPHYR_FORCE_EXTERNAL_MERGE support")


def copy_experiment_to_worktree(worktree: str) -> None:
    print("\n=== Copying experiment script to main worktree ===")
    src = os.path.join(REPO_ROOT, EXPERIMENT)
    dst_dir = os.path.join(worktree, os.path.dirname(EXPERIMENT))
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(worktree, EXPERIMENT)

    with open(src) as f:
        content = f.read()

    with open(dst, "w") as f:
        f.write(content)
    print(f"  Copied {EXPERIMENT} → {dst}")


def submit_job(cwd: str, label: str, output_prefix: str, max_files: str) -> None:
    print(f"\n=== Submitting job: {label} (prefix={output_prefix}, max_files={max_files}) ===")
    cmd = [
        *IRIS_BASE_CMD,
        "-e",
        FORCE_ENV[0],
        FORCE_ENV[1],
        "-e",
        "OUTPUT_PREFIX",
        output_prefix,
        "-e",
        "MAX_FILES",
        max_files,
        "--",
        "python",
        EXPERIMENT,
    ]
    run_live(cmd, cwd=cwd)


def main() -> None:
    print("=" * 70)
    print("A/B Benchmark (exact dedup): main vs arrow-scatter-reduce")
    print("  Variants: 10% files only")
    print("=" * 70)

    delete_old_outputs()
    worktree = setup_main_worktree()
    patch_main_worktree(worktree)
    copy_experiment_to_worktree(worktree)

    for variant in VARIANTS:
        label = variant["label"]
        max_files = variant["max_files"]

        main_prefix = MAIN_PREFIX_FMT.format(**variant)
        branch_prefix = BRANCH_PREFIX_FMT.format(**variant)

        submit_job(worktree, f"main-{label}", main_prefix, max_files)
        submit_job(REPO_ROOT, f"branch-{label}", branch_prefix, max_files)

    print("\n" + "=" * 70)
    print("2 jobs submitted. Monitor via:")
    print("  uv run iris --config=lib/iris/examples/marin-dev.yaml job list")
    print("=" * 70)


if __name__ == "__main__":
    main()
