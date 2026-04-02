#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A/B benchmark: run nemotron_1slice_fuzzy on main vs arrow-scatter-reduce.

Creates a worktree for main, patches both branches with ZEPHYR_FORCE_EXTERNAL_MERGE,
deletes stale output buckets, and submits both jobs via iris.
"""

import os
import shutil
import subprocess
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKTREE_DIR = os.path.join(tempfile.gettempdir(), "marin-main-bench")
EXPERIMENT = "experiments/dedup/nemotron_1slice_fuzzy.py"

# Output prefix names (used in marin_temp_bucket calls inside the experiment scripts)
MAIN_PREFIX = "arrow-scatter-bench-main"
BRANCH_PREFIX = "arrow-scatter-bench-fast"

IRIS_BASE_CMD = [
    "uv",
    "run",
    "iris",
    "--config=lib/iris/examples/marin.yaml",
    "job",
    "run",
    "--no-wait",
    "--memory=0.5g",
    "--cpu=0",
    "--region=europe-west4",
]

# The env var that forces external merge in both old and new code paths
FORCE_ENV = ("ZEPHYR_FORCE_EXTERNAL_MERGE", "1")


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def run_live(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run with stdout/stderr going to terminal."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def delete_old_outputs() -> None:
    """Delete previous benchmark outputs from GCS temp buckets."""
    print("\n=== Deleting previous benchmark outputs ===")
    for prefix in (MAIN_PREFIX, BRANCH_PREFIX):
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
    """Create a git worktree for main, return path."""
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
    """Apply ZEPHYR_FORCE_EXTERNAL_MERGE env var check to main's plan.py."""
    print("\n=== Patching main worktree plan.py ===")
    plan_py = os.path.join(worktree, "lib/zephyr/src/zephyr/plan.py")

    with open(plan_py) as f:
        content = f.read()

    # Main's code has:
    #     use_external = (
    #         external_sort_dir is not None
    #         and isinstance(shard, ScatterShard)
    #         and shard.needs_external_sort(_TaskResources.from_environment().memory_bytes)
    #     )
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
        print("  WARNING: Could not find expected code pattern in main's plan.py")
        print("  Searching for alternative patterns...")
        # Check if already patched
        if "ZEPHYR_FORCE_EXTERNAL_MERGE" in content:
            print("  Already patched, skipping")
            return
        raise RuntimeError("Cannot patch main's plan.py — expected code not found")

    content = content.replace(old, new)
    with open(plan_py, "w") as f:
        f.write(content)
    print("  Patched plan.py with ZEPHYR_FORCE_EXTERNAL_MERGE support")


def copy_experiment_to_worktree(worktree: str) -> None:
    """Copy the experiment script to main worktree, adjusting the output prefix."""
    print("\n=== Copying experiment script to main worktree ===")
    src = os.path.join(REPO_ROOT, EXPERIMENT)
    dst_dir = os.path.join(worktree, os.path.dirname(EXPERIMENT))
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(worktree, EXPERIMENT)

    with open(src) as f:
        content = f.read()

    # Replace the branch prefix with the main prefix
    content = content.replace(BRANCH_PREFIX, MAIN_PREFIX)
    with open(dst, "w") as f:
        f.write(content)
    print(f"  Copied {EXPERIMENT} → {dst} (prefix={MAIN_PREFIX})")


def submit_job(cwd: str, label: str) -> str:
    """Submit an iris job and return the job ID."""
    print(f"\n=== Submitting job: {label} ===")
    cmd = [
        *IRIS_BASE_CMD,
        "-e",
        FORCE_ENV[0],
        FORCE_ENV[1],
        "--",
        "python",
        EXPERIMENT,
    ]
    run_live(cmd, cwd=cwd)
    return label


def main() -> None:
    print("=" * 60)
    print("A/B Benchmark: main vs arrow-scatter-reduce")
    print("=" * 60)

    # 1. Delete old outputs
    delete_old_outputs()

    # 2. Set up main worktree
    worktree = setup_main_worktree()

    # 3. Patch main worktree
    patch_main_worktree(worktree)

    # 4. Copy experiment to main worktree
    copy_experiment_to_worktree(worktree)

    # 5. Submit main job
    submit_job(worktree, "main")

    # 6. Submit branch job (from repo root)
    submit_job(REPO_ROOT, "arrow-scatter-reduce")

    print("\n" + "=" * 60)
    print("Both jobs submitted. Monitor via:")
    print("  uv run iris --config=lib/iris/examples/marin.yaml job list")
    print("=" * 60)


if __name__ == "__main__":
    main()
