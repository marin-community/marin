# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit Iris jobs from a filtered workspace.

This wrapper exists for large Marin workspaces where the raw ``iris job run``
CLI would otherwise bundle tracked experiment artifacts that are irrelevant to
the submitted job. It keeps the Iris execution path, but stages a smaller
temporary workspace before delegating to the standard Iris CLI.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from iris.cluster.client.bundle import collect_workspace_files
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_WORKING_DIR_EXCLUDES = [
    ".git",
    ".venv",
    ".uv-cache",
    ".pytest_cache",
    ".ruff_cache",
    ".agents",
    ".claude",
    ".codex",
    ".specstory",
    "docs/",
    "data_browser",
    "logs/",
    "tests/snapshots",
    "wandb",
    "**/.DS_Store",
    "**/.idea",
    "**/.idea/**",
    "**/__pycache__",
    "**/*.pyc",
    "**/*.pack",
    "lib/levanter/docs",
    "lib/dupekit/target",
    "lib/dupekit/**/*.so",
    "experiments/domain_phase_mix/offline_rl/artifacts",
    "experiments/domain_phase_mix/offline_rl/policy_assets",
    "experiments/domain_phase_mix/validation_artifacts",
    "experiments/domain_phase_mix/exploratory/two_phase_starcoder_models.pkl",
    "experiments/domain_phase_mix/exploratory/three_phase_plots",
    "experiments/domain_phase_mix/exploratory/holdout_plots",
    "experiments/domain_phase_mix/exploratory/holdout_plots_116",
    "experiments/domain_phase_mix/exploratory/three_phase_starcoder_plots",
    "experiments/domain_phase_mix/exploratory/two_phase_starcoder_plots",
    "experiments/domain_phase_mix/exploratory/two_phase_starcoder_50",
    "experiments/domain_phase_mix/exploratory/overfitting_gap_general_plots",
    "experiments/domain_phase_mix/exploratory/*.png",
    "experiments/domain_phase_mix/exploratory/two_phase_many/surrogate_search/*.png",
    "experiments/domain_phase_mix/exploratory/two_phase_many/surrogate_search/*.csv",
    "experiments/domain_phase_mix/exploratory/two_phase_many/surrogate_search/*.json",
    "experiments/domain_phase_mix/exploratory/two_phase_many/surrogate_search/*.md",
    "experiments/domain_phase_mix/exploratory/two_phase_many/surrogate_search/*.tex",
    "experiments/domain_phase_mix/exploratory/two_phase_many/dsre_ceq_debug/*.png",
    "experiments/domain_phase_mix/exploratory/two_phase_many/dsre_ceq_debug/*.csv",
    "experiments/domain_phase_mix/exploratory/two_phase_many/*.png",
]


def _matches_exclude(relative_path: str, pattern: str) -> bool:
    relative_path = relative_path.replace("\\", "/")
    if pattern.endswith("/"):
        prefix = pattern.rstrip("/")
        return relative_path == prefix or relative_path.startswith(f"{prefix}/")
    if "/" not in pattern:
        return relative_path == pattern or relative_path.startswith(f"{pattern}/")
    return fnmatch.fnmatch(relative_path, pattern) or relative_path == pattern or relative_path.startswith(f"{pattern}/")


def _should_exclude(relative_path: str, patterns: list[str]) -> bool:
    return any(_matches_exclude(relative_path, pattern) for pattern in patterns)


def _create_filtered_workspace(
    workspace: Path,
    exclude_patterns: list[str],
) -> Path:
    files = collect_workspace_files(workspace)
    temp_workspace = Path(tempfile.mkdtemp(prefix="iris_workspace_"))

    copied_count = 0
    for file_path in files:
        relative_path = file_path.relative_to(workspace).as_posix()
        if _should_exclude(relative_path, exclude_patterns):
            continue
        destination = temp_workspace / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination)
        copied_count += 1

    logger.info(
        "Prepared filtered Iris workspace with %d files at %s",
        copied_count,
        temp_workspace,
    )
    return temp_workspace


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Iris config file passed through to `iris --config ... job run`.",
    )
    parser.add_argument(
        "--working-dir-exclude",
        action="append",
        default=[],
        help="Additional glob-style path pattern to exclude from the staged workspace.",
    )
    parser.add_argument(
        "--no-default-working-dir-excludes",
        action="store_true",
        help="Disable the standard Marin workspace exclusions.",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to `iris job run`. Start with `--`.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.cmd:
        raise SystemExit("Expected arguments for `iris job run` after `--`.")
    forwarded_args = list(args.cmd)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    if not forwarded_args:
        raise SystemExit("Expected Iris `job run` arguments after `--`.")

    exclude_patterns = [] if args.no_default_working_dir_excludes else list(DEFAULT_WORKING_DIR_EXCLUDES)
    exclude_patterns.extend(args.working_dir_exclude)

    workspace_root = Path.cwd().resolve()
    config_path = args.config.resolve()
    iris_cmd = [
        "uv",
        "run",
        "iris",
        "--config",
        str(config_path),
        "job",
        "run",
        *forwarded_args,
    ]

    logger.info("Running Iris job submission via filtered workspace")
    logger.info("Command: %s", subprocess.list2cmdline(iris_cmd))
    temp_workspace = _create_filtered_workspace(workspace_root, exclude_patterns)
    try:
        result = subprocess.run(
            iris_cmd,
            cwd=temp_workspace,
            check=False,
        )
        return result.returncode
    finally:
        shutil.rmtree(temp_workspace, ignore_errors=True)


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    raise SystemExit(main())
