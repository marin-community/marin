# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check Harbor's standalone lock and run its offline runtime tests."""

import os
import subprocess
import sys
from pathlib import Path

TEST_ROOTS = (
    "vendor/harbor/tests/unit",
    "lib/shellbox/tests/test_http_service.py",
    "lib/shellbox/tests/test_iris_machine.py",
)


def run_harbor_tests(repo_root: Path, extra_args: tuple[str, ...] = ()) -> int:
    """Run the same isolated Harbor checks locally and in CI."""
    lock = subprocess.run(["uv", "lock", "--project", "vendor/harbor", "--check"], cwd=repo_root, check=False)
    if lock.returncode:
        return lock.returncode
    pytest_command = [
        "uv",
        "run",
        "--isolated",
        "--project",
        "vendor/harbor",
        "--frozen",
        "--extra",
        "runtime",
        "--extra",
        "analysis",
        "--group",
        "test",
        "python",
        "-m",
        "pytest",
        "-q",
        "-c",
        "vendor/harbor/pyproject.toml",
    ]
    result = subprocess.run(
        [*pytest_command, *TEST_ROOTS, *extra_args],
        cwd=repo_root,
        check=False,
    )
    if result.returncode:
        return result.returncode
    # The acquisition tests' neighboring conftest imports Marin and Fray, which
    # are outside Harbor's independent environment.
    result = subprocess.run(
        [
            *pytest_command,
            "--noconftest",
            "experiments/post_training/task_curation/tests/test_harbor_hub.py",
            *extra_args,
        ],
        cwd=repo_root,
        check=False,
    )
    if result.returncode:
        return result.returncode
    return subprocess.run(
        [
            *pytest_command,
            "--noconftest",
            "vendor/harbor/tests/marin",
            "experiments/evaluation/images/tau3/test_build.py",
            *extra_args,
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "lib/marin/src")},
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(run_harbor_tests(Path(__file__).resolve().parents[2], tuple(sys.argv[1:])))
