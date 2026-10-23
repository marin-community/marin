# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check Harbor's standalone lock and run its offline runtime tests."""

import subprocess
import sys
from pathlib import Path

HARBOR_TEST_PATTERNS = (
    "vendor/harbor/tests/unit/models",
    "vendor/harbor/tests/unit/test_job_*.py",
    "vendor/harbor/tests/unit/test_trial_*.py",
    "vendor/harbor/tests/unit/test_verifier*.py",
    "vendor/harbor/tests/unit/test_task_client.py",
    "vendor/harbor/tests/unit/test_archive.py",
    "vendor/harbor/tests/unit/test_benchmark_metadata.py",
    "vendor/harbor/tests/unit/test_environment_preflight.py",
    "vendor/harbor/tests/unit/test_runtime.py",
    "vendor/harbor/tests/unit/test_task_packager.py",
    "vendor/harbor/tests/unit/test_task_packager_multi_step.py",
    "vendor/harbor/tests/unit/test_telemetry_observer.py",
    "vendor/harbor/tests/unit/agents/test_tau3_acp_relay.py",
    "vendor/harbor/tests/unit/agents/installed/test_mini_swe_agent.py",
    "vendor/harbor/tests/unit/agents/installed/test_mini_swe_context.py",
    "vendor/harbor/tests/unit/agents/installed/test_pi.py",
    "vendor/harbor/tests/unit/agents/installed/test_sotopia.py",
    "vendor/harbor/tests/unit/environments/test_daytona*.py",
)


def run_harbor_tests(repo_root: Path, extra_args: tuple[str, ...] = ()) -> int:
    """Run the same isolated Harbor checks locally and in CI."""
    lock = subprocess.run(["uv", "lock", "--project", "vendor/harbor", "--check"], cwd=repo_root, check=False)
    if lock.returncode:
        return lock.returncode
    test_paths = sorted({str(path) for pattern in HARBOR_TEST_PATTERNS for path in repo_root.glob(pattern)})
    pytest_command = [
        "uv",
        "run",
        "--isolated",
        "--project",
        "vendor/harbor",
        "--frozen",
        "--extra",
        "runtime",
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
        [*pytest_command, *test_paths, *extra_args],
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
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(run_harbor_tests(Path(__file__).resolve().parents[2], tuple(sys.argv[1:])))
