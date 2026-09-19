# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Policy checks for Iris jobs launched by GitHub Actions."""

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB_CONFIG_DIRS = (REPO_ROOT / ".github" / "workflows", REPO_ROOT / ".github" / "actions")
MAX_IRIS_JOB_TIMEOUT = 24 * 60 * 60
IRIS_JOB_RUN = re.compile(r"(?:^\s*(?:uv run\s+)?|\$\()\S*iris\b.*\bjob\s+run\b")
STORAGE_REPORT = re.compile(r"\bscripts/ops/storage/generate_report\.py\b")


def _run_blocks(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        run = value.get("run")
        if isinstance(run, str):
            yield run
        for child in value.values():
            yield from _run_blocks(child)
    elif isinstance(value, list):
        for child in value:
            yield from _run_blocks(child)


def _timeout_seconds(command: str, option: str) -> int | None:
    match = re.search(rf"{re.escape(option)}(?:=|\s+)(\d+)\b", command)
    return int(match.group(1)) if match else None


def test_ci_iris_jobs_have_bounded_server_timeouts():
    failures = []
    command_types = ((IRIS_JOB_RUN, "--timeout"), (STORAGE_REPORT, "--job-timeout"))

    for config_dir in GITHUB_CONFIG_DIRS:
        for path in sorted((*config_dir.rglob("*.yaml"), *config_dir.rglob("*.yml"))):
            document = yaml.safe_load(path.read_text())
            for run_block in _run_blocks(document):
                command = re.sub(r"\\\s*\n\s*", " ", run_block)
                for line in command.splitlines():
                    for pattern, option in command_types:
                        if not pattern.search(line):
                            continue
                        timeout = _timeout_seconds(line, option)
                        if timeout is None or not 0 < timeout <= MAX_IRIS_JOB_TIMEOUT:
                            failures.append(f"{path.relative_to(REPO_ROOT)}: {option}={timeout!r}")

    assert not failures, "CI Iris jobs need server timeouts of at most 24 hours:\n" + "\n".join(failures)
