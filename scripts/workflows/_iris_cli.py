# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared Iris CLI plumbing: binary discovery, job-list JSON parsing, and core data types.

This module is internal to scripts/workflows/. The leading underscore signals that no
external caller should import it directly; promote to a public module only if a third
workflow script acquires the same dependency.
"""

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal


class IrisJobState(StrEnum):
    PENDING = "JOB_STATE_PENDING"
    BUILDING = "JOB_STATE_BUILDING"
    RUNNING = "JOB_STATE_RUNNING"
    SUCCEEDED = "JOB_STATE_SUCCEEDED"
    FAILED = "JOB_STATE_FAILED"
    CANCELLED = "JOB_STATE_CANCELLED"


_ACTIVE_STATES: frozenset[IrisJobState] = frozenset({IrisJobState.PENDING, IrisJobState.BUILDING, IrisJobState.RUNNING})


@dataclass(frozen=True)
class IrisJobStatus:
    job_id: str
    state: IrisJobState
    error: str | None


@dataclass(frozen=True)
class DiagnosticsRequest:
    job_id: str
    output_dir: Path
    iris_config: Path | None
    provider: Literal["gcp", "coreweave"]
    project: str | None
    controller_label: str | None
    namespace: str | None
    service_account: str | None
    ssh_key: Path | None
    kubeconfig: Path | None
    controller_url: str | None = None


def iris_command(repo_root: Path) -> list[str]:
    """Return the argv prefix to invoke iris.

    Order:
    1. `<repo_root>/.venv/bin/iris` if it exists.
    2. ["uv", "run", "--package", "iris", "iris"] otherwise.
    """
    venv_iris = repo_root / ".venv" / "bin" / "iris"
    if venv_iris.exists():
        return [str(venv_iris)]
    return ["uv", "run", "--package", "iris", "iris"]


def _iris_flags(iris_config: Path | None, controller_url: str | None) -> list[str]:
    """Return the config/URL flags to prepend to an iris subcommand.

    Exactly one of iris_config and controller_url should be provided.
    If both are given, controller_url takes precedence.
    """
    if controller_url is not None:
        return [f"--controller-url={controller_url}"]
    if iris_config is not None:
        return [f"--config={iris_config}"]
    return []


def job_status(
    job_id: str,
    *,
    iris_config: Path | None,
    prefix: str | None = None,
    repo_root: Path,
    controller_url: str | None = None,
) -> IrisJobStatus:
    """Run `iris job list --json --prefix <prefix>` and select the row whose job_id equals job_id.

    Args:
        job_id: The exact job ID to look up.
        iris_config: Path to the iris config file, or None to omit --config.
        prefix: Optional prefix filter passed to `iris job list --prefix`. When None,
            the job_id itself is used as the prefix so the result set is small.
        repo_root: Repository root, used to locate the iris binary.
        controller_url: Optional controller URL (e.g. http://localhost:PORT) passed as
            --controller-url. Takes precedence over iris_config when both are given.

    Returns:
        IrisJobStatus for the matching job.

    Raises:
        RuntimeError: If the iris invocation fails or the JSON is malformed.
        LookupError: If no row with the requested job_id is present in the output.
    """
    cmd = iris_command(repo_root)
    cmd += _iris_flags(iris_config, controller_url)
    effective_prefix = prefix if prefix is not None else job_id
    cmd += ["job", "list", "--json", "--prefix", effective_prefix]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"iris job list failed (exit {result.returncode}): {result.stderr.strip()}")

    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"iris job list returned malformed JSON: {exc}") from exc

    if not isinstance(rows, list):
        raise RuntimeError(f"iris job list returned unexpected JSON shape (expected list): {type(rows)}")

    for row in rows:
        if row.get("job_id") == job_id:
            raw_state = row.get("state", "")
            try:
                state = IrisJobState(raw_state)
            except ValueError:
                # Unknown state surfaced by iris — treat as a terminal failure.
                state = IrisJobState.FAILED
            error: str | None = row.get("error") or None
            return IrisJobStatus(job_id=job_id, state=state, error=error)

    raise LookupError(f"Job not found in iris job list output: {job_id!r}")


def wait_for_job(
    job_id: str,
    *,
    iris_config: Path | None,
    prefix: str | None,
    poll_interval: float,
    timeout: float | None,
    repo_root: Path,
    controller_url: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> IrisJobStatus:
    """Poll until the job reaches a terminal state.

    Args:
        job_id: The exact job ID to wait on.
        iris_config: Path to the iris config file, or None.
        prefix: Optional prefix for the iris job list query.
        poll_interval: Seconds between polls.
        timeout: Maximum seconds to wait, or None for no timeout.
        repo_root: Repository root used to locate the iris binary.
        controller_url: Optional controller URL passed as --controller-url. Takes
            precedence over iris_config when both are given.
        sleep: Injectable sleep function so tests can drive virtual time.
        monotonic: Injectable clock function for timeout measurement.

    Returns:
        IrisJobStatus when a terminal state is reached.

    Raises:
        TimeoutError: When timeout elapses before a terminal state is reached.
    """
    start = monotonic()
    while True:
        status = job_status(
            job_id, iris_config=iris_config, prefix=prefix, repo_root=repo_root, controller_url=controller_url
        )
        if status.state not in _ACTIVE_STATES:
            return status

        if timeout is not None and (monotonic() - start) >= timeout:
            raise TimeoutError(f"Timed out waiting for job {job_id!r} after {timeout}s")

        sleep(poll_interval)
