# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the distributed diagnostic probe in a running task."""

import json
from pathlib import Path
from typing import Protocol

from rigging.timing import Timestamp

from iris.cluster.runtime import distributed_diagnostic_probe
from iris.cluster.runtime.profile import ExecResult

DEFAULT_COLLECTOR_TIMEOUT_SECONDS = distributed_diagnostic_probe.DEFAULT_COLLECTOR_TIMEOUT_SECONDS
MIN_COLLECTOR_TIMEOUT_SECONDS = distributed_diagnostic_probe.MIN_COLLECTOR_TIMEOUT_SECONDS
MAX_COLLECTOR_TIMEOUT_SECONDS = distributed_diagnostic_probe.MAX_COLLECTOR_TIMEOUT_SECONDS
PROBE_EXIT_HEADROOM_SECONDS = 10

_PROBE_PATH = Path(distributed_diagnostic_probe.__file__)


class DistributedDiagnosticDispatch(Protocol):
    pyspy_bin: str

    def run_diagnostic_probe(
        self,
        probe_path: Path,
        arguments: list[str],
        *,
        timeout: int,
    ) -> ExecResult:
        """Copy or execute the maintained probe in the target environment."""
        ...


def capture_distributed_diagnostic(
    dispatch: DistributedDiagnosticDispatch,
    *,
    pid: str,
    source: str,
    attempt_id: int | None,
    timeout: int = DEFAULT_COLLECTOR_TIMEOUT_SECONDS,
) -> bytes:
    """Capture bounded task-local evidence and return its JSON bundle."""
    if not MIN_COLLECTOR_TIMEOUT_SECONDS <= timeout <= MAX_COLLECTOR_TIMEOUT_SECONDS:
        raise ValueError(
            f"collector timeout must be between {MIN_COLLECTOR_TIMEOUT_SECONDS} "
            f"and {MAX_COLLECTOR_TIMEOUT_SECONDS} seconds"
        )

    arguments = [
        "--pid",
        pid,
        "--source",
        source,
        "--captured-at",
        Timestamp.now().as_naive_utc().isoformat(),
        "--timeout",
        str(timeout),
        "--py-spy",
        dispatch.pyspy_bin,
    ]
    if attempt_id is not None:
        arguments.extend(["--attempt-id", str(attempt_id)])
    result = dispatch.run_diagnostic_probe(
        _PROBE_PATH,
        arguments,
        timeout=timeout + PROBE_EXIT_HEADROOM_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"distributed diagnostic probe failed (exit {result.returncode}): {result.stderr}")
    if len(result.stdout) > distributed_diagnostic_probe.MAX_BUNDLE_BYTES:
        raise RuntimeError("distributed diagnostic probe exceeded the profile transport limit")
    try:
        bundle = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"distributed diagnostic probe returned invalid JSON: {exc}") from exc
    if bundle.get("schema_version") != distributed_diagnostic_probe.SCHEMA_VERSION:
        raise RuntimeError(f"unsupported distributed diagnostic schema: {bundle.get('schema_version')!r}")
    return result.stdout
