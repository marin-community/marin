# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observe an explicitly bound source runtime before executing source commands."""

import json

from shellbox.machine import Command, ExitReason, Machine
from taskcompendium.harbor.protected_runtime import (
    RUNTIME_OBSERVATION_SCRIPT,
    ExpectedRuntime,
    RuntimeObservation,
    RuntimeRole,
)
from upath import UPath

from harbor.trial.errors import RuntimeServiceError
from harbor.utils.artifact_writer import get_artifact_writer

RUNTIME_OBSERVATION_OUTPUT_BYTES = 256 * 1024


async def _persist_observation(destination: UPath, payload: str) -> None:
    writer = get_artifact_writer()
    future = await writer.submit_required(
        destination, lambda: payload, description="Protected source runtime observation"
    )
    await writer.await_write(future)


async def observe_runtime(
    machine: Machine, expected: ExpectedRuntime, role: RuntimeRole, destination: UPath, timeout: float
) -> None:
    """Persist actual interpreter evidence before comparing its frozen expectations."""
    try:
        result = await machine.run(
            Command(
                ("python3", "-c", RUNTIME_OBSERVATION_SCRIPT, json.dumps(sorted(expected.files))),
                cwd="/",
                timeout=timeout,
                output_limit_bytes=RUNTIME_OBSERVATION_OUTPUT_BYTES,
            )
        )
    except Exception as error:
        await _persist_observation(destination, json.dumps({
            "role": role.value, "exit_code": None, "exit_reason": None,
            "stdout": None, "stderr": None, "exception_type": type(error).__name__, "detail": str(error),
        }))
        raise RuntimeServiceError(f"Protected {role.value} runtime observation transport failed") from error
    payload = json.dumps(
        {
            "role": role.value,
            "exit_code": result.exit_code,
            "exit_reason": result.reason.value,
            "stdout": result.stdout.decode(errors="replace"),
            "stderr": result.stderr.decode(errors="replace"),
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
        }
    )
    await _persist_observation(destination, payload)
    if result.reason != ExitReason.EXITED or result.exit_code != 0 or result.stdout_truncated:
        raise RuntimeServiceError(f"Protected {role.value} runtime observation did not complete")
    try:
        expected.require_observation(RuntimeObservation.model_validate_json(result.stdout))
    except ValueError as error:
        raise RuntimeServiceError(
            f"Protected {role.value} runtime differs from its frozen environment: {error}"
        ) from error
