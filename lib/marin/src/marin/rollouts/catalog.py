# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog discovery records for evaluation and reinforcement-learning rollouts."""

import logging
import time
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import ClassVar

from finelog.client import FlushResult, LogClient, TableSpec
from iris.runtime import telemetry as runtime_telemetry

logger = logging.getLogger(__name__)

ROLLOUT_RUNS_NAMESPACE = "marin.rollout_runs"
_FLUSH_TIMEOUT = 10.0
_ROLLOUT_RUNS_TABLE_SPEC = TableSpec(version=1)


class RolloutRunKind(StrEnum):
    """The workflow that produced a set of rollouts."""

    EVALUATION = "evaluation"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


@dataclass(frozen=True)
class RolloutRunRecord:
    """One terminal attempt in the append-only rollout catalog."""

    key_column: ClassVar[str] = "timestamp_ms"

    timestamp_ms: int
    run_id: str
    attempt_id: str
    run_kind: str
    producer: str
    status: str
    rollout_uri: str
    storage_format: str
    artifact_uri: str | None
    model: str | None
    job_id: str | None
    attributes: dict[str, str]


def record_rollout_run(record: RolloutRunRecord) -> None:
    """Write a rollout discovery row when running under Iris.

    Rollouts and their native artifacts remain authoritative when Finelog is
    unavailable, so catalog failures are reported without failing the run.
    """
    try:
        _record_rollout_run(record)
    except Exception:
        logger.warning(
            "could not write rollout catalog record for %s/%s",
            record.run_id,
            record.attempt_id or "<iris-attempt>",
            exc_info=True,
        )


def _record_rollout_run(record: RolloutRunRecord) -> None:
    runtime = runtime_telemetry.resolve(run_id=record.run_id)
    if runtime is None:
        logger.debug("no in-cluster Iris context; skipping rollout catalog record for %s", record.run_id)
        return

    resolved = replace(
        record,
        attempt_id=record.attempt_id or runtime.attributes["execution_uid"],
        job_id=record.job_id or runtime.attributes.get("job_id"),
    )
    client = LogClient.connect(runtime.endpoint, resolver=runtime.resolver)
    try:
        # Object-native tables replay their complete live spool when a regional
        # forwarder first discovers them. Legacy-local tables deliberately seed
        # a new forwarding cursor at the tip, which would omit this table's
        # first catalog row from the hub.
        table = client.get_table(ROLLOUT_RUNS_NAMESPACE, RolloutRunRecord, table_spec=_ROLLOUT_RUNS_TABLE_SPEC)
        table.write((resolved,))
        result = table.flush(timeout=_FLUSH_TIMEOUT)
    finally:
        client.close()
    if result is not FlushResult.SUCCEEDED:
        logger.warning(
            "rollout catalog record for %s/%s was not confirmed: %s",
            resolved.run_id,
            resolved.attempt_id,
            result.value,
        )


def rollout_run_record(
    *,
    run_id: str,
    attempt_id: str = "",
    run_kind: RolloutRunKind,
    producer: str,
    status: str,
    rollout_uri: str,
    storage_format: str,
    artifact_uri: str | None = None,
    model: str | None = None,
    job_id: str | None = None,
    attributes: dict[str, str] | None = None,
) -> RolloutRunRecord:
    """Build a timestamped rollout-catalog row."""
    return RolloutRunRecord(
        timestamp_ms=time.time_ns() // 1_000_000,
        run_id=run_id,
        attempt_id=attempt_id,
        run_kind=run_kind.value,
        producer=producer,
        status=status,
        rollout_uri=rollout_uri,
        storage_format=storage_format,
        artifact_uri=artifact_uri,
        model=model,
        job_id=job_id,
        attributes=dict(attributes or {}),
    )
