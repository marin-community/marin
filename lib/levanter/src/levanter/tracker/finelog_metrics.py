# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed Levanter metric events written directly to Finelog."""

import dataclasses
import time
from typing import ClassVar

import numpy as np
from finelog.client import FlushResult, LogClient
from finelog.client.log_client import Table
from iris.runtime import telemetry as runtime_telemetry

from levanter.tracker.histogram import SummaryStats

LEVANTER_METRICS_NAMESPACE = "levanter.metrics"


@dataclasses.dataclass(frozen=True)
class LevanterMetricRow:
    """One scalar, summary, or histogram point for a training run."""

    key_column: ClassVar[str] = "timestamp_ms"

    timestamp_ms: int
    run_id: str
    execution_uid: str | None
    job_id: str | None
    node_name: str | None
    process_index: int | None
    step: int | None
    name: str
    kind: str
    value: float | None
    min: float | None
    max: float | None
    count: int | None
    nonzero_count: int | None
    sum: float | None
    sum_squares: float | None
    mean: float | None
    variance: float | None
    rms: float | None
    bucket_limits: list[float] | None
    bucket_counts: list[int] | None
    unit: str | None
    attributes: dict[str, str]
    batch_id: str | None
    record_index: int | None


@dataclasses.dataclass(frozen=True)
class _MetricIdentity:
    run_id: str
    execution_uid: str | None
    job_id: str | None
    node_name: str | None
    process_index: int | None


class LevanterMetricsWriter:
    """Nonblocking typed writer for one Levanter run."""

    def __init__(self, client: LogClient, table: Table, identity: _MetricIdentity):
        self._client = client
        self._table = table
        self._identity = identity

    @classmethod
    def from_iris(cls, run_id: str | None, process_index: int) -> "LevanterMetricsWriter | None":
        """Create a writer from the Iris runtime, or return None outside Iris."""
        runtime = runtime_telemetry.resolve(run_id=run_id, process_index=process_index)
        if runtime is None:
            return None
        identity = _MetricIdentity(
            run_id=runtime.attributes["run_id"],
            execution_uid=runtime.attributes.get("execution_uid"),
            job_id=runtime.attributes.get("job_id"),
            node_name=runtime.attributes.get("node_name"),
            process_index=int(runtime.attributes["process_index"]) if "process_index" in runtime.attributes else None,
        )
        client = LogClient.connect(runtime.endpoint, resolver=runtime.resolver)
        namespace = f"{LEVANTER_METRICS_NAMESPACE}.{identity.run_id}"
        table = client.get_table(namespace, LevanterMetricRow)
        return cls(client, table, identity)

    def scalar(
        self,
        name: str,
        value: float,
        *,
        step: int | None,
        unit: str | None = None,
        attributes: dict[str, str] | None = None,
    ) -> None:
        self._write(
            step=step,
            name=name,
            kind="scalar",
            value=float(value),
            unit=unit,
            attributes=attributes,
        )

    def summary(
        self,
        name: str,
        stats: SummaryStats,
        *,
        step: int | None,
        unit: str | None = None,
        attributes: dict[str, str] | None = None,
    ) -> None:
        histogram = stats.histogram
        self._write(
            step=step,
            name=name,
            kind="histogram" if histogram is not None else "summary",
            min=_as_float(stats.min),
            max=_as_float(stats.max),
            count=_as_int(stats.num),
            nonzero_count=_as_int(stats.nonzero_count),
            sum=_as_float(stats.sum),
            sum_squares=_as_float(stats.sum_squares),
            mean=_as_float(stats.mean),
            variance=_as_float(stats.variance),
            rms=_as_float(stats.rms),
            bucket_limits=None if histogram is None else _as_float_list(histogram.bucket_limits),
            bucket_counts=None if histogram is None else _as_int_list(histogram.bucket_counts),
            unit=unit,
            attributes=attributes,
        )

    def flush(self, timeout: float | None = None) -> FlushResult:
        return self._table.flush(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def _write(
        self,
        *,
        step: int | None,
        name: str,
        kind: str,
        value: float | None = None,
        min: float | None = None,
        max: float | None = None,
        count: int | None = None,
        nonzero_count: int | None = None,
        sum: float | None = None,
        sum_squares: float | None = None,
        mean: float | None = None,
        variance: float | None = None,
        rms: float | None = None,
        bucket_limits: list[float] | None = None,
        bucket_counts: list[int] | None = None,
        unit: str | None,
        attributes: dict[str, str] | None,
    ) -> None:
        self._table.write(
            [
                LevanterMetricRow(
                    timestamp_ms=time.time_ns() // 1_000_000,
                    run_id=self._identity.run_id,
                    execution_uid=self._identity.execution_uid,
                    job_id=self._identity.job_id,
                    node_name=self._identity.node_name,
                    process_index=self._identity.process_index,
                    step=step,
                    name=name,
                    kind=kind,
                    value=value,
                    min=min,
                    max=max,
                    count=count,
                    nonzero_count=nonzero_count,
                    sum=sum,
                    sum_squares=sum_squares,
                    mean=mean,
                    variance=variance,
                    rms=rms,
                    bucket_limits=bucket_limits,
                    bucket_counts=bucket_counts,
                    unit=unit,
                    attributes=dict(attributes or {}),
                    batch_id=None,
                    record_index=None,
                )
            ]
        )


def _as_float(value: object) -> float:
    return float(np.asarray(value))


def _as_int(value: object) -> int:
    return int(np.asarray(value))


def _as_float_list(value: object) -> list[float]:
    return np.asarray(value).astype(float).tolist()


def _as_int_list(value: object) -> list[int]:
    return np.asarray(value).astype(np.int64).tolist()
