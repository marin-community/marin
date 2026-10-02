# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Comparable batch-generation measurements, independent of the inference runtime."""

import dataclasses
import hashlib
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from rigging.provenance import Provenance


@dataclass(frozen=True)
class TokenWorkload:
    """Exact tokenized prompts shared by both runtimes; EOS stopping is disabled."""

    prompts: list[list[int]]
    output_tokens: int

    def __post_init__(self):
        if not self.prompts or any(not prompt for prompt in self.prompts):
            raise ValueError("Every batch must contain nonempty prompts")
        if self.output_tokens < 2:
            raise ValueError("At least two output tokens are required to measure decode")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(dataclasses.asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class BatchMeasurement:
    """Host-observed batch latency, with one first-token timestamp per request."""

    elapsed: float
    first_token: list[float]
    tokens: list[list[int]]


@dataclass(frozen=True)
class BatchSummary:
    elapsed: float
    first_token: list[float]
    all_first_tokens: float
    output_tokens_per_second: float
    mean_time_after_first_token_per_output_token: float
    output_sha256: str


def summarize_batch(workload: TokenWorkload, measurement: BatchMeasurement) -> BatchSummary:
    """Reject partial outputs and summarize throughput without counting prompt tokens."""
    if len(measurement.tokens) != len(workload.prompts) or any(
        len(tokens) != workload.output_tokens for tokens in measurement.tokens
    ):
        raise ValueError("Incomplete generation: each prompt must produce exactly output_tokens tokens")
    if len(measurement.first_token) != len(workload.prompts) or any(
        not 0 < first <= measurement.elapsed for first in measurement.first_token
    ):
        raise ValueError("Missing or invalid first-token measurements")
    # Batch decode starts only after every request has a first token. With chunked admission,
    # decode can overlap later prefills, so only end-to-end throughput is directly comparable.
    all_first = max(measurement.first_token)
    return BatchSummary(
        elapsed=measurement.elapsed,
        first_token=measurement.first_token,
        all_first_tokens=all_first,
        output_tokens_per_second=len(workload.prompts) * workload.output_tokens / measurement.elapsed,
        mean_time_after_first_token_per_output_token=statistics.mean(
            (measurement.elapsed - first) / (workload.output_tokens - 1) for first in measurement.first_token
        ),
        output_sha256=hashlib.sha256(json.dumps(measurement.tokens).encode()).hexdigest(),
    )


@dataclass(frozen=True)
class BenchmarkReport:
    schema_version: int
    timing_boundary: str
    workload: TokenWorkload
    workload_sha256: str
    compile_only: float | None
    first_batch_including_compile: BatchSummary
    warmup: list[BatchSummary]
    samples: list[BatchSummary]
    median_output_tokens_per_second: float
    validation_tokens: list[list[int]]
    validation_output_sha256: str


def measure_batches(
    workload: TokenWorkload,
    generate: Callable[[TokenWorkload], BatchMeasurement],
    *,
    warmup_batches: int,
    measured_batches: int,
) -> BenchmarkReport:
    """Keep the first batch and warmup outside steady-state throughput."""
    if warmup_batches < 1 or measured_batches < 1:
        raise ValueError("Use at least one warmup and one measured batch")
    cold = summarize_batch(workload, generate(workload))
    warmup = [summarize_batch(workload, generate(workload)) for _ in range(warmup_batches)]
    samples = [summarize_batch(workload, generate(workload)) for _ in range(measured_batches)]
    validation = generate(workload)
    validation_summary = summarize_batch(workload, validation)
    return BenchmarkReport(
        schema_version=2,
        timing_boundary="offline_batch_host_submission_to_host_tokens",
        workload=workload,
        workload_sha256=workload.sha256,
        compile_only=None,
        first_batch_including_compile=cold,
        warmup=warmup,
        samples=samples,
        median_output_tokens_per_second=statistics.median(row.output_tokens_per_second for row in samples),
        validation_tokens=validation.tokens,
        validation_output_sha256=validation_summary.output_sha256,
    )


@dataclass(frozen=True)
class SourceProvenance:
    revision: str | None
    dirty: bool | None
    origin: str
    tree_hash: str | None = None


def source_provenance(revision: str | None = None, dirty: bool | None = None) -> SourceProvenance:
    """Use launcher provenance for source bundles, or inspect the current Git checkout."""
    if revision is not None:
        return SourceProvenance(revision, dirty, "launcher")
    if dirty is not None:
        raise ValueError("An explicit dirty state requires a source revision")
    captured = Provenance.capture()
    if not captured.base_commit:
        return SourceProvenance(None, None, "unavailable")
    return SourceProvenance(captured.base_commit, captured.dirty, "capture", captured.tree_hash or None)
