# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run recoverable TaskTrove MCQA routing as a versioned artifact."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, lower, run
from marin.execution.remote import remote
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import launch_provenance

from experiments.post_training.tasktrove.mcqa_routing import (
    DECISIONS_FILENAME,
    GLM_BULK_TOKEN_ENV,
    MCQA_SOURCE,
    POLICY_VERSION,
    ROUTE_MAPPINGS_FILENAME,
    WORKER_SUMMARY_FILENAME,
    Route,
    RoutingConfig,
    run_worker,
)

logger = logging.getLogger(__name__)

DEFAULT_INPUT = "s3://marin-us-east-02a/marin/tasktrove/routing/mcqa-glm53-v1/input/mechanical-ledger.parquet"
DEFAULT_RELAY_JOB = "/muchanem/glm53-relay-08a"
ARTIFACT_VERSION = "2026.09.18.1"
DEFAULT_WORKERS = 64
DEFAULT_REQUEST_BATCH_SIZE = 20
DEFAULT_SAMPLE_SIZE = 0
DEFAULT_SAMPLE_SEED = "tasktrove-mcqa-glm53-full-v1"
DEFAULT_WORKER_CPU = 2
DEFAULT_WORKER_RAM = "4g"
DEFAULT_WORKER_DISK = "8g"
_GIT_REVISION_RUNTIME_ARG = "git_revision"


@dataclass(frozen=True)
class RoutingArtifactConfig:
    input_path: str
    output_path: str
    source: str
    git_revision: str
    sample_size: int
    sample_seed: str
    request_batch_size: int
    relay_job: str
    poll_seconds: float
    worker_count: int
    worker_cpu: float
    worker_ram: str
    worker_disk: str

    def worker_config(self) -> RoutingConfig:
        return RoutingConfig(
            input_path=self.input_path,
            output_path=self.output_path,
            source=self.source,
            git_revision=self.git_revision,
            sample_size=self.sample_size,
            sample_seed=self.sample_seed,
            request_batch_size=self.request_batch_size,
            relay_job=self.relay_job,
            poll_seconds=self.poll_seconds,
            expected_workers=self.worker_count,
        )


def _read_jsonl(path: StoragePath) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: StoragePath, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))


def aggregate_worker_outputs(config: RoutingArtifactConfig) -> dict:
    """Validate and combine worker outputs into the artifact root."""
    output_root = StoragePath(config.output_path)
    worker_summaries = []
    decisions = []
    mappings = []
    for worker_index in range(config.worker_count):
        worker_root = output_root / f"worker-{worker_index:03d}"
        worker_summaries.append(json.loads((worker_root / WORKER_SUMMARY_FILENAME).read_text()))
        decisions.extend(_read_jsonl(worker_root / DECISIONS_FILENAME))
        mappings.extend(_read_jsonl(worker_root / ROUTE_MAPPINGS_FILENAME))

    task_ids = [row["task_id"] for row in decisions]
    mapping_ids = [row["task_id"] for row in mappings]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("worker decisions contain duplicate task IDs")
    if set(task_ids) != set(mapping_ids) or len(mapping_ids) != len(set(mapping_ids)):
        raise ValueError("worker route mappings do not match the unique decision task IDs")
    if config.sample_size > 0 and len(task_ids) != config.sample_size:
        raise ValueError(f"workers routed {len(task_ids)} tasks, expected {config.sample_size}")

    decisions.sort(key=lambda row: row["task_id"])
    mappings.sort(key=lambda row: row["task_id"])
    _write_jsonl(output_root / DECISIONS_FILENAME, decisions)
    _write_jsonl(output_root / ROUTE_MAPPINGS_FILENAME, mappings)

    summary = {
        "input": config.input_path,
        "output": config.output_path,
        "source": config.source,
        "git_revision": config.git_revision,
        "policy_version": POLICY_VERSION,
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed,
        "worker_count": config.worker_count,
        "request_batch_size": config.request_batch_size,
        "routed_rows": len(decisions),
        "requests": sum(worker["requests"] for worker in worker_summaries),
        "degraded_requests": sum(worker["degraded_requests"] for worker in worker_summaries),
        "fallback_rows": sum(worker["fallback_rows"] for worker in worker_summaries),
        "route_counts": dict(Counter(row["route"] for row in decisions)),
        "subject_counts": dict(Counter(row["subject"] for row in decisions)),
        "route_subject_counts": {
            route: dict(Counter(row["subject"] for row in decisions if row["route"] == route)) for route in Route
        },
        "workers": worker_summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _write_or_check_run_config(config: RoutingArtifactConfig) -> None:
    StoragePath(config.output_path).mkdirs()
    path = StoragePath(config.output_path) / "run-config.json"
    expected = json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != expected:
            raise ValueError(f"{path} does not match this launch; use a new output path")
        return
    path.write_text(expected)


def run_routing_artifact(config: RoutingArtifactConfig) -> None:
    """Run the worker gang and assemble its complete route ledger."""
    _write_or_check_run_config(config)
    token = os.environ[GLM_BULK_TOKEN_ENV]
    run_hash = hashlib.sha256(config.output_path.encode()).hexdigest()[:8]
    worker_resources = ResourceConfig.with_cpu(
        cpu=config.worker_cpu,
        ram=config.worker_ram,
        disk=config.worker_disk,
        replicas=config.worker_count,
    )
    route_workers = remote(
        run_worker,
        name=f"tasktrove-mcqa-routing-{run_hash}",
        resources=worker_resources,
        env_vars={GLM_BULK_TOKEN_ENV: token},
    )
    route_workers(config.worker_config())
    summary = aggregate_worker_outputs(config)
    logger.info("Routed %d tasks: %s", summary["routed_rows"], summary["route_counts"])


def routing_step(
    *,
    input_path: str,
    git_revision: str,
    sample_size: int,
    sample_seed: str,
    worker_count: int = DEFAULT_WORKERS,
    request_batch_size: int = DEFAULT_REQUEST_BATCH_SIZE,
    source: str = MCQA_SOURCE,
    relay_job: str = DEFAULT_RELAY_JOB,
    poll_seconds: float = 10,
) -> ArtifactStep[Artifact]:
    """Build a recoverable MCQA route-mapping artifact."""
    if worker_count < 1:
        raise ValueError(f"worker_count must be positive, got {worker_count}")
    if sample_size < 0:
        raise ValueError(f"sample_size must be non-negative, got {sample_size}")
    if request_batch_size < 1:
        raise ValueError(f"request_batch_size must be positive, got {request_batch_size}")

    def build_config(ctx: StepContext) -> RoutingArtifactConfig:
        return RoutingArtifactConfig(
            input_path=input_path,
            output_path=ctx.output_path,
            source=source,
            git_revision=ctx.runtime_arg(_GIT_REVISION_RUNTIME_ARG),
            sample_size=sample_size,
            sample_seed=sample_seed,
            request_batch_size=request_batch_size,
            relay_job=relay_job,
            poll_seconds=poll_seconds,
            worker_count=worker_count,
            worker_cpu=DEFAULT_WORKER_CPU,
            worker_ram=DEFAULT_WORKER_RAM,
            worker_disk=DEFAULT_WORKER_DISK,
        )

    return ArtifactStep(
        name="tasktrove/mcqa-routing",
        version=ARTIFACT_VERSION,
        artifact_type=Artifact,
        run=run_routing_artifact,
        build_config=build_config,
        runtime_args={_GIT_REVISION_RUNTIME_ARG: git_revision},
    )


def _launch_commit() -> str:
    provenance = launch_provenance()
    if provenance.dirty:
        raise click.ClickException("MCQA routing must launch from a clean working tree")
    if not provenance.base_commit:
        raise click.ClickException("MCQA routing requires a Git commit in the launch provenance")
    return provenance.base_commit


@click.group(help=__doc__)
def main() -> None:
    """Generate the MCQA routing artifact."""
    pass


@main.command("generate")
@click.option("--input-path", default=DEFAULT_INPUT, show_default=True)
@click.option("--worker-count", type=click.IntRange(min=1), default=DEFAULT_WORKERS, show_default=True)
@click.option(
    "--request-batch-size",
    type=click.IntRange(min=1),
    default=DEFAULT_REQUEST_BATCH_SIZE,
    show_default=True,
)
@click.option("--run", "do_run", is_flag=True, help="Build the artifact; the default prints its plan")
def generate(
    input_path: str,
    worker_count: int,
    request_batch_size: int,
    do_run: bool,
) -> None:
    step = routing_step(
        input_path=input_path,
        git_revision=_launch_commit(),
        sample_size=DEFAULT_SAMPLE_SIZE,
        sample_seed=DEFAULT_SAMPLE_SEED,
        worker_count=worker_count,
        request_batch_size=request_batch_size,
    )
    click.echo(f"Output: {step.path()}")
    if do_run:
        run(step)
    else:
        click.echo(lower(step))


if __name__ == "__main__":
    main()
