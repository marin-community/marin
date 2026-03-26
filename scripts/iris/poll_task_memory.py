#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Poll Iris task resource usage and append JSONL samples.

Examples:
  uv run python scripts/iris/poll_task_memory.py \
      --config lib/iris/examples/marin.yaml \
      --task-id /ahmed/job/train/0

  uv run python scripts/iris/poll_task_memory.py \
      --config lib/iris/examples/marin.yaml \
      --task-id /ahmed/job/train/0 \
      --interval 5 \
      --output scratch/train-memory.jsonl
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import click

from iris.cli.main import create_client_token_provider
from iris.client import IrisClient
from iris.cluster.config import IrisConfig
from iris.cluster.types import JobName
from iris.rpc import cluster_pb2

DEFAULT_INTERVAL = 5.0
DEFAULT_TIMEOUT_MS = 30_000
DEFAULT_OUTPUT_DIR = Path("scratch")
TERMINAL_TASK_STATES = {
    cluster_pb2.TASK_STATE_SUCCEEDED,
    cluster_pb2.TASK_STATE_FAILED,
    cluster_pb2.TASK_STATE_KILLED,
    cluster_pb2.TASK_STATE_WORKER_FAILED,
}


@dataclass(frozen=True)
class TaskMemorySample:
    timestamp: str
    unix_time: float
    task_id: str
    task_state: str
    task_exit_code: int
    task_error: str
    current_attempt_id: int
    current_attempt_state: str
    current_attempt_exit_code: int
    current_attempt_error: str
    worker_id: str
    worker_address: str
    memory_mb: int
    memory_peak_mb: int
    disk_mb: int
    cpu_percent: int
    cpu_millicores: int
    process_count: int
    last_failed_attempt_id: int | None
    last_failed_attempt_state: str | None
    last_failed_attempt_exit_code: int | None
    last_failed_attempt_error: str | None


def _default_output_path(task_name: JobName) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    parent_job, _ = task_name.require_task()
    leaf = parent_job.name
    return DEFAULT_OUTPUT_DIR / f"{timestamp}_{leaf}_memory.jsonl"


def _timestamp_now() -> tuple[str, float]:
    now = datetime.now(tz=UTC)
    return now.isoformat(), now.timestamp()


def _attempt_by_id(status: cluster_pb2.TaskStatus, attempt_id: int) -> cluster_pb2.TaskAttempt | None:
    for attempt in status.attempts:
        if attempt.attempt_id == attempt_id:
            return attempt
    return None


def _last_failed_attempt(status: cluster_pb2.TaskStatus) -> cluster_pb2.TaskAttempt | None:
    failed_attempts = [
        attempt
        for attempt in status.attempts
        if attempt.state
        in {
            cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.TASK_STATE_KILLED,
            cluster_pb2.TASK_STATE_WORKER_FAILED,
        }
    ]
    if not failed_attempts:
        return None
    return max(failed_attempts, key=lambda attempt: attempt.attempt_id)


def _sample_from_status(status: cluster_pb2.TaskStatus) -> TaskMemorySample:
    usage = status.resource_usage
    timestamp, unix_time = _timestamp_now()
    current_attempt = _attempt_by_id(status, status.current_attempt_id)
    last_failed_attempt = _last_failed_attempt(status)
    return TaskMemorySample(
        timestamp=timestamp,
        unix_time=unix_time,
        task_id=status.task_id,
        task_state=cluster_pb2.TaskState.Name(status.state),
        task_exit_code=status.exit_code,
        task_error=status.error,
        current_attempt_id=status.current_attempt_id,
        current_attempt_state=(
            cluster_pb2.TaskState.Name(current_attempt.state) if current_attempt else "TASK_STATE_UNKNOWN"
        ),
        current_attempt_exit_code=current_attempt.exit_code if current_attempt else 0,
        current_attempt_error=current_attempt.error if current_attempt else "",
        worker_id=status.worker_id,
        worker_address=status.worker_address,
        memory_mb=usage.memory_mb,
        memory_peak_mb=usage.memory_peak_mb,
        disk_mb=usage.disk_mb,
        cpu_percent=usage.cpu_percent,
        cpu_millicores=usage.cpu_millicores,
        process_count=usage.process_count,
        last_failed_attempt_id=last_failed_attempt.attempt_id if last_failed_attempt else None,
        last_failed_attempt_state=cluster_pb2.TaskState.Name(last_failed_attempt.state) if last_failed_attempt else None,
        last_failed_attempt_exit_code=last_failed_attempt.exit_code if last_failed_attempt else None,
        last_failed_attempt_error=last_failed_attempt.error if last_failed_attempt else None,
    )


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--task-id", required=True, type=str, help="Canonical Iris task id, e.g. /user/job/.../0")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--interval", type=float, default=DEFAULT_INTERVAL, show_default=True)
@click.option(
    "--samples",
    type=int,
    default=0,
    show_default=True,
    help="Stop after N samples; 0 means run until the task is terminal or interrupted.",
)
@click.option("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, show_default=True)
@click.option("--flush-every", type=int, default=1, show_default=True)
def main(
    config_path: Path,
    task_id: str,
    output: Path | None,
    interval: float,
    samples: int,
    timeout_ms: int,
    flush_every: int,
) -> None:
    """Poll task resource usage and append JSONL records."""
    if interval <= 0:
        raise click.ClickException("--interval must be > 0")
    if flush_every <= 0:
        raise click.ClickException("--flush-every must be > 0")

    task_name = JobName.from_wire(task_id)
    task_name.require_task()

    iris_config = IrisConfig.load(config_path)
    token_provider = None
    if iris_config.proto.HasField("auth"):
        cluster_name = iris_config.proto.name or "default"
        token_provider = create_client_token_provider(iris_config.proto.auth, cluster_name=cluster_name)

    platform = iris_config.platform()
    controller_address = iris_config.controller_address() or platform.discover_controller(iris_config.proto.controller)
    output_path = output or _default_output_path(task_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Polling {task_id}")
    click.echo(f"Writing JSONL to {output_path}")

    written = 0
    with platform.tunnel(address=controller_address) as controller_url:
        with IrisClient.remote(controller_url, timeout_ms=timeout_ms, token_provider=token_provider) as client:
            with output_path.open("a", encoding="utf-8") as f:
                while True:
                    status = client.task_status(task_name)
                    sample = _sample_from_status(status)
                    f.write(json.dumps(asdict(sample), sort_keys=True) + "\n")
                    written += 1
                    if written % flush_every == 0:
                        f.flush()
                    last_failed_suffix = ""
                    if (
                        sample.last_failed_attempt_id is not None
                        and sample.last_failed_attempt_id != sample.current_attempt_id
                    ):
                        last_failed_suffix = (
                            f" last_failed_attempt={sample.last_failed_attempt_id}"
                            f" last_failed_exit={sample.last_failed_attempt_exit_code}"
                        )
                    click.echo(
                        f"{sample.timestamp} attempt={sample.current_attempt_id}"
                        f" state={sample.current_attempt_state}"
                        f" mem_mb={sample.memory_mb}"
                        f" peak_mb={sample.memory_peak_mb}"
                        f" cpu_pct={sample.cpu_percent}"
                        f"{last_failed_suffix}"
                    )

                    if status.state in TERMINAL_TASK_STATES:
                        f.flush()
                        click.echo(f"Task reached terminal state: {sample.task_state}")
                        return
                    if samples and written >= samples:
                        f.flush()
                        click.echo(f"Collected {written} samples")
                        return
                    time.sleep(interval)


if __name__ == "__main__":
    main()
