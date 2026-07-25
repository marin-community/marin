# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable GPU telemetry capture for training jobs.

The sampler runs outside the trainer process so JAX compilation, checkpointing,
or a busy Python runtime cannot delay telemetry collection. The parent process
starts the sampler with :func:`nvidia_smi_telemetry`; the child process runs
``nvidia-smi``, parses rows from stdout, and writes JSONL chunks to a remote
``fsspec`` URI through :class:`marin.monitoring.jsonl_writer.JsonlChunkWriter`.
If the pod disappears, already-flushed chunks remain durable and only queued or
in-flight records can be lost.
"""

import contextlib
import csv
import datetime as dt
import json
import logging
import multiprocessing
import select
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventType
from pathlib import Path

from marin.monitoring.jsonl_writer import BackpressurePolicy, JsonlChunkWriter, JsonlChunkWriterConfig

logger = logging.getLogger(__name__)

DEFAULT_NVIDIA_SMI_FIELDS = (
    "timestamp",
    "index",
    "name",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "pstate",
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
)


@dataclass(frozen=True)
class NvidiaSmiTelemetryConfig:
    """Configuration for a durable ``nvidia-smi`` telemetry process.

    Args:
        output_uri: Destination directory for JSONL chunks and manifest.
        interval: Seconds between ``nvidia-smi`` samples.
        records_per_chunk: Number of samples per durable JSONL chunk file.
        max_queue_items: Maximum JSONL records queued in the writer actor before
            backpressure policy is applied.
        backpressure_policy: Whether telemetry writes block or drop when the
            writer queue is full.
        log_every: Log writer progress after this many accepted/dropped records.
        query_fields: ``nvidia-smi --query-gpu`` fields.
        command: Optional full command. Tests and non-NVIDIA probes can pass a
            command that emits ``nvidia-smi --format=csv``-style output.
        start_method: Multiprocessing start method used for the telemetry
            process.
        stop_timeout: Seconds to wait for graceful process shutdown before
            terminating it.
        require_command: Fail before training starts if the command executable
            is not present.
    """

    output_uri: str
    interval: float = 5.0
    records_per_chunk: int = 120
    max_queue_items: int = 10_000
    backpressure_policy: BackpressurePolicy = BackpressurePolicy.BLOCK
    log_every: int = 1_000
    query_fields: tuple[str, ...] = DEFAULT_NVIDIA_SMI_FIELDS
    command: tuple[str, ...] = ()
    start_method: str = "spawn"
    stop_timeout: float = 30.0
    require_command: bool = True

    def __post_init__(self) -> None:
        if not self.output_uri:
            raise ValueError("output_uri must be non-empty")
        if self.interval <= 0:
            raise ValueError("interval must be positive")
        if self.records_per_chunk <= 0:
            raise ValueError("records_per_chunk must be positive")
        if self.max_queue_items <= 0:
            raise ValueError("max_queue_items must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if isinstance(self.backpressure_policy, str):
            object.__setattr__(self, "backpressure_policy", BackpressurePolicy(self.backpressure_policy))
        if not isinstance(self.backpressure_policy, BackpressurePolicy):
            raise ValueError("backpressure_policy must be a BackpressurePolicy")
        if self.stop_timeout <= 0:
            raise ValueError("stop_timeout must be positive")
        if self.command and not self.command[0]:
            raise ValueError("command executable must be non-empty")
        if not self.command and not self.query_fields:
            raise ValueError("query_fields must be non-empty when command is not set")


def build_nvidia_smi_command(config: NvidiaSmiTelemetryConfig) -> tuple[str, ...]:
    """Return the command used by the telemetry process."""
    if config.command:
        return config.command
    return (
        "nvidia-smi",
        f"--query-gpu={','.join(config.query_fields)}",
        "--format=csv",
        "-l",
        _format_seconds_for_nvidia_smi(config.interval),
    )


@dataclass(frozen=True)
class NvidiaSmiTelemetryHandle:
    """Handle for a running telemetry process."""

    process: multiprocessing.Process
    stop_event: EventType
    stop_timeout: float

    def stop(self) -> None:
        """Request shutdown, flush the final chunk, and reap the process."""
        self.stop_event.set()
        self.process.join(timeout=self.stop_timeout)
        if self.process.is_alive():
            logger.warning("GPU telemetry process did not stop in %.1fs; terminating", self.stop_timeout)
            self.process.terminate()
            self.process.join(timeout=5.0)
        if self.process.exitcode not in (0, None):
            logger.warning("GPU telemetry process exited with code %s", self.process.exitcode)


def start_nvidia_smi_telemetry(config: NvidiaSmiTelemetryConfig) -> NvidiaSmiTelemetryHandle:
    """Start a background process that writes GPU telemetry JSONL chunks."""
    command = build_nvidia_smi_command(config)
    if config.require_command and shutil.which(command[0]) is None:
        raise FileNotFoundError(f"GPU telemetry command not found: {command[0]}")
    ctx = multiprocessing.get_context(config.start_method)
    stop_event = ctx.Event()
    process = ctx.Process(
        target=run_nvidia_smi_telemetry,
        args=(config, stop_event),
        name="nvidia-smi-telemetry",
        daemon=False,
    )
    process.start()
    return NvidiaSmiTelemetryHandle(process=process, stop_event=stop_event, stop_timeout=config.stop_timeout)


@contextlib.contextmanager
def nvidia_smi_telemetry(config: NvidiaSmiTelemetryConfig) -> Iterator[NvidiaSmiTelemetryHandle]:
    """Run durable GPU telemetry while the body executes."""
    handle = start_nvidia_smi_telemetry(config)
    try:
        yield handle
    finally:
        handle.stop()


def run_nvidia_smi_telemetry(config: NvidiaSmiTelemetryConfig, stop_event: EventType) -> None:
    """Run ``nvidia-smi`` and write parsed samples as JSONL chunks."""
    command = build_nvidia_smi_command(config)
    writer_config = JsonlChunkWriterConfig(
        output_uri=config.output_uri,
        records_per_chunk=config.records_per_chunk,
        max_queue_items=config.max_queue_items,
        backpressure_policy=config.backpressure_policy,
        log_every=config.log_every,
    )
    process: subprocess.Popen[str] | None = None
    with JsonlChunkWriter(writer_config) as writer:
        writer.write(
            {
                "record_type": "metadata",
                "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                "command": list(command),
                "query_fields": list(config.query_fields),
                "interval": config.interval,
            }
        )
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            header: list[str] | None = None
            while not stop_event.is_set():
                ready, _, _ = select.select([process.stdout], [], [], 1.0)
                if not ready:
                    if process.poll() is not None:
                        break
                    continue
                line = process.stdout.readline()
                if line == "":
                    if process.poll() is not None:
                        break
                    continue
                row = next(csv.reader([line.rstrip("\n")]))
                if header is None:
                    header = [_normalize_field_name(field) for field in row]
                    continue
                values = {header[index]: value.strip() for index, value in enumerate(row) if index < len(header)}
                writer.write(
                    {
                        "record_type": "gpu_sample",
                        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "nvidia_smi": values,
                    }
                )
            if process.poll() is not None and process.returncode not in (0, None):
                stderr = process.stderr.read() if process.stderr is not None else ""
                writer.write(
                    {
                        "record_type": "error",
                        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "message": "telemetry command exited nonzero",
                        "returncode": process.returncode,
                        "stderr": stderr.strip(),
                    }
                )
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            writer.write(
                {
                    "record_type": "metadata",
                    "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "event": "telemetry_stop",
                }
            )


def _normalize_field_name(field: str) -> str:
    field = field.strip()
    if " [" in field:
        field = field.split(" [", 1)[0]
    return field.replace(".", "_").replace(" ", "_").replace("-", "_")


def _format_seconds_for_nvidia_smi(seconds: float) -> str:
    if seconds.is_integer():
        return str(int(seconds))
    return str(seconds)


if __name__ == "__main__":
    config_path = Path(sys.argv[1])
    event = multiprocessing.Event()
    run_nvidia_smi_telemetry(NvidiaSmiTelemetryConfig(**json.loads(config_path.read_text())), event)
