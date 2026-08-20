#!/usr/bin/env python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure Zephyr scatter-write and sorted-merge memory on one Iris task.

The driver emits one ``MEMORY_RESULT: {...}`` JSON row per fresh child
process. Input preparation is a separate operation so read measurements do
not include Parquet generation in their task cgroup peak.

Examples:
    uv run python lib/zephyr/tests/benchmark_shuffle_memory.py \
        --operation write --target-estimated-bytes 64MiB --output-root /tmp/write

    uv run python lib/zephyr/tests/benchmark_shuffle_memory.py \
        --operation prepare-read --input-count 8 --rows-per-input 10000 \
        --item-bytes 1024 --output-root /tmp/read-input

    uv run python lib/zephyr/tests/benchmark_shuffle_memory.py \
        --operation read-collect --input-root /tmp/read-input \
        --output-root /tmp/read-output
"""

import json
import os
import random
import resource
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import humanfriendly
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import StoragePath
from zephyr import memory_budget
from zephyr.shuffle import (
    ScatterWriter,
    _DATAFRAME_ROW_COUNT,
    _PAYLOAD_COL,
    _SORT_KEY_COL,
    _merge_sorted_frames,
    _scan_scatter_parquet,
    _task_memory_bytes,
    _unify_frame_schemas,
)

_RESULT_PREFIX = "MEMORY_RESULT: "
_RSS_SAMPLE_INTERVAL = 0.01
_FRAME_TARGET_BYTES = 64 * 1024 * 1024
_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ReadManifest:
    input_paths: tuple[str, ...]
    row_counts: tuple[int, ...]
    item_bytes: tuple[int, ...]
    estimated_bytes: tuple[int, ...]
    total_rows: int
    avg_item_bytes: float

    def to_json(self) -> dict[str, Any]:
        return {
            "input_paths": list(self.input_paths),
            "row_counts": list(self.row_counts),
            "item_bytes": list(self.item_bytes),
            "estimated_bytes": list(self.estimated_bytes),
            "total_rows": self.total_rows,
            "avg_item_bytes": self.avg_item_bytes,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ReadManifest":
        return cls(
            input_paths=tuple(value["input_paths"]),
            row_counts=tuple(value["row_counts"]),
            item_bytes=tuple(value["item_bytes"]),
            estimated_bytes=tuple(value["estimated_bytes"]),
            total_rows=int(value["total_rows"]),
            avg_item_bytes=float(value["avg_item_bytes"]),
        )


class RssSampler:
    """Sample this process's RSS until stopped."""

    def __init__(self, interval: float = _RSS_SAMPLE_INTERVAL):
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, name="rss-sampler", daemon=True)
        self.peak_bytes = _rss_bytes()

    def _sample(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.peak_bytes = max(self.peak_bytes, _rss_bytes())
            except OSError:
                return

    def __enter__(self) -> "RssSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self.peak_bytes = max(self.peak_bytes, _rss_bytes())
        except OSError:
            pass


def _parse_bytes(value: str) -> int:
    return int(humanfriendly.parse_size(value, binary=True))


def _validate_calibration_path(path: str, role: str) -> None:
    iris_environment = any(
        os.environ.get(name) for name in ("IRIS_JOB_ID", "IRIS_TASK_ID", "IRIS_CONTROLLER_URL")
    )
    if not iris_environment:
        return
    if "/tmp/ttl=1d/zephyr-shuffle-calibration/" not in path:
        raise ValueError(f"Iris calibration {role} must use the one-day TTL calibration prefix, got {path}")


def _binary_array(data: bytes, width: int, rows: int) -> pa.Array:
    offsets = np.arange(0, (rows + 1) * width, width, dtype=np.int32)
    return pa.Array.from_buffers(pa.binary(), rows, [None, pa.py_buffer(offsets), pa.py_buffer(data)])


def _frame(
    rows: int,
    item_bytes: int,
    seed: int,
    *,
    row_offset: int,
    num_targets: int = 1,
    hot_shard_fraction: float = 0.0,
) -> pl.DataFrame:
    rng = random.Random(seed)
    payload = rng.randbytes(rows * item_bytes)
    payload_array = _binary_array(payload, item_bytes, rows)

    row_ids = np.arange(row_offset, row_offset + rows, dtype=np.uint64)
    key_bytes = row_ids.astype(">u8", copy=False).tobytes()
    key_array = _binary_array(key_bytes, 8, rows)
    sort_key = pa.StructArray.from_arrays(
        [key_array, pa.nulls(rows)],
        names=["key", "sort_value"],
    )

    columns: dict[str, pa.Array] = {
        _PAYLOAD_COL: payload_array,
        _SORT_KEY_COL: sort_key,
    }
    if num_targets > 0:
        shards = (row_ids % num_targets).astype(np.int32)
        if hot_shard_fraction > 0:
            hot_rows = int(rows * hot_shard_fraction)
            shards[:hot_rows] = 0
        columns["__zephyr_shard__"] = pa.array(shards, type=pa.int32())
    return pl.from_arrow(pa.table(columns), rechunk=False)


def _warm_polars() -> None:
    pl.DataFrame({"x": [2, 1]}).sort("x")


def _rss_bytes() -> int:
    with open("/proc/self/statm") as statm:
        resident_pages = int(statm.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _process_peak_bytes() -> int:
    scale = 1024 if sys.platform == "linux" else 1
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


def _cgroup_memory_bytes(name: str) -> int | None:
    candidates = (
        Path(f"/sys/fs/cgroup/{name}"),
        Path(f"/sys/fs/cgroup/memory/{name.replace('.', '_in_bytes')}"),
    )
    for path in candidates:
        try:
            value = path.read_text().strip()
        except OSError:
            continue
        if value and value != "max":
            return int(value)
    return None


def _common_result(
    *,
    operation: str,
    baseline_rss_bytes: int,
    peak_rss_bytes: int,
    bytes_at_risk: float,
    elapsed: float,
    repetition: int,
) -> dict[str, Any]:
    growth = max(0, peak_rss_bytes - baseline_rss_bytes)
    return {
        "operation": operation,
        "repetition": repetition,
        "included_in_fit": repetition >= 0,
        "host": os.uname().nodename,
        "iris_job_id": os.environ.get("IRIS_JOB_ID"),
        "iris_task_id": os.environ.get("IRIS_TASK_ID"),
        "calibration_tree_id": os.environ.get("CALIBRATION_TREE_ID"),
        "git_commit": os.environ.get("MARIN_GIT_COMMIT") or os.environ.get("GIT_COMMIT"),
        "polars_version": pl.__version__,
        "polars_threads": pl.thread_pool_size(),
        "baseline_rss_bytes": baseline_rss_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "process_high_water_rss_bytes": _process_peak_bytes(),
        "cgroup_current_bytes": _cgroup_memory_bytes("memory.current"),
        "cgroup_peak_bytes": _cgroup_memory_bytes("memory.peak"),
        "bytes_at_risk": bytes_at_risk,
        "rss_growth_bytes": growth,
        "observed_r": growth / bytes_at_risk if bytes_at_risk > 0 else None,
        "elapsed_seconds": elapsed,
    }


def _write_child(
    *,
    target_estimated_bytes: int,
    exact_rows: int | None,
    item_bytes: int,
    num_targets: int,
    hot_shard_fraction: float,
    output_root: str,
    seed: int,
    repetition: int,
    auto_flush: bool,
) -> dict[str, Any]:
    _warm_polars()
    writer = ScatterWriter(
        data_path=f"{output_root.rstrip('/')}/write-rep-{repetition}",
        key_fn=lambda item: item,
        source_shard=0,
    )
    baseline = max(_rss_bytes(), _process_peak_bytes())
    estimated_bytes_per_row = float(max(item_bytes + 24, 1))
    rows = 0
    estimated_bytes = 0
    max_buffer_estimated_bytes = 0
    frames: list[pl.DataFrame] = []
    started = time.monotonic()
    with RssSampler() as sampler:
        while (rows < exact_rows) if exact_rows is not None else (estimated_bytes < target_estimated_bytes):
            if exact_rows is None:
                remaining_bytes = target_estimated_bytes - estimated_bytes
            else:
                remaining_bytes = int((exact_rows - rows) * estimated_bytes_per_row)
            frame_target_bytes = min(_FRAME_TARGET_BYTES, max(1, remaining_bytes))
            rows_per_frame = min(_DATAFRAME_ROW_COUNT, max(1, int(frame_target_bytes / estimated_bytes_per_row)))
            if exact_rows is not None:
                rows_per_frame = min(rows_per_frame, exact_rows - rows)
            frame = _frame(
                rows_per_frame,
                item_bytes,
                seed + len(frames),
                row_offset=rows,
                num_targets=num_targets,
                hot_shard_fraction=hot_shard_fraction,
            )
            frames.append(frame)
            rows += len(frame)
            frame_estimated_bytes = int(frame.estimated_size())
            estimated_bytes += frame_estimated_bytes
            estimated_bytes_per_row = frame.estimated_size() / len(frame)

            if auto_flush:
                buffered_after_write = writer._buffer_estimated_bytes + frame_estimated_bytes
                writer.write(frame)
                max_buffer_estimated_bytes = max(max_buffer_estimated_bytes, buffered_after_write)
                frames.clear()

        if not auto_flush:
            writer._frames = frames
            writer._buffer_estimated_bytes = estimated_bytes
            max_buffer_estimated_bytes = estimated_bytes
            writer._flush()
        writer.close()
    elapsed = time.monotonic() - started

    if writer._total_rows_written != rows:
        raise RuntimeError(f"write row mismatch: expected {rows}, wrote {writer._total_rows_written}")

    peak = max(sampler.peak_bytes, _process_peak_bytes())
    return {
        **_common_result(
            operation="write",
            baseline_rss_bytes=baseline,
            peak_rss_bytes=peak,
            bytes_at_risk=estimated_bytes,
            elapsed=elapsed,
            repetition=repetition,
        ),
        "target_estimated_bytes": target_estimated_bytes,
        "target_rows": exact_rows,
        "estimated_bytes": estimated_bytes,
        "max_buffer_estimated_bytes": max_buffer_estimated_bytes,
        "auto_flush": auto_flush,
        "flush_count": len(writer._chunk_paths),
        "rows": rows,
        "item_bytes": item_bytes,
        "num_targets": num_targets,
        "hot_shard_fraction": hot_shard_fraction,
        "output_files": list(writer._chunk_paths),
    }


def _read_shape(
    input_index: int,
    input_count: int,
    rows_per_input: int,
    item_bytes: int,
    secondary_item_bytes: int | None,
    secondary_fraction: float,
    secondary_row_fraction: float,
) -> tuple[int, int]:
    if input_index >= int(input_count * (1 - secondary_fraction)) and secondary_item_bytes is not None:
        item_bytes = secondary_item_bytes
    if input_index >= int(input_count * (1 - secondary_row_fraction)):
        rows_per_input = max(1, rows_per_input // 10)
    return rows_per_input, item_bytes


def _prepare_read_inputs(
    *,
    input_count: int,
    rows_per_input: int,
    item_bytes: int,
    secondary_item_bytes: int | None,
    secondary_fraction: float,
    secondary_row_fraction: float,
    output_root: str,
    seed: int,
) -> ReadManifest:
    input_paths: list[str] = []
    row_counts: list[int] = []
    item_widths: list[int] = []
    estimated_sizes: list[int] = []

    for input_index in range(input_count):
        rows, width = _read_shape(
            input_index,
            input_count,
            rows_per_input,
            item_bytes,
            secondary_item_bytes,
            secondary_fraction,
            secondary_row_fraction,
        )
        path = f"{output_root.rstrip('/')}/input-{input_index:04d}.parquet"
        first_table = _frame(
            min(rows, max(1, _FRAME_TARGET_BYTES // max(width + 24, 1))),
            width,
            seed + input_index * 100_000,
            row_offset=input_index,
            num_targets=0,
        ).to_arrow()
        estimated_size = 0
        written_rows = 0
        with open_url(path, "wb") as output:
            with pq.ParquetWriter(output, first_table.schema, compression="zstd") as writer:
                batch_index = 0
                while written_rows < rows:
                    batch_rows = min(len(first_table), rows - written_rows)
                    if batch_index == 0 and batch_rows == len(first_table):
                        table = first_table
                    else:
                        table = _frame(
                            batch_rows,
                            width,
                            seed + input_index * 100_000 + batch_index,
                            row_offset=written_rows * input_count + input_index,
                            num_targets=0,
                        ).to_arrow()
                    estimated_size += int(pl.from_arrow(table, rechunk=False).estimated_size())
                    writer.write_table(table)
                    written_rows += batch_rows
                    batch_index += 1
        input_paths.append(path)
        row_counts.append(rows)
        item_widths.append(width)
        estimated_sizes.append(estimated_size)

    total_rows = sum(row_counts)
    manifest = ReadManifest(
        input_paths=tuple(input_paths),
        row_counts=tuple(row_counts),
        item_bytes=tuple(item_widths),
        estimated_bytes=tuple(estimated_sizes),
        total_rows=total_rows,
        avg_item_bytes=sum(estimated_sizes) / total_rows,
    )
    StoragePath(f"{output_root.rstrip('/')}/{_MANIFEST_NAME}").write_text(json.dumps(manifest.to_json()))
    return manifest


def _load_read_manifest(input_root: str) -> ReadManifest:
    value = json.loads(StoragePath(f"{input_root.rstrip('/')}/{_MANIFEST_NAME}").read_text())
    return ReadManifest.from_json(value)


def _read_child(
    *,
    operation: str,
    input_root: str,
    output_root: str,
    input_count: int | None,
    streaming_rows: int,
    fan_in: int,
    repetition: int,
) -> dict[str, Any]:
    manifest = _load_read_manifest(input_root)
    paths = manifest.input_paths[:input_count]
    rows = manifest.row_counts[:input_count]
    estimated = manifest.estimated_bytes[:input_count]
    total_rows = sum(rows)
    avg_item_bytes = sum(estimated) / total_rows

    frames = _unify_frame_schemas([_scan_scatter_parquet(path) for path in paths])
    _warm_polars()
    baseline = max(_rss_bytes(), _process_peak_bytes())
    total_payload_bytes = sum(
        width * count for width, count in zip(manifest.item_bytes[:input_count], rows, strict=True)
    )
    task_memory_bytes = _task_memory_bytes()
    if operation == "read-planned":
        fan_in = memory_budget.read_merge_fan_in(
            task_memory_bytes,
            baseline,
            avg_item_bytes,
            len(frames),
            total_payload_bytes,
            pl.thread_pool_size(),
        )
    effective_fan_in = min(len(frames), fan_in) if operation in {"read-external", "read-planned"} else len(frames)
    bytes_at_risk = effective_fan_in * streaming_rows * avg_item_bytes
    mean_input_payload_bytes = total_payload_bytes / len(frames)
    model_active_rows = min(
        effective_fan_in * memory_budget.STREAMING_CHUNK_SIZE_ROWS,
        total_payload_bytes / avg_item_bytes,
    )
    model_batch_bytes = model_active_rows * avg_item_bytes
    model_expanded_batch_bytes = min(
        memory_budget.R_READ_MAX * model_batch_bytes,
        memory_budget.R_READ_PAYLOAD * model_batch_bytes
        + memory_budget.READ_ROW_OVERHEAD_BYTES * model_active_rows,
    )
    model_thread_shard_bytes = min(pl.thread_pool_size(), effective_fan_in) * mean_input_payload_bytes
    model_active_input_payload_bytes = min(total_payload_bytes, effective_fan_in * mean_input_payload_bytes)
    model_buffered_input_payload_bytes = max(0.0, model_active_input_payload_bytes - model_batch_bytes)
    model_predicted_growth_bytes = (
        memory_budget.FIXED_OVERHEAD_READ_BYTES
        + model_expanded_batch_bytes
        + memory_budget.R_READ_BUFFERED_INPUT_PAYLOAD * model_buffered_input_payload_bytes
        + memory_budget.R_READ_THREAD_SHARD * model_thread_shard_bytes
        + (memory_budget.R_READ_SPILL_PAYLOAD * total_payload_bytes if effective_fan_in < len(frames) else 0.0)
    )
    output_rows = 0
    output_payload_bytes = 0
    max_output_batch_rows = 0
    max_output_batch_bytes = 0
    started = time.monotonic()
    with pl.Config() as polars_config, RssSampler() as sampler:
        polars_config.set_streaming_chunk_size(streaming_rows)
        if operation == "read-collect":
            batches = _merge_sorted_frames(
                frames=frames,
                sort_key=_SORT_KEY_COL,
                external_sort_dir=f"{output_root.rstrip('/')}/spill-rep-{repetition}",
                fan_in=max(fan_in, len(frames)),
                shard=0,
            )
            for batch in batches:
                output_rows += len(batch)
                output_payload_bytes += int(batch[_PAYLOAD_COL].bin.size().sum())
                max_output_batch_rows = max(max_output_batch_rows, len(batch))
                max_output_batch_bytes = max(max_output_batch_bytes, int(batch.estimated_size()))
        elif operation == "read-sink":
            output_path = f"{output_root.rstrip('/')}/sink-rep-{repetition}.parquet"
            with open_url(output_path, "wb") as output:
                pl.merge_sorted(frames, key=_SORT_KEY_COL).sink_parquet(output, compression="zstd")
            output_rows = total_rows
            output_payload_bytes = sum(
                width * count for width, count in zip(manifest.item_bytes[:input_count], rows, strict=True)
            )
        elif operation in {"read-external", "read-planned"}:
            batches = _merge_sorted_frames(
                frames=frames,
                sort_key=_SORT_KEY_COL,
                external_sort_dir=f"{output_root.rstrip('/')}/spill-rep-{repetition}",
                fan_in=fan_in,
                shard=0,
            )
            for batch in batches:
                output_rows += len(batch)
                output_payload_bytes += int(batch[_PAYLOAD_COL].bin.size().sum())
                max_output_batch_rows = max(max_output_batch_rows, len(batch))
                max_output_batch_bytes = max(max_output_batch_bytes, int(batch.estimated_size()))
        else:
            raise ValueError(f"unknown read operation {operation}")
    elapsed = time.monotonic() - started

    if output_rows != total_rows:
        raise RuntimeError(f"read row mismatch: expected {total_rows}, produced {output_rows}")

    peak = max(sampler.peak_bytes, _process_peak_bytes())
    return {
        **_common_result(
            operation=operation,
            baseline_rss_bytes=baseline,
            peak_rss_bytes=peak,
            bytes_at_risk=bytes_at_risk,
            elapsed=elapsed,
            repetition=repetition,
        ),
        "input_root": input_root,
        "input_count": len(frames),
        "streaming_rows": streaming_rows,
        "fan_in": fan_in,
        "effective_fan_in": effective_fan_in,
        "avg_item_bytes": avg_item_bytes,
        "input_rows": total_rows,
        "output_rows": output_rows,
        "output_payload_bytes": output_payload_bytes,
        "max_output_batch_rows": max_output_batch_rows,
        "max_output_batch_bytes": max_output_batch_bytes,
        "task_memory_bytes": task_memory_bytes,
        "total_payload_bytes": total_payload_bytes,
        "mean_input_payload_bytes": mean_input_payload_bytes,
        "model_batch_bytes": model_batch_bytes,
        "model_expanded_batch_bytes": model_expanded_batch_bytes,
        "model_thread_shard_bytes": model_thread_shard_bytes,
        "model_active_input_payload_bytes": model_active_input_payload_bytes,
        "model_buffered_input_payload_bytes": model_buffered_input_payload_bytes,
        "model_predicted_growth_bytes": model_predicted_growth_bytes,
        "model_utilization": (peak - baseline) / model_predicted_growth_bytes,
    }


def _run_children(repetitions: int, threads: int, allow_child_failure: bool) -> None:
    child_args = [arg for arg in sys.argv[1:] if arg not in {"--child"}]
    env = os.environ.copy()
    env["POLARS_MAX_THREADS"] = str(threads)
    for repetition in range(-1, repetitions):
        command = [sys.executable, __file__, *child_args, "--child", "--repetition", str(repetition)]
        completed = subprocess.run(command, env=env, check=False)
        if completed.returncode != 0:
            failure = {
                "operation": "child-failure",
                "repetition": repetition,
                "returncode": completed.returncode,
            }
            print(_RESULT_PREFIX + json.dumps(failure), flush=True)
            if not allow_child_failure:
                raise SystemExit(completed.returncode)


@click.command()
@click.option(
    "--operation",
    type=click.Choice(["write", "prepare-read", "read-collect", "read-sink", "read-external", "read-planned"]),
    required=True,
)
@click.option("--target-estimated-bytes", type=str, default="64MiB")
@click.option("--rows", type=int, default=None)
@click.option("--item-bytes", type=int, default=200)
@click.option("--num-targets", type=int, default=128)
@click.option("--hot-shard-fraction", type=float, default=0.0)
@click.option("--input-count", type=int, default=None)
@click.option("--rows-per-input", type=int, default=10_000)
@click.option("--secondary-item-bytes", type=int, default=None)
@click.option("--secondary-fraction", type=float, default=0.0)
@click.option("--secondary-row-fraction", type=float, default=0.0)
@click.option("--streaming-rows", type=int, default=10_000)
@click.option("--fan-in", type=int, default=8)
@click.option("--input-root", type=str, default=None)
@click.option("--output-root", type=str, default=None)
@click.option("--seed", type=int, default=17)
@click.option("--threads", type=int, default=2)
@click.option("--repetitions", type=int, default=5)
@click.option("--auto-flush", is_flag=True)
@click.option("--allow-child-failure", is_flag=True)
@click.option("--child", is_flag=True, hidden=True)
@click.option("--repetition", type=int, default=0, hidden=True)
def main(
    operation: str,
    target_estimated_bytes: str,
    rows: int | None,
    item_bytes: int,
    num_targets: int,
    hot_shard_fraction: float,
    input_count: int | None,
    rows_per_input: int,
    secondary_item_bytes: int | None,
    secondary_fraction: float,
    secondary_row_fraction: float,
    streaming_rows: int,
    fan_in: int,
    input_root: str | None,
    output_root: str | None,
    seed: int,
    threads: int,
    repetitions: int,
    auto_flush: bool,
    allow_child_failure: bool,
    child: bool,
    repetition: int,
) -> None:
    """Run one memory-calibration cell."""
    run_id = os.environ.get("CALIBRATION_RUN_ID", f"local-{int(time.time())}")
    if output_root is None:
        output_root = marin_temp_bucket(ttl_days=1, prefix=f"zephyr-shuffle-calibration/{run_id}")
    _validate_calibration_path(output_root, "output")

    if operation == "prepare-read":
        if input_count is None:
            raise click.UsageError("--input-count is required for prepare-read")
        manifest = _prepare_read_inputs(
            input_count=input_count,
            rows_per_input=rows_per_input,
            item_bytes=item_bytes,
            secondary_item_bytes=secondary_item_bytes,
            secondary_fraction=secondary_fraction,
            secondary_row_fraction=secondary_row_fraction,
            output_root=output_root,
            seed=seed,
        )
        print(_RESULT_PREFIX + json.dumps({"operation": operation, **manifest.to_json()}), flush=True)
        return

    if not child:
        _run_children(repetitions, threads, allow_child_failure)
        return

    if operation == "write":
        result = _write_child(
            target_estimated_bytes=_parse_bytes(target_estimated_bytes),
            exact_rows=rows,
            item_bytes=item_bytes,
            num_targets=num_targets,
            hot_shard_fraction=hot_shard_fraction,
            output_root=output_root,
            seed=seed,
            repetition=repetition,
            auto_flush=auto_flush,
        )
    else:
        if input_root is None:
            raise click.UsageError("--input-root is required for read operations")
        _validate_calibration_path(input_root, "input")
        result = _read_child(
            operation=operation,
            input_root=input_root,
            output_root=output_root,
            input_count=input_count,
            streaming_rows=streaming_rows,
            fan_in=fan_in,
            repetition=repetition,
        )
    print(_RESULT_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
