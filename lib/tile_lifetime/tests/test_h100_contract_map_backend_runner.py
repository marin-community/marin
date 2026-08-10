# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import importlib
import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tile_lifetime.h100_contract_map_benchmark import (
    ArchitectureStatus,
    BackendVariant,
    default_h100_contract_map_benchmark_plan,
)

runner = importlib.import_module("lib.tile_lifetime.benchmarks.h100_contract_map_backend_runner")


def test_runner_preflight_records_exact_clean_source_tools_and_h100(tmp_path: Path) -> None:
    config = _runner_config(tmp_path)

    def run(command, **kwargs):
        arguments = tuple(str(value) for value in command)
        if arguments[1:3] == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(arguments, 0, config.source_sha + "\n", "")
        if arguments[1:3] == ("status", "--porcelain"):
            return subprocess.CompletedProcess(arguments, 0, "", "")
        if "--query-gpu=name,compute_cap" in arguments:
            return subprocess.CompletedProcess(arguments, 0, "NVIDIA H100 80GB HBM3, 9.0\n", "")
        return subprocess.CompletedProcess(arguments, 0, f"{Path(arguments[0]).name} exact-version\n", "")

    evidence = runner.require_clean_h100_preflight(config, run=run)

    assert evidence.source_sha == config.source_sha
    assert evidence.gpu_name == "NVIDIA H100 80GB HBM3"
    assert evidence.compute_capability == "9.0"
    assert evidence.architecture == "sm_90a"
    assert tuple(tool.name for tool in evidence.tools) == (
        "git",
        "nvidia_smi",
        "nvcc",
        "ptxas",
        "cuobjdump",
        "ncu",
        "nsys",
    )
    assert all(len(tool.sha256) == 64 and tool.version_output.endswith("exact-version") for tool in evidence.tools)


def test_runner_preflight_rejects_existing_artifact_directory_before_commands(tmp_path: Path) -> None:
    config = _runner_config(tmp_path)
    config.artifact_directory.mkdir()

    def run(*args, **kwargs):
        raise AssertionError("preflight must reject before invoking external tools")

    with pytest.raises(ValueError, match="must not already exist"):
        runner.require_clean_h100_preflight(config, run=run)


def test_runner_preflight_rejects_tracked_or_untracked_source_changes(tmp_path: Path) -> None:
    config = _runner_config(tmp_path)

    def run(command, **kwargs):
        arguments = tuple(str(value) for value in command)
        if arguments[1:3] == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(arguments, 0, config.source_sha + "\n", "")
        if arguments[1:3] == ("status", "--porcelain"):
            assert "--untracked-files=all" in arguments
            return subprocess.CompletedProcess(arguments, 0, "?? injected.py\n", "")
        raise AssertionError(f"preflight continued after dirty source: {arguments}")

    with pytest.raises(ValueError, match="no modifications or untracked files"):
        runner.require_clean_h100_preflight(config, run=run)


def test_runner_ncu_parser_requires_every_closed_launch_metric(tmp_path: Path) -> None:
    output = tmp_path / "ncu.csv"
    _write_ncu_csv(output)

    records = runner.parse_ncu_metrics(output)

    assert records == (
        runner.NcuKernelMetrics(
            name="KernelA",
            block_size=(256, 1, 1),
            registers_per_thread=48,
            static_shared_memory_bytes=128,
            dynamic_shared_memory_bytes=0,
            active_blocks_per_sm=2,
            limiting_occupancy_resource="registers,warps",
            achieved_occupancy=0.625,
        ),
    )
    lines = output.read_text().splitlines()
    output.write_text("\n".join(line for line in lines if "launch__registers_per_thread" not in line) + "\n")
    with pytest.raises(ValueError, match="omits metrics"):
        runner.parse_ncu_metrics(output)


def test_runner_nsys_parser_attributes_kernel_and_copy_activity_to_exact_ranges(tmp_path: Path) -> None:
    database_path = tmp_path / "trace.sqlite"
    with sqlite3.connect(database_path) as database:
        database.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT)")
        database.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        database.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER)")
        database.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, bytes INTEGER, copyKind INTEGER)"
        )
        database.execute("INSERT INTO NVTX_EVENTS VALUES (100, 400, 'contract_map.steady.0.ordinary_xla')")
        database.execute("INSERT INTO StringIds VALUES (7, 'fusion_kernel')")
        database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (150, 250, 7)")
        database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (260, 280, 64, 8)")

    records = runner.parse_nsys_sqlite(database_path, ("contract_map.steady.0.ordinary_xla",))

    assert records == (
        runner.TraceRange(
            name="contract_map.steady.0.ordinary_xla",
            ordered_kernel_names=("fusion_kernel",),
            kernel_duration_ns=100,
            device_to_device_count=1,
            device_to_device_bytes=64,
            host_to_device_count=0,
            host_to_device_bytes=0,
            unexpected_copy_count=0,
        ),
    )


def test_runner_merges_device_and_logical_timings_only_for_exact_copy_free_schedule() -> None:
    plan = default_h100_contract_map_benchmark_plan()
    worker_rows = []
    traces = []
    for row in plan.timing.steady_state_schedule:
        worker_rows.append(
            {
                "sample_index": row.sample_index,
                "backend_order": [backend.value for backend in row.backend_order],
                "logical_training_step_ns": {backend.value: 10_000 + row.sample_index for backend in BackendVariant},
            }
        )
        for backend in row.backend_order:
            traces.append(
                runner.TraceRange(
                    name=f"contract_map.steady.{row.sample_index}.{backend.value}",
                    ordered_kernel_names=("first", "second") * plan.timing.iterations_per_sample,
                    kernel_duration_ns=2_000 * plan.timing.iterations_per_sample,
                    device_to_device_count=0,
                    device_to_device_bytes=0,
                    host_to_device_count=0,
                    host_to_device_bytes=0,
                    unexpected_copy_count=0,
                )
            )

    raw, summary = runner.merge_trace_timing(plan, {"raw_samples": worker_rows}, tuple(traces))

    assert len(raw) == 24
    assert tuple(raw[0]["measurements_ns"]) == tuple(backend.value for backend in BackendVariant)
    assert raw[0]["measurements_ns"][BackendVariant.ORDINARY_XLA.value] == {
        "kernel_only": 2_000,
        "logical_training_step": 10_000,
    }
    assert summary[BackendVariant.SHUTTLE_FAST.value]["launch_count"] == 2
    copied = replace(traces[0], device_to_device_count=1, device_to_device_bytes=64)
    with pytest.raises(ValueError, match="unexpected copies"):
        runner.merge_trace_timing(plan, {"raw_samples": worker_rows}, (copied, *traces[1:]))


def test_runner_rejects_missing_ordinary_xla_ptx_and_cubin(tmp_path: Path) -> None:
    config = _runner_config(tmp_path)
    dump_directory = tmp_path / "compile_dump"
    dump_directory.mkdir()

    with pytest.raises(RuntimeError, match="must dump exactly one ordinary-XLA PTX and cubin"):
        runner.retain_ordinary_xla_cuda_artifacts(
            config,
            dump_directory=dump_directory,
            retained_directory=tmp_path / "retained",
        )


def test_worker_environment_replaces_inherited_cache_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/inherited/cache")
    monkeypatch.setenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "99")

    uncached = runner._worker_environment(tmp_path / "uncached_dump")
    isolated = runner._worker_environment(tmp_path / "cached_dump", tmp_path / "isolated_cache")

    assert "JAX_COMPILATION_CACHE_DIR" not in uncached
    assert "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in uncached
    assert isolated["JAX_COMPILATION_CACHE_DIR"] == str((tmp_path / "isolated_cache").resolve())
    assert isolated["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0"


def test_runner_never_publishes_an_invalid_or_conforming_bundle(tmp_path: Path) -> None:
    output = tmp_path / "accepted_bundle.json"
    with pytest.raises(ValueError, match="architecture-nonconforming"):
        runner.publish_validated_bundle(output, {"architecture_status": "conforming", "records": ()})
    assert not output.exists()
    with pytest.raises(ValueError, match="all 24"):
        runner.publish_validated_bundle(
            output,
            {"architecture_status": ArchitectureStatus.NONCONFORMING.value, "records": ()},
        )
    assert not output.exists()


def _runner_config(tmp_path: Path) -> Any:
    source_root = tmp_path / "source"
    source_root.mkdir()
    toolkit = tmp_path / "cuda" / "bin"
    toolkit.mkdir(parents=True)
    tools = tmp_path / "tools"
    tools.mkdir()

    def executable(path: Path) -> Path:
        path.write_text(path.name)
        path.chmod(0o755)
        return path.resolve()

    return runner.RunnerConfig(
        source_root=source_root,
        source_sha="1" * 40,
        artifact_directory=tmp_path / "artifacts",
        tools=runner.ToolPaths(
            git=executable(tools / "git"),
            nvidia_smi=executable(tools / "nvidia-smi"),
            nvcc=executable(toolkit / "nvcc"),
            ptxas=executable(toolkit / "ptxas"),
            cuobjdump=executable(toolkit / "cuobjdump"),
            ncu=executable(tools / "ncu"),
            nsys=executable(tools / "nsys"),
        ),
        require_jax_version="1.2.3",
    )


def _write_ncu_csv(path: Path) -> None:
    values = {
        "launch__block_size": "256",
        "launch__registers_per_thread": "48",
        "launch__shared_mem_per_block_static": "128",
        "launch__shared_mem_per_block_dynamic": "0",
        "launch__occupancy_limit_blocks": "4",
        "launch__occupancy_limit_registers": "2",
        "launch__occupancy_limit_shared_mem": "8",
        "launch__occupancy_limit_warps": "2",
        "sm__warps_active.avg.pct_of_peak_sustained_active": "62.5",
    }
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ID", "Kernel Name", "Metric Name", "Metric Value"))
        writer.writeheader()
        for metric, value in values.items():
            writer.writerow({"ID": "1", "Kernel Name": "KernelA", "Metric Name": metric, "Metric Value": value})
