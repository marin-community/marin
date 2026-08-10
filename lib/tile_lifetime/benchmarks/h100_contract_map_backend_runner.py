# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute the reviewed generic Contract/Map H100 evidence protocol.

The coordinator is fail-closed: it writes an accepted bundle only after every
worker, compiler artifact, Nsight Compute metric, Nsight Systems CUDA activity,
numerical floor, and the existing 24-record validator succeeds.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

_ARCHITECTURE = "sm_90a"
_COMPUTE_CAPABILITY = "9.0"
_OUTPUT_NAMES = ("forward", "dx", "dw0", "dw1")
_CACHE_ENVIRONMENT = {
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
}
_NCU_METRICS = (
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)


class WorkerMode(StrEnum):
    """Isolated process roles used by the coordinator."""

    CASE = "case"
    COMPILE = "compile"
    PROFILE = "profile"


@dataclass(frozen=True)
class ToolPaths:
    """Exact external tools admitted by the H100 evidence runner."""

    git: Path
    nvidia_smi: Path
    nvcc: Path
    ptxas: Path
    cuobjdump: Path
    ncu: Path
    nsys: Path

    def items(self) -> tuple[tuple[str, Path], ...]:
        return tuple((field, getattr(self, field)) for field in self.__dataclass_fields__)


@dataclass(frozen=True)
class RunnerConfig:
    """Closed coordinator configuration checked before importing JAX."""

    source_root: Path
    source_sha: str
    artifact_directory: Path
    tools: ToolPaths
    require_jax_version: str


@dataclass(frozen=True)
class ToolIdentity:
    """Immutable path, content hash, and version output for one executable."""

    name: str
    path: str
    sha256: str
    version_output: str


@dataclass(frozen=True)
class PreflightEvidence:
    """Source, device, and tool identities captured before artifact creation."""

    source_sha: str
    gpu_name: str
    compute_capability: str
    architecture: str
    tools: tuple[ToolIdentity, ...]


@dataclass(frozen=True)
class NcuKernelMetrics:
    """Required launch and occupancy metrics for one profiled kernel launch."""

    name: str
    block_size: tuple[int, int, int]
    registers_per_thread: int
    static_shared_memory_bytes: int
    dynamic_shared_memory_bytes: int
    active_blocks_per_sm: int
    limiting_occupancy_resource: str
    achieved_occupancy: float


@dataclass(frozen=True)
class NcuProfileEvidence:
    """Metrics and retained profiler source/SASS export for one execution."""

    metrics: tuple[NcuKernelMetrics, ...]
    report_path: str
    report_sha256: str
    sass_source_path: str
    sass_source_sha256: str
    final_hlo: str


@dataclass(frozen=True)
class TraceRange:
    """CUDA activity contained by one exact steady-state NVTX range."""

    name: str
    ordered_kernel_names: tuple[str, ...]
    kernel_duration_ns: int
    device_to_device_count: int
    device_to_device_bytes: int
    host_to_device_count: int
    host_to_device_bytes: int
    unexpected_copy_count: int


@dataclass(frozen=True)
class GeneratedArtifact:
    """Compiled source and retained CUDA artifacts for one generated backend."""

    case_id: str
    backend: str
    physical_digest: str
    source_path: str
    source_sha256: str
    shared_library_path: str
    shared_library_sha256: str
    ptx_path: str
    ptx_sha256: str
    cubin_path: str
    cubin_sha256: str
    cubin_sass_path: str
    cubin_sass_sha256: str
    loaded_image_sass_path: str
    loaded_image_sass_sha256: str
    compiler_flags: tuple[str, ...]
    ptxas_resources: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class OrdinaryXlaExecutableEvidence:
    """Final-HLO-derived ordinary-XLA ABI and structural manifest."""

    kernel_only_boundary: dict[str, Any]
    logical_training_step_boundary: dict[str, Any]
    manifest: dict[str, Any]


def file_sha256(path: Path) -> str:
    """Return the SHA-256 content identity of one retained artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_cuda_kernel_name(name: str) -> str:
    """Return one exact comparison identity for profiler kernel names."""
    normalized = " ".join(name.strip().split())
    if normalized.startswith("void "):
        normalized = normalized.removeprefix("void ")
    simple = re.fullmatch(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\(.*\))?", normalized)
    if simple is not None:
        return simple.group("name")
    if not normalized:
        raise ValueError("CUDA kernel identity must be nonempty")
    return normalized


def cuda_sass_kernel_names(sass: str) -> tuple[str, ...]:
    """Return exact normalized entry names from a cuobjdump SASS artifact."""
    names = tuple(
        normalize_cuda_kernel_name(match.group("name"))
        for match in re.finditer(r"^\s*Function\s*:\s*(?P<name>\S+)\s*$", sass, flags=re.MULTILINE)
    )
    if not names:
        raise ValueError("cuobjdump SASS contains no function identities")
    if len(set(names)) != len(names):
        raise ValueError("cuobjdump SASS repeats a function identity")
    return names


def require_clean_h100_preflight(
    config: RunnerConfig,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> PreflightEvidence:
    """Validate exact source, tools, one H100, and a fresh output path."""
    if config.artifact_directory.exists():
        raise ValueError(f"artifact directory must not already exist: {config.artifact_directory}")
    if not config.source_root.is_dir():
        raise ValueError(f"source root does not exist: {config.source_root}")
    if config.artifact_directory.is_relative_to(config.source_root):
        raise ValueError("artifact directory must be outside the exact source worktree")
    if len(config.source_sha) != 40 or any(character not in "0123456789abcdef" for character in config.source_sha):
        raise ValueError("source_sha must be a full lowercase Git SHA")
    if not config.require_jax_version.strip():
        raise ValueError("require_jax_version must be nonempty")
    for name, path in config.tools.items():
        if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"{name} must be an absolute executable file: {path}")
    toolkit_bin = config.tools.nvcc.parent
    if config.tools.ptxas.parent != toolkit_bin or config.tools.cuobjdump.parent != toolkit_bin:
        raise ValueError("nvcc, ptxas, and cuobjdump must come from one CUDA toolkit bin directory")

    head = _checked_output(run, (str(config.tools.git), "rev-parse", "HEAD"), cwd=config.source_root)
    if head != config.source_sha:
        raise ValueError(f"source SHA mismatch: checkout is {head}, required {config.source_sha}")
    dirty = _checked_output(
        run,
        (str(config.tools.git), "status", "--porcelain", "--untracked-files=all"),
        cwd=config.source_root,
        allow_empty=True,
    )
    if dirty:
        raise ValueError("source worktree must have no modifications or untracked files")

    gpu_lines = _checked_output(
        run,
        (
            str(config.tools.nvidia_smi),
            "--query-gpu=name,compute_cap",
            "--format=csv,noheader,nounits",
        ),
    ).splitlines()
    if len(gpu_lines) != 1:
        raise RuntimeError(f"runner requires exactly one visible H100, found {gpu_lines}")
    fields = tuple(field.strip() for field in gpu_lines[0].split(","))
    if len(fields) != 2 or "H100" not in fields[0] or fields[1] != _COMPUTE_CAPABILITY:
        raise RuntimeError(f"runner requires an H100 with compute capability 9.0, found {gpu_lines[0]!r}")

    versions = {
        "git": (str(config.tools.git), "--version"),
        "nvidia_smi": (str(config.tools.nvidia_smi), "--version"),
        "nvcc": (str(config.tools.nvcc), "--version"),
        "ptxas": (str(config.tools.ptxas), "--version"),
        "cuobjdump": (str(config.tools.cuobjdump), "--version"),
        "ncu": (str(config.tools.ncu), "--version"),
        "nsys": (str(config.tools.nsys), "--version"),
    }
    identities = tuple(
        ToolIdentity(
            name=name,
            path=str(path),
            sha256=file_sha256(path),
            version_output=_checked_output(run, versions[name]),
        )
        for name, path in config.tools.items()
    )
    return PreflightEvidence(
        source_sha=head,
        gpu_name=fields[0],
        compute_capability=fields[1],
        architecture=_ARCHITECTURE,
        tools=identities,
    )


def _checked_output(
    run: Callable[..., subprocess.CompletedProcess[str]],
    command: tuple[str, ...],
    *,
    cwd: Path | None = None,
    allow_empty: bool = False,
) -> str:
    completed = run(command, cwd=cwd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with {completed.returncode}: {command}: {completed.stderr.strip()}")
    output = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
    if not output and not allow_empty:
        raise RuntimeError(f"command returned no identity output: {command}")
    return output


def parse_ncu_metrics(path: Path) -> tuple[NcuKernelMetrics, ...]:
    """Parse the exact raw Nsight Compute metric rows and reject omissions."""
    rows = tuple(csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith("==")))
    if not rows:
        raise ValueError("Nsight Compute output contains no metric rows")
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        name = normalize_cuda_kernel_name(_csv_field(row, "Kernel Name"))
        identifier = _csv_field(row, "ID")
        key = (identifier, name)
        if key not in grouped:
            grouped[key] = {}
            order.append(key)
        metric = _csv_field(row, "Metric Name")
        value = _csv_field(row, "Metric Value")
        if metric in grouped[key]:
            raise ValueError(f"Nsight Compute repeats metric {metric!r} for kernel {name!r}")
        grouped[key][metric] = value

    records: list[NcuKernelMetrics] = []
    for key in order:
        name = key[1]
        metrics = grouped[key]
        missing = tuple(metric for metric in _NCU_METRICS if metric not in metrics)
        if missing:
            raise ValueError(f"Nsight Compute omits metrics for kernel {name!r}: {missing}")
        limits = {
            "blocks": _metric_int(metrics["launch__occupancy_limit_blocks"]),
            "registers": _metric_int(metrics["launch__occupancy_limit_registers"]),
            "shared_memory": _metric_int(metrics["launch__occupancy_limit_shared_mem"]),
            "warps": _metric_int(metrics["launch__occupancy_limit_warps"]),
        }
        active_blocks = min(limits.values())
        limiting = ",".join(name for name, value in limits.items() if value == active_blocks)
        records.append(
            NcuKernelMetrics(
                name=name,
                block_size=(_metric_int(metrics["launch__block_size"]), 1, 1),
                registers_per_thread=_metric_int(metrics["launch__registers_per_thread"]),
                static_shared_memory_bytes=_metric_int(metrics["launch__shared_mem_per_block_static"]),
                dynamic_shared_memory_bytes=_metric_int(metrics["launch__shared_mem_per_block_dynamic"]),
                active_blocks_per_sm=active_blocks,
                limiting_occupancy_resource=limiting,
                achieved_occupancy=_metric_float(metrics["sm__warps_active.avg.pct_of_peak_sustained_active"]) / 100.0,
            )
        )
    return tuple(records)


def _csv_field(row: Mapping[str | None, str | list[str] | None], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Nsight Compute row omits {name!r}: {row}")
    return value.strip()


def _metric_float(value: str) -> float:
    normalized = value.replace(",", "").replace("%", "").strip()
    parsed = float(normalized)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"Nsight Compute metric must be finite and nonnegative: {value!r}")
    return parsed


def _metric_int(value: str) -> int:
    parsed = _metric_float(value)
    if not parsed.is_integer():
        raise ValueError(f"Nsight Compute integer metric is fractional: {value!r}")
    return int(parsed)


def parse_nsys_sqlite(path: Path, expected_ranges: tuple[str, ...]) -> tuple[TraceRange, ...]:
    """Read CUDA kernels and copies contained by each required NVTX range."""
    with sqlite3.connect(path) as database:
        tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"}
        if not required.issubset(tables):
            raise ValueError(f"Nsight Systems SQLite export omits CUPTI tables: {tuple(sorted(required - tables))}")
        ranges = _nsys_ranges(database)
        kernels = _nsys_kernels(database)
        copies = _nsys_copies(database)

    if tuple(ranges) != expected_ranges:
        raise ValueError("Nsight Systems NVTX ranges do not match the exact steady-state schedule")
    records = []
    for name in expected_ranges:
        start, end = ranges[name]
        contained_kernels = tuple(kernel for kernel in kernels if start <= kernel[0] and kernel[1] <= end)
        contained_copies = tuple(copy for copy in copies if start <= copy[0] and copy[1] <= end)
        if not contained_kernels:
            raise ValueError(f"Nsight Systems range {name!r} contains no CUDA kernels")
        d2d = tuple(copy for copy in contained_copies if copy[3] == "device_to_device")
        h2d = tuple(copy for copy in contained_copies if copy[3] == "host_to_device")
        unexpected = tuple(copy for copy in contained_copies if copy[3] not in {"device_to_device", "host_to_device"})
        records.append(
            TraceRange(
                name=name,
                ordered_kernel_names=tuple(kernel[2] for kernel in contained_kernels),
                kernel_duration_ns=sum(kernel[1] - kernel[0] for kernel in contained_kernels),
                device_to_device_count=len(d2d),
                device_to_device_bytes=sum(copy[2] for copy in d2d),
                host_to_device_count=len(h2d),
                host_to_device_bytes=sum(copy[2] for copy in h2d),
                unexpected_copy_count=len(unexpected),
            )
        )
    return tuple(records)


def _nsys_ranges(database: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    columns = _table_columns(database, "NVTX_EVENTS")
    for required in ("start", "end", "text"):
        if required not in columns:
            raise ValueError(f"NVTX_EVENTS omits required column {required!r}")
    records: dict[str, tuple[int, int]] = {}
    for start, end, text in database.execute("SELECT start, end, text FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        if isinstance(text, str) and text.startswith("contract_map.steady."):
            if text in records:
                raise ValueError(f"Nsight Systems repeats steady-state NVTX range {text!r}")
            records[text] = (int(start), int(end))
    return records


def _nsys_kernels(database: sqlite3.Connection) -> tuple[tuple[int, int, str], ...]:
    columns = _table_columns(database, "CUPTI_ACTIVITY_KIND_KERNEL")
    name_column = "demangledName" if "demangledName" in columns else "shortName"
    if not {"start", "end", name_column}.issubset(columns):
        raise ValueError("CUPTI kernel table omits start, end, or kernel-name identity")
    query = (
        f"SELECT kernel.start, kernel.end, strings.value "
        f"FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel JOIN StringIds AS strings "
        f"ON kernel.{name_column} = strings.id ORDER BY kernel.start"
    )
    return tuple(
        (int(start), int(end), normalize_cuda_kernel_name(str(name))) for start, end, name in database.execute(query)
    )


def _nsys_copies(database: sqlite3.Connection) -> tuple[tuple[int, int, int, str], ...]:
    columns = _table_columns(database, "CUPTI_ACTIVITY_KIND_MEMCPY")
    if not {"start", "end", "bytes", "copyKind"}.issubset(columns):
        raise ValueError("CUPTI memcpy table omits start, end, bytes, or copyKind")
    kinds = {1: "host_to_device", 2: "device_to_host", 8: "device_to_device"}
    return tuple(
        (int(start), int(end), int(size), kinds.get(int(kind), f"copy_kind_{kind}"))
        for start, end, size, kind in database.execute(
            "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"
        )
    )


def _table_columns(database: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in database.execute(f"PRAGMA table_info({table})")}


def compile_generated_candidates(config: RunnerConfig) -> tuple[GeneratedArtifact, ...]:
    """Compile both generated policies for all four anonymous cases."""
    # JAX and tile_lifetime imports stay behind the package-independent preflight.
    import jaxlib  # noqa: PLC0415

    from tile_lifetime.contract_map_backend_resources import (  # noqa: PLC0415
        contract_map_compile_plan,
        parse_ptxas_kernel_resources,
    )
    from tile_lifetime.cuda_toolchain import (  # noqa: PLC0415
        cuda_toolkit_link_flags,
        cuda_toolkit_shared_library_link_flags,
    )

    training = importlib.import_module("lib.tile_lifetime.benchmarks.h100_contract_map_backend_training")

    include_directory = Path(jaxlib.__file__).resolve().parent / "include"
    ffi_header = include_directory / "xla/ffi/api/ffi.h"
    if not ffi_header.is_file():
        raise ValueError(f"JAX typed-FFI header does not exist: {ffi_header}")
    artifacts = []
    for candidate in training.generated_contract_map_candidates():
        directory = config.artifact_directory / "generated" / candidate.case.case_id / candidate.backend.value
        directory.mkdir(parents=True)
        plan = contract_map_compile_plan(
            candidate.generated,
            artifact_directory=directory,
            nvcc=config.tools.nvcc,
            include_directory=include_directory,
        )
        plan.source_path.write_text(candidate.generated.source + "\n")
        shared_command = (
            *plan.shared_library_command[:-2],
            "-cudart=none",
            *cuda_toolkit_link_flags(config.tools.nvcc, runtime_search_path=True),
            *cuda_toolkit_shared_library_link_flags(config.tools.nvcc, ("cudart",)),
            "-o",
            str(plan.shared_library_path),
        )
        shared = _run_retained(shared_command)
        _run_retained(plan.ptx_command)
        _run_retained(plan.cubin_command)
        cubin_sass = _run_retained(plan.sass_command)
        plan.sass_path.write_text(cubin_sass.stdout)
        loaded_image_sass_path = plan.shared_library_path.with_suffix(".loaded.sass")
        loaded_image_sass = _run_retained((str(config.tools.cuobjdump), "--dump-sass", str(plan.shared_library_path)))
        if cuda_sass_kernel_names(loaded_image_sass.stdout) != tuple(candidate.generated.kernel_names):
            raise RuntimeError("loaded shared-library SASS does not contain the exact generated kernel topology")
        loaded_image_sass_path.write_text(loaded_image_sass.stdout)
        for path in (
            plan.source_path,
            plan.shared_library_path,
            plan.ptx_path,
            plan.cubin_path,
            plan.sass_path,
            loaded_image_sass_path,
        ):
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"CUDA compilation omitted required artifact: {path}")
        ptxas_output = "\n".join(part for part in (shared.stdout, shared.stderr) if part)
        resources = parse_ptxas_kernel_resources(
            ptxas_output,
            expected_kernel_names=candidate.generated.kernel_names,
        )
        artifacts.append(
            GeneratedArtifact(
                case_id=candidate.case.case_id,
                backend=candidate.backend.value,
                physical_digest=candidate.generated.physical_digest,
                source_path=str(plan.source_path),
                source_sha256=file_sha256(plan.source_path),
                shared_library_path=str(plan.shared_library_path),
                shared_library_sha256=file_sha256(plan.shared_library_path),
                ptx_path=str(plan.ptx_path),
                ptx_sha256=file_sha256(plan.ptx_path),
                cubin_path=str(plan.cubin_path),
                cubin_sha256=file_sha256(plan.cubin_path),
                cubin_sass_path=str(plan.sass_path),
                cubin_sass_sha256=file_sha256(plan.sass_path),
                loaded_image_sass_path=str(loaded_image_sass_path),
                loaded_image_sass_sha256=file_sha256(loaded_image_sass_path),
                compiler_flags=shared_command,
                ptxas_resources=tuple(asdict(resource) for resource in resources),
            )
        )
    return tuple(artifacts)


def _run_retained(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"CUDA artifact command failed with {completed.returncode}: {command}: "
            f"{completed.stdout}\n{completed.stderr}"
        )
    return completed


def run_worker(args: argparse.Namespace) -> None:
    """Run one isolated JAX process role and write its structured result."""
    # These optional accelerator imports must observe worker-specific cache and
    # XLA dump environment variables set before process startup.
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import jaxlib  # noqa: PLC0415

    if jax.__version__ != args.require_jax_version or jaxlib.__version__ != args.require_jax_version:
        raise RuntimeError(
            f"worker requires jax/jaxlib {args.require_jax_version}, found {jax.__version__}/{jaxlib.__version__}"
        )
    for distribution in ("jax-cuda13-plugin", "jax-cuda13-pjrt"):
        if importlib.metadata.version(distribution) != args.require_jax_version:
            raise RuntimeError(f"worker requires {distribution} {args.require_jax_version}")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or "H100" not in devices[0].device_kind:
        raise RuntimeError(f"worker requires exactly one visible H100, found {devices}")

    context = _worker_case_context(args, jax=jax, jnp=jnp)
    if args.worker is WorkerMode.COMPILE:
        result = _run_compile_worker(args, context, jax=jax)
    elif args.worker is WorkerMode.CASE:
        result = _run_case_worker(args, context, jax=jax)
    elif args.worker is WorkerMode.PROFILE:
        result = _run_profile_worker(args, context, jax=jax)
    else:
        raise ValueError(f"unsupported worker mode: {args.worker}")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class _WorkerCaseContext:
    case: Any
    inputs: tuple[Any, Any, Any, Any]
    candidates: Mapping[str, Any]
    artifacts: Mapping[str, GeneratedArtifact]
    libraries: tuple[ctypes.CDLL, ...]


def _worker_case_context(args: argparse.Namespace, *, jax: Any, jnp: Any) -> _WorkerCaseContext:
    import numpy as np  # noqa: PLC0415

    from tile_lifetime.h100_contract_map_benchmark import BackendVariant  # noqa: PLC0415
    from tile_lifetime.jax_contract_map_backend_ffi import (  # noqa: PLC0415
        register_cuda_contract_map_backend_ffi,
    )

    training = importlib.import_module("lib.tile_lifetime.benchmarks.h100_contract_map_backend_training")
    all_candidates = training.generated_contract_map_candidates()
    selected = tuple(candidate for candidate in all_candidates if candidate.case.case_id == args.case_id)
    if len(selected) != 2:
        raise ValueError(f"case worker requires two generated candidates for {args.case_id!r}")
    artifacts_payload = json.loads(args.generated_manifest.read_text())
    artifacts = {
        record["backend"]: _generated_artifact_from_json(record)
        for record in artifacts_payload
        if record["case_id"] == args.case_id
    }
    expected_generated = {
        BackendVariant.SHUTTLE_SOURCE_ORDERED.value,
        BackendVariant.SHUTTLE_FAST.value,
    }
    if set(artifacts) != expected_generated:
        raise ValueError(f"generated manifest omits candidates for {args.case_id!r}")
    candidates = {candidate.backend.value: candidate for candidate in selected}
    libraries = []
    for backend in expected_generated:
        candidate = candidates[backend]
        artifact = artifacts[backend]
        if artifact.physical_digest != candidate.generated.physical_digest:
            raise ValueError(f"generated artifact identity mismatch for {args.case_id}/{backend}")
        library_path = Path(artifact.shared_library_path)
        if file_sha256(library_path) != artifact.shared_library_sha256:
            raise ValueError(f"generated shared-library content changed for {args.case_id}/{backend}")
        library = ctypes.CDLL(str(library_path))
        register_cuda_contract_map_backend_ffi(candidate.generated, library)
        libraries.append(library)

    case = selected[0].case
    rng = np.random.default_rng(int(case.case_id[-8:], 16))
    host_inputs = (
        (rng.normal(scale=0.15, size=(case.rows, case.reduction))).astype(np.float32),
        (rng.normal(scale=0.15, size=(case.reduction, case.features))).astype(np.float32),
        (rng.normal(scale=0.15, size=(case.features, case.reduction))).astype(np.float32),
        (rng.normal(scale=0.15, size=(case.rows, case.reduction))).astype(np.float32),
    )
    inputs = tuple(jax.device_put(jnp.asarray(value, dtype=jnp.bfloat16)) for value in host_inputs)
    jax.block_until_ready(inputs)
    return _WorkerCaseContext(
        case=case,
        inputs=inputs,
        candidates=candidates,
        artifacts=artifacts,
        libraries=tuple(libraries),
    )


def _generated_artifact_from_json(record: Mapping[str, Any]) -> GeneratedArtifact:
    return GeneratedArtifact(
        case_id=str(record["case_id"]),
        backend=str(record["backend"]),
        physical_digest=str(record["physical_digest"]),
        source_path=str(record["source_path"]),
        source_sha256=str(record["source_sha256"]),
        shared_library_path=str(record["shared_library_path"]),
        shared_library_sha256=str(record["shared_library_sha256"]),
        ptx_path=str(record["ptx_path"]),
        ptx_sha256=str(record["ptx_sha256"]),
        cubin_path=str(record["cubin_path"]),
        cubin_sha256=str(record["cubin_sha256"]),
        cubin_sass_path=str(record["cubin_sass_path"]),
        cubin_sass_sha256=str(record["cubin_sass_sha256"]),
        loaded_image_sass_path=str(record["loaded_image_sass_path"]),
        loaded_image_sass_sha256=str(record["loaded_image_sass_sha256"]),
        compiler_flags=tuple(str(value) for value in record["compiler_flags"]),
        ptxas_resources=tuple(dict(value) for value in record["ptxas_resources"]),
    )


def _compiled_backend(context: _WorkerCaseContext, backend: str, *, jax: Any) -> tuple[Any, int]:
    from tile_lifetime.h100_contract_map_benchmark import BackendVariant  # noqa: PLC0415
    from tile_lifetime.jax_contract_map_backend_ffi import (  # noqa: PLC0415
        call_cuda_contract_map_backend_forward_ffi,
        call_cuda_contract_map_backend_reverse_ffi,
    )

    training = importlib.import_module("lib.tile_lifetime.benchmarks.h100_contract_map_backend_training")
    if backend == BackendVariant.ORDINARY_XLA.value:

        def step(activation: Any, first_weight: Any, second_weight: Any, cotangent: Any) -> tuple[Any, ...]:
            return training.natural_jax_training_step(
                context.case.scalar_map,
                activation,
                first_weight,
                second_weight,
                cotangent,
            )

    else:
        candidate = context.candidates[backend]

        def step(activation: Any, first_weight: Any, second_weight: Any, cotangent: Any) -> tuple[Any, ...]:
            forward = call_cuda_contract_map_backend_forward_ffi(
                candidate.generated,
                activation,
                first_weight,
                second_weight,
            )
            reverse = call_cuda_contract_map_backend_reverse_ffi(
                candidate.generated,
                activation,
                first_weight,
                second_weight,
                forward.preactivation,
                forward.hidden,
                cotangent,
            )
            return (
                forward.output,
                reverse.input_adjoint,
                reverse.first_weight_adjoint,
                reverse.second_weight_adjoint,
            )

    started = time.perf_counter_ns()
    compiled = jax.jit(step).lower(*context.inputs).compile()
    compile_ns = time.perf_counter_ns() - started
    return compiled, compile_ns


def _run_compile_worker(args: argparse.Namespace, context: _WorkerCaseContext, *, jax: Any) -> dict[str, Any]:
    compiled, worker_compile_ns = _compiled_backend(context, args.backend, jax=jax)
    started = time.perf_counter_ns()
    output = compiled(*context.inputs)
    jax.block_until_ready(output)
    first_execution_ns = time.perf_counter_ns() - started
    cache_directory = Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", ""))
    if not cache_directory.is_dir():
        raise RuntimeError("compile worker requires an existing isolated JAX_COMPILATION_CACHE_DIR")
    cache_files = tuple(sorted(path for path in cache_directory.rglob("*") if path.is_file()))
    if not cache_files:
        raise RuntimeError("compile worker produced no persistent-cache artifact")
    identity = hashlib.sha256()
    for path in cache_files:
        identity.update(str(path.relative_to(cache_directory)).encode())
        identity.update(bytes.fromhex(file_sha256(path)))
    return {
        "case_id": args.case_id,
        "backend": args.backend,
        "cache_kind": args.cache_kind,
        "worker_compile_ns": worker_compile_ns,
        "first_execution_ns": first_execution_ns,
        "persistent_cache_identity": identity.hexdigest(),
        "final_hlo": compiled.as_text(),
    }


def _run_profile_worker(args: argparse.Namespace, context: _WorkerCaseContext, *, jax: Any) -> dict[str, Any]:
    compiled, _ = _compiled_backend(context, args.backend, jax=jax)
    jax.block_until_ready(compiled(*context.inputs))
    with _NvtxRange("contract_map.profile", args.nvcc):
        output = compiled(*context.inputs)
        jax.block_until_ready(output)
    return {
        "case_id": args.case_id,
        "backend": args.backend,
        "profiled": True,
        "final_hlo": compiled.as_text(),
    }


def _run_case_worker(args: argparse.Namespace, context: _WorkerCaseContext, *, jax: Any) -> dict[str, Any]:
    from tile_lifetime.h100_contract_map_benchmark import (  # noqa: PLC0415
        BackendVariant,
        default_h100_contract_map_benchmark_plan,
    )

    executables = {}
    for backend in BackendVariant:
        executable, _ = _compiled_backend(context, backend.value, jax=jax)
        executables[backend.value] = executable

    numerical = _numerical_evidence(context, executables, jax=jax)
    warmups: dict[str, list[int]] = {backend.value: [] for backend in BackendVariant}
    timing = default_h100_contract_map_benchmark_plan().timing
    for _ in range(timing.warmup_iterations):
        for backend in BackendVariant:
            started = time.perf_counter_ns()
            output = executables[backend.value](*context.inputs)
            jax.block_until_ready(output)
            warmups[backend.value].append(time.perf_counter_ns() - started)

    raw_samples = []
    profiler = _CudaProfiler(args.nvcc)
    profiler.start()
    try:
        for schedule in timing.steady_state_schedule:
            row: dict[str, Any] = {
                "sample_index": schedule.sample_index,
                "backend_order": [backend.value for backend in schedule.backend_order],
                "logical_training_step_ns": {},
            }
            for backend in schedule.backend_order:
                range_name = f"contract_map.steady.{schedule.sample_index}.{backend.value}"
                with _NvtxRange(range_name, args.nvcc):
                    started = time.perf_counter_ns()
                    output = None
                    for _ in range(timing.iterations_per_sample):
                        output = executables[backend.value](*context.inputs)
                    jax.block_until_ready(output)
                    elapsed = time.perf_counter_ns() - started
                row["logical_training_step_ns"][backend.value] = elapsed // timing.iterations_per_sample
            raw_samples.append(row)
    finally:
        profiler.stop()

    final_hlo = {backend: executable.as_text() for backend, executable in executables.items()}
    return {
        "case_id": args.case_id,
        "final_hlo": final_hlo,
        "numerical": numerical,
        "warmup_samples_ns": warmups,
        "raw_samples": raw_samples,
    }


class _NvtxRange:
    def __init__(self, name: str, nvcc: Path):
        from tile_lifetime.cuda_toolchain import cuda_toolkit_shared_library  # noqa: PLC0415

        self._name = name
        self._library = ctypes.CDLL(str(cuda_toolkit_shared_library(nvcc, "nvToolsExt")))
        self._library.nvtxRangePushA.argtypes = (ctypes.c_char_p,)
        self._library.nvtxRangePushA.restype = ctypes.c_int
        self._library.nvtxRangePop.argtypes = ()
        self._library.nvtxRangePop.restype = ctypes.c_int

    def __enter__(self) -> None:
        if self._library.nvtxRangePushA(self._name.encode()) < 0:
            raise RuntimeError(f"NVTX rejected range {self._name!r}")

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._library.nvtxRangePop() < 0:
            raise RuntimeError(f"NVTX failed to close range {self._name!r}")


class _CudaProfiler:
    def __init__(self, nvcc: Path):
        from tile_lifetime.cuda_toolchain import cuda_toolkit_shared_library  # noqa: PLC0415

        self._library = ctypes.CDLL(str(cuda_toolkit_shared_library(nvcc, "cudart")))
        self._library.cudaProfilerStart.restype = ctypes.c_int
        self._library.cudaProfilerStop.restype = ctypes.c_int

    def start(self) -> None:
        if self._library.cudaProfilerStart() != 0:
            raise RuntimeError("cudaProfilerStart failed")

    def stop(self) -> None:
        if self._library.cudaProfilerStop() != 0:
            raise RuntimeError("cudaProfilerStop failed")


def _numerical_evidence(context: _WorkerCaseContext, executables: Mapping[str, Any], *, jax: Any) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415

    from tile_lifetime.contract_map_backend import (  # noqa: PLC0415
        execute_contract_map_source_ordered_forward,
        execute_contract_map_source_ordered_reverse,
    )
    from tile_lifetime.h100_contract_map_benchmark import (  # noqa: PLC0415
        REVIEWED_NUMERICAL_FLOORS_SHA256,
        BackendVariant,
        validate_backend_numerical_evidence,
    )

    host_inputs = tuple(np.asarray(value, dtype=np.float32) for value in context.inputs)
    activation, first_weight, second_weight, cotangent = host_inputs
    real_reference = _real_algebra_reference(
        context.case.scalar_map.value,
        activation,
        first_weight,
        second_weight,
        cotangent,
    )
    source_candidate = context.candidates[BackendVariant.SHUTTLE_SOURCE_ORDERED.value]
    source_forward = execute_contract_map_source_ordered_forward(
        source_candidate.program,
        activation,
        first_weight,
        second_weight,
    )
    source_reverse = execute_contract_map_source_ordered_reverse(
        source_candidate.program,
        activation,
        first_weight,
        second_weight,
        source_forward,
        cotangent,
    )
    source_reference = (
        source_forward.output,
        source_reverse.input_adjoint,
        source_reverse.first_weight_adjoint,
        source_reverse.second_weight_adjoint,
    )

    evidence: dict[str, Any] = {}
    for backend in BackendVariant:
        repeats = []
        for _ in range(3):
            outputs = executables[backend.value](*context.inputs)
            jax.block_until_ready(outputs)
            repeats.append(tuple(np.asarray(output) for output in outputs))
        reference = source_reference if backend is BackendVariant.SHUTTLE_SOURCE_ORDERED else real_reference
        output_evidence = {
            name: _output_numerical_evidence(index, repeats, reference[index])
            for index, name in enumerate(_OUTPUT_NAMES)
        }
        validate_backend_numerical_evidence(backend, output_evidence)
        evidence[backend.value] = {
            "reviewed_floors_sha256": REVIEWED_NUMERICAL_FLOORS_SHA256,
            "floors_passed_before_timing": True,
            "outputs": output_evidence,
        }
    return evidence


def _real_algebra_reference(
    scalar_map: str,
    activation: Any,
    first_weight: Any,
    second_weight: Any,
    cotangent: Any,
) -> tuple[Any, ...]:
    import numpy as np  # noqa: PLC0415

    x = np.asarray(activation, dtype=np.float64)
    w0 = np.asarray(first_weight, dtype=np.float64)
    w1 = np.asarray(second_weight, dtype=np.float64)
    dy = np.asarray(cotangent, dtype=np.float64)
    z = x @ w0
    if scalar_map == "sigmoid_product":
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        hidden = z * sigmoid
        derivative = sigmoid + z * sigmoid * (1.0 - sigmoid)
    elif scalar_map == "tanh_product":
        tanh = np.tanh(z)
        hidden = z * tanh
        derivative = tanh + z * (1.0 - tanh * tanh)
    elif scalar_map == "cubic_mix":
        hidden = z + z * z * z
        derivative = 1.0 + 3.0 * z * z
    else:
        raise ValueError(f"unsupported scalar Map for real-algebra reference: {scalar_map!r}")
    output = hidden @ w1
    hidden_adjoint = dy @ w1.T
    preactivation_adjoint = hidden_adjoint * derivative
    return (
        output,
        preactivation_adjoint @ w0.T,
        x.T @ preactivation_adjoint,
        hidden.T @ dy,
    )


def _output_numerical_evidence(index: int, repeats: Sequence[Sequence[Any]], reference: Any) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415

    actual_repeats = tuple(np.asarray(outputs[index]) for outputs in repeats)
    expected = np.asarray(reference)
    first = actual_repeats[0]
    difference = np.abs(first.astype(np.float32) - expected.astype(np.float32))
    nonfinite = int(np.count_nonzero(~np.isfinite(first)) + np.count_nonzero(~np.isfinite(difference)))
    ulp = _bfloat16_ulp_distance(first, expected)
    pairwise = []
    for left, right in itertools.combinations(range(len(actual_repeats)), 2):
        drift = np.abs(actual_repeats[left].astype(np.float32) - actual_repeats[right].astype(np.float32))
        drift_ulp = _bfloat16_ulp_distance(actual_repeats[left], actual_repeats[right])
        pairwise.append(
            {
                "left_repeat_index": left,
                "right_repeat_index": right,
                "maximum_absolute_error": float(drift.max(initial=0.0)),
                "mean_absolute_error": float(drift.mean()),
                "maximum_ulp_distance": int(drift_ulp.max(initial=0)),
                "mean_ulp_distance": float(drift_ulp.mean()),
            }
        )
    return {
        "maximum_absolute_error": float(difference.max(initial=0.0)),
        "mean_absolute_error": float(difference.mean()),
        "maximum_ulp_distance": int(ulp.max(initial=0)),
        "mean_ulp_distance": float(ulp.mean()),
        "nonfinite_values": nonfinite,
        "repeat_hashes": [hashlib.sha256(value.tobytes(order="C")).hexdigest() for value in actual_repeats],
        "pairwise_drift": pairwise,
    }


def _bfloat16_ulp_distance(left: Any, right: Any) -> Any:
    import numpy as np  # noqa: PLC0415

    left_array = np.asarray(left)
    right_array = np.asarray(right, dtype=left_array.dtype)
    left_bits = left_array.view(np.uint16).astype(np.int32)
    right_bits = right_array.view(np.uint16).astype(np.int32)
    left_ordered = np.where(left_bits & 0x8000, 0x8000 - (left_bits & 0x7FFF), 0x8000 + left_bits)
    right_ordered = np.where(right_bits & 0x8000, 0x8000 - (right_bits & 0x7FFF), 0x8000 + right_bits)
    return np.abs(left_ordered - right_ordered)


def derive_ordinary_xla_executable_evidence(
    final_hlo: str,
    *,
    rows: int,
    reduction: int,
    features: int,
    profiled_launches: Sequence[str],
) -> OrdinaryXlaExecutableEvidence:
    """Derive the ordinary forward-plus-VJP boundary from final optimized HLO."""
    from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text  # noqa: PLC0415

    module = parse_hlo_module_text(final_hlo)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    parameter_roles = ("x", "w0", "w1", "do")
    output_roles = ("y", "dx", "dw0", "dw1")
    expected_inputs = (
        f"bf16[{rows},{reduction}]{{1,0}}",
        f"bf16[{reduction},{features}]{{1,0}}",
        f"bf16[{features},{reduction}]{{1,0}}",
        f"bf16[{rows},{reduction}]{{1,0}}",
    )
    expected_outputs = (
        f"bf16[{rows},{reduction}]{{1,0}}",
        f"bf16[{rows},{reduction}]{{1,0}}",
        f"bf16[{reduction},{features}]{{1,0}}",
        f"bf16[{features},{reduction}]{{1,0}}",
    )
    parameters: dict[int, Any] = {}
    for instruction in entry.instructions:
        if instruction.opcode != "parameter":
            continue
        match = re.search(r"parameter\((?P<number>[0-9]+)\)", instruction.attributes)
        if match is None:
            raise ValueError(f"ordinary-XLA parameter %{instruction.name} has no number")
        number = int(match.group("number"))
        if number in parameters:
            raise ValueError(f"ordinary-XLA entry repeats parameter({number})")
        parameters[number] = instruction
    if tuple(sorted(parameters)) != tuple(range(4)):
        raise ValueError("ordinary-XLA entry must expose exactly x, w0, w1, and do as parameter(0..3)")
    actual_inputs = tuple(parameters[index].shape for index in range(4))
    if actual_inputs != expected_inputs:
        raise ValueError(f"ordinary-XLA parameter layouts changed: {actual_inputs}")

    root = entry.root
    if root.opcode != "tuple" or len(root.operands) != 4:
        raise ValueError("ordinary-XLA root must be the exact y, dx, dw0, dw1 tuple")
    actual_outputs = tuple(instructions[operand].shape for operand in root.operands)
    if actual_outputs != expected_outputs:
        raise ValueError(f"ordinary-XLA root layouts changed: {actual_outputs}")
    normalized_root_shape = "".join(root.shape.split())
    expected_root_shape = "(" + ",".join(expected_outputs) + ")"
    if normalized_root_shape != expected_root_shape:
        raise ValueError("ordinary-XLA tuple result layouts disagree with root operands")

    reachable = {root.name}
    pending = list(root.operands)
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        if name not in instructions:
            raise ValueError(f"ordinary-XLA entry references missing instruction %{name}")
        reachable.add(name)
        pending.extend(instructions[name].operands)

    entry_adapters = tuple(
        instruction
        for instruction in entry.instructions
        if instruction.name in reachable and instruction.opcode in {"copy", "transpose", "bitcast"}
    )
    if entry_adapters:
        raise ValueError(
            "ordinary-XLA entry adapters require materialization evidence before logical-boundary publication: "
            f"{tuple((instruction.name, instruction.opcode) for instruction in entry_adapters)}"
        )
    normalized_launches = tuple(normalize_cuda_kernel_name(name) for name in profiled_launches)
    if not normalized_launches:
        raise ValueError("ordinary-XLA executable evidence requires profiler-owned launches")

    input_layouts = list(actual_inputs)
    output_layouts = list(actual_outputs)
    boundary = {
        "input_layouts": input_layouts,
        "output_layouts": output_layouts,
        "layout_adapters": [],
        "materialized_copies": [],
        "transposes": [],
        "bitcasts": [],
        "saved_state_names_and_bytes": {},
        "recompute_operations": [],
    }
    fusions = []
    custom_calls = []
    computation_names = {computation.name for computation in module.computations}
    for instruction in entry.instructions:
        if instruction.name not in reachable:
            continue
        if instruction.opcode == "fusion":
            calls = re.findall(r"(?:calls|to_apply)=%?([A-Za-z0-9_.-]+)", instruction.attributes)
            kinds = re.findall(r"(?:^|,\s*)kind=([A-Za-z0-9_.-]+)", instruction.attributes)
            if len(calls) != 1 or calls[0] not in computation_names or len(kinds) != 1:
                raise ValueError(f"ordinary-XLA fusion %{instruction.name} has ambiguous call or kind facts")
            fusions.append(
                {
                    "name": instruction.name,
                    "shape": instruction.shape,
                    "operands": list(instruction.operands),
                    "called_computation": calls[0],
                    "kind": kinds[0],
                }
            )
        elif instruction.opcode == "custom-call":
            targets = re.findall(r'(?:^|,\s*)custom_call_target="([^"]+)"', instruction.attributes)
            if len(targets) != 1:
                raise ValueError(f"ordinary-XLA custom call %{instruction.name} has ambiguous target facts")
            custom_calls.append(
                {
                    "name": instruction.name,
                    "shape": instruction.shape,
                    "operands": list(instruction.operands),
                    "target": targets[0],
                    "side_effect": "unproven_by_hlo",
                }
            )
    manifest = {
        "executable": "ordinary_xla",
        "hlo_sha256": hashlib.sha256(final_hlo.encode()).hexdigest(),
        "logical_inputs": [
            {"role": role, "parameter_number": index, "shape_layout": shape}
            for index, (role, shape) in enumerate(zip(parameter_roles, actual_inputs, strict=True))
        ],
        "logical_outputs": [
            {"role": role, "root_operand": operand, "shape_layout": shape}
            for role, operand, shape in zip(output_roles, root.operands, actual_outputs, strict=True)
        ],
        "entry_copies": [],
        "entry_transposes": [],
        "entry_bitcasts": [],
        "fusions": fusions,
        "custom_calls": custom_calls,
        "observed_launch_facts": {
            "source": "nsys+ncu",
            "launch_count": len(normalized_launches),
            "ordered_kernel_names": list(normalized_launches),
            "hlo_to_launch_mapping": "not_claimed",
        },
        "saved_state_status": "no_cross_entry_state",
        "recompute_status": "not_proven",
        "boundary_relationship": "same_executable_no_entry_adapters",
    }
    return OrdinaryXlaExecutableEvidence(
        kernel_only_boundary=dict(boundary),
        logical_training_step_boundary=dict(boundary),
        manifest=manifest,
    )


def run_coordinator(config: RunnerConfig) -> Path:
    """Execute all reviewed phases and publish the accepted bundle last."""
    if os.environ.get("XLA_FLAGS"):
        raise ValueError("coordinator requires XLA_FLAGS to be unset; workers use an exact closed flag set")
    preflight = require_clean_h100_preflight(config)

    # Every import below may load JAX transitively and must remain after the
    # package-independent source, tool, and device preflight.
    from tile_lifetime.contract_map_backend_resources import (  # noqa: PLC0415
        expected_contract_map_logical_boundary,
    )
    from tile_lifetime.h100_contract_map_benchmark import (  # noqa: PLC0415
        ArchitectureStatus,
        BackendVariant,
        ExternalComparator,
        MeasurementBoundary,
        comparator_decision,
        default_h100_contract_map_benchmark_plan,
    )

    training = importlib.import_module("lib.tile_lifetime.benchmarks.h100_contract_map_backend_training")
    config.artifact_directory.mkdir(parents=True)
    (config.artifact_directory / "preflight.json").write_text(
        json.dumps(asdict(preflight), indent=2, sort_keys=True) + "\n"
    )
    generated_artifacts = compile_generated_candidates(config)
    generated_manifest = config.artifact_directory / "generated_manifest.json"
    generated_manifest.write_text(json.dumps([asdict(record) for record in generated_artifacts], indent=2) + "\n")
    artifacts_by_identity = {(record.case_id, record.backend): record for record in generated_artifacts}
    candidates = training.generated_contract_map_candidates()
    candidates_by_identity = {(candidate.case.case_id, candidate.backend.value): candidate for candidate in candidates}
    plan = default_h100_contract_map_benchmark_plan()
    payloads = []
    case_manifests = []
    for case in plan.cases:
        case_directory = config.artifact_directory / "cases" / case.case_id
        case_directory.mkdir(parents=True)
        case_result, trace_records = _run_profiled_case(
            config,
            case.case_id,
            generated_manifest,
            case_directory,
        )
        compile_records = {
            backend.value: _run_cache_protocol(
                config,
                case.case_id,
                backend.value,
                generated_manifest,
                case_directory / "cache" / backend.value,
            )
            for backend in BackendVariant
        }
        ncu_records = {
            backend.value: _run_ncu_profile(
                config,
                case.case_id,
                backend.value,
                generated_manifest,
                case_directory / "ncu" / backend.value,
            )
            for backend in BackendVariant
        }
        final_hlo_by_backend = {
            backend.value: validated_executable_hlo(
                backend.value,
                case_worker_hlo=case_result["final_hlo"][backend.value],
                cache_protocol=compile_records[backend.value],
                profile_worker_hlo=ncu_records[backend.value].final_hlo,
            )
            for backend in BackendVariant
        }
        raw_samples, trace_summary = merge_trace_timing(plan, case_result, trace_records)
        retained = _retain_backend_artifacts(
            case,
            candidates_by_identity,
            artifacts_by_identity,
            ncu_records,
            trace_summary,
            final_hlo_by_backend,
            case_directory,
            case_directory / "cache" / BackendVariant.ORDINARY_XLA.value / "compile_dump_0",
        )
        case_payloads = []
        for backend in BackendVariant:
            candidate = candidates_by_identity.get((case.case_id, backend.value))
            compiled = compile_records[backend.value]
            backend_artifacts = retained[backend.value]
            timing = {
                "compile_samples_ns": [record["compile_ns"] for record in compiled["compile"]],
                "first_execution_samples_ns": [record["first_execution_ns"] for record in compiled["compile"]],
                "warmup_iterations": plan.timing.warmup_iterations,
                "warmup_samples_ns": case_result["warmup_samples_ns"][backend.value],
                "persistent_cache_cold_samples_ns": [record["compile_ns"] for record in compiled["cold"]],
                "persistent_cache_hit_samples_ns": [record["compile_ns"] for record in compiled["hit"]],
                "steady_state_schedule": _serialized_schedule(plan),
                "raw_samples": raw_samples,
            }
            for boundary in MeasurementBoundary:
                if candidate is None:
                    logical = backend_artifacts["logical_boundaries"][boundary.value]
                else:
                    logical = expected_contract_map_logical_boundary(
                        candidate.generated,
                        kernel_only=boundary is MeasurementBoundary.KERNEL_ONLY,
                    ).to_evidence()
                payload = {
                    "identity": {
                        "case_id": case.case_id,
                        "backend": backend.value,
                        "measurement_boundary": boundary.value,
                    },
                    "artifacts": backend_artifacts["artifacts"],
                    "resources": backend_artifacts["resources"],
                    "copies": trace_summary[backend.value]["copies"],
                    "logical_boundary": logical,
                    "provenance": {
                        "command": [sys.executable, *sys.argv],
                        "environment": {
                            "architecture": _ARCHITECTURE,
                            "compute_capability": preflight.compute_capability,
                            "gpu_name": preflight.gpu_name,
                            "jax_version": config.require_jax_version,
                        },
                        "compiler_flags": backend_artifacts["compiler_flags"],
                        "source_sha": config.source_sha,
                        "persistent_cache_identity": compiled["persistent_cache_identity"],
                    },
                    "numerical": case_result["numerical"][backend.value],
                    "timing": timing,
                }
                case_payloads.append(payload)
        payloads.extend(case_payloads)
        case_manifests.append({"case_id": case.case_id, "trace": trace_summary})

    accepted = tuple(payloads)
    decisions = tuple(comparator_decision(comparator, plan.features) for comparator in ExternalComparator)
    if any(decision.admitted for decision in decisions):
        raise AssertionError("dense Contract/Map execution must not admit FA4 or Grug comparators")
    bundle = {
        "schema": "shuttle.h100_contract_map_executed_bundle.v2",
        "architecture_status": ArchitectureStatus.NONCONFORMING.value,
        "source_sha": config.source_sha,
        "preflight": asdict(preflight),
        "external_comparators": [asdict(decision) for decision in decisions],
        "case_manifests": case_manifests,
        "records": accepted,
    }
    return publish_validated_bundle(config.artifact_directory / "accepted_bundle.json", bundle)


def publish_validated_bundle(output: Path, bundle: Mapping[str, Any]) -> Path:
    """Atomically publish only a complete bundle accepted by the reviewed schema."""
    from tile_lifetime.h100_contract_map_benchmark import (  # noqa: PLC0415
        ArchitectureStatus,
        validate_result_evidence_bundle,
    )

    if bundle.get("architecture_status") != ArchitectureStatus.NONCONFORMING.value:
        raise ValueError("executed Contract/Map evidence must remain architecture-nonconforming")
    records = bundle.get("records")
    if not isinstance(records, tuple):
        raise ValueError("executed evidence records must be an immutable tuple before publication")
    validate_result_evidence_bundle(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return output


def _serialized_schedule(plan: Any) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": row.sample_index,
            "cycle_index": row.cycle_index,
            "backend_order": [backend.value for backend in row.backend_order],
        }
        for row in plan.timing.steady_state_schedule
    ]


def _worker_base_command(
    config: RunnerConfig,
    *,
    worker: WorkerMode,
    case_id: str,
    backend: str,
    generated_manifest: Path,
    json_output: Path,
    cache_kind: str = "none",
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        worker.value,
        "--case-id",
        case_id,
        "--backend",
        backend,
        "--generated-manifest",
        str(generated_manifest),
        "--json-output",
        str(json_output),
        "--nvcc",
        str(config.tools.nvcc),
        "--require-jax-version",
        config.require_jax_version,
        "--cache-kind",
        cache_kind,
    )


def _worker_environment(dump_directory: Path, cache_directory: Path | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    flags = (
        f"--xla_dump_to={dump_directory.resolve()} "
        "--xla_dump_hlo_as_text=true --xla_dump_hlo_as_proto=true --xla_gpu_dump_llvmir=true"
    )
    environment["XLA_FLAGS"] = flags
    environment.pop("JAX_COMPILATION_CACHE_DIR", None)
    for name in _CACHE_ENVIRONMENT:
        environment.pop(name, None)
    if cache_directory is not None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        environment["JAX_COMPILATION_CACHE_DIR"] = str(cache_directory.resolve())
        environment.update(_CACHE_ENVIRONMENT)
    return environment


def _run_worker_command(
    command: tuple[str, ...],
    *,
    environment: Mapping[str, str],
    json_output: Path,
) -> dict[str, Any]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed with {completed.returncode}: {command}: {completed.stdout}\n{completed.stderr}"
        )
    if not json_output.is_file():
        raise RuntimeError(f"worker succeeded without structured output: {json_output}")
    return json.loads(json_output.read_text())


def run_timed_compile_worker_command(
    command: tuple[str, ...],
    *,
    environment: Mapping[str, str],
    json_output: Path,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Measure process spawn through compile-worker result publication."""
    started = now()
    completed = run(command, check=False, capture_output=True, text=True, env=environment)
    elapsed = now() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"compile worker failed with {completed.returncode}: {command}: {completed.stdout}\n{completed.stderr}"
        )
    if not json_output.is_file():
        raise RuntimeError(f"compile worker succeeded without structured output: {json_output}")
    result = json.loads(json_output.read_text())
    if elapsed <= 0:
        raise RuntimeError("compile worker elapsed time must be positive")
    result["compile_ns"] = elapsed
    return result


def validated_cache_protocol_identity(
    compile_records: Sequence[Mapping[str, Any]],
    cold_records: Sequence[Mapping[str, Any]],
    hit_records: Sequence[Mapping[str, Any]],
    *,
    required_processes: int,
) -> str:
    """Require every declared isolated root to converge to one cache identity."""
    groups = {"compile": compile_records, "cold": cold_records, "hit": hit_records}
    for name, records in groups.items():
        if len(records) != required_processes:
            raise ValueError(f"cache protocol requires {required_processes} {name} roots")
    identities = {str(record["persistent_cache_identity"]) for records in groups.values() for record in records}
    if (
        len(identities) != 1
        or len(next(iter(identities))) != 64
        or any(character not in "0123456789abcdef" for character in next(iter(identities)))
    ):
        raise ValueError("all compile, cold, and hit roots must converge to one cache content identity")
    return next(iter(identities))


def validated_executable_hlo(
    backend: str,
    *,
    case_worker_hlo: str,
    cache_protocol: Mapping[str, Any],
    profile_worker_hlo: str,
) -> str:
    """Bind timed, profiled, and cache-worker evidence to one exact executable HLO."""
    compile_records = tuple(cache_protocol["compile"])
    if not compile_records:
        raise ValueError(f"{backend} executable evidence has no compile worker")
    authoritative = str(compile_records[0]["final_hlo"])
    observed = [case_worker_hlo, profile_worker_hlo]
    for group in ("compile", "cold", "hit"):
        observed.extend(str(record["final_hlo"]) for record in cache_protocol[group])
    if not authoritative.strip() or any(value != authoritative for value in observed):
        raise ValueError(f"{backend} final HLO differs across compile, cache, timing, or profile workers")
    return authoritative


def _run_profiled_case(
    config: RunnerConfig,
    case_id: str,
    generated_manifest: Path,
    directory: Path,
) -> tuple[dict[str, Any], tuple[TraceRange, ...]]:
    result_path = directory / "case_result.json"
    worker = _worker_base_command(
        config,
        worker=WorkerMode.CASE,
        case_id=case_id,
        backend="all",
        generated_manifest=generated_manifest,
        json_output=result_path,
    )
    report_base = directory / "steady_trace"
    command = (
        str(config.tools.nsys),
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx",
        "--capture-range=cudaProfilerApi",
        "--stop-on-range-end=true",
        "--output",
        str(report_base),
        *worker,
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_worker_environment(directory / "xla_dump"),
    )
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"Nsight Systems case worker failed: {command}: {completed.stdout}\n{completed.stderr}")
    report = report_base.with_suffix(".nsys-rep")
    sqlite_path = directory / "steady_trace.sqlite"
    export = (
        str(config.tools.nsys),
        "export",
        "--force-overwrite=true",
        "--type=sqlite",
        "--output",
        str(sqlite_path),
        str(report),
    )
    _run_retained(export)
    result = json.loads(result_path.read_text())
    expected_ranges = tuple(
        f"contract_map.steady.{row['sample_index']}.{backend}"
        for row in result["raw_samples"]
        for backend in row["backend_order"]
    )
    return result, parse_nsys_sqlite(sqlite_path, expected_ranges)


def _run_cache_protocol(
    config: RunnerConfig,
    case_id: str,
    backend: str,
    generated_manifest: Path,
    directory: Path,
) -> dict[str, Any]:
    from tile_lifetime.h100_contract_map_benchmark import default_h100_contract_map_benchmark_plan  # noqa: PLC0415

    protocol = default_h100_contract_map_benchmark_plan().timing
    directory.mkdir(parents=True)
    compile_records = []
    for index in range(protocol.compile_processes):
        root = directory / "compile_roots" / str(index)
        result = directory / f"compile_{index}.json"
        command = _worker_base_command(
            config,
            worker=WorkerMode.COMPILE,
            case_id=case_id,
            backend=backend,
            generated_manifest=generated_manifest,
            json_output=result,
            cache_kind="compile",
        )
        compile_records.append(
            run_timed_compile_worker_command(
                command,
                environment=_worker_environment(directory / f"compile_dump_{index}", root),
                json_output=result,
            )
        )
    cold_records = []
    hit_records = []
    for index in range(protocol.persistent_cache_cold_processes):
        root = directory / "paired_roots" / str(index)
        cold_result = directory / f"cold_{index}.json"
        cold_command = _worker_base_command(
            config,
            worker=WorkerMode.COMPILE,
            case_id=case_id,
            backend=backend,
            generated_manifest=generated_manifest,
            json_output=cold_result,
            cache_kind="cold",
        )
        cold = run_timed_compile_worker_command(
            cold_command,
            environment=_worker_environment(directory / f"cold_dump_{index}", root),
            json_output=cold_result,
        )
        hit_result = directory / f"hit_{index}.json"
        hit_command = _worker_base_command(
            config,
            worker=WorkerMode.COMPILE,
            case_id=case_id,
            backend=backend,
            generated_manifest=generated_manifest,
            json_output=hit_result,
            cache_kind="hit",
        )
        hit = run_timed_compile_worker_command(
            hit_command,
            environment=_worker_environment(directory / f"hit_dump_{index}", root),
            json_output=hit_result,
        )
        if cold["persistent_cache_identity"] != hit["persistent_cache_identity"]:
            raise ValueError(f"persistent cache content identity changed between cold and hit for {case_id}/{backend}")
        cold_records.append(cold)
        hit_records.append(hit)
    identity = validated_cache_protocol_identity(
        compile_records,
        cold_records,
        hit_records,
        required_processes=protocol.compile_processes,
    )
    return {
        "compile": compile_records,
        "cold": cold_records,
        "hit": hit_records,
        "persistent_cache_identity": identity,
    }


def _run_ncu_profile(
    config: RunnerConfig,
    case_id: str,
    backend: str,
    generated_manifest: Path,
    directory: Path,
) -> NcuProfileEvidence:
    directory.mkdir(parents=True)
    result = directory / "profile_worker.json"
    csv_path = directory / "ncu.csv"
    report_path = directory / "profile.ncu-rep"
    sass_source_path = directory / "ncu_sass_source.txt"
    worker = _worker_base_command(
        config,
        worker=WorkerMode.PROFILE,
        case_id=case_id,
        backend=backend,
        generated_manifest=generated_manifest,
        json_output=result,
    )
    command = (
        str(config.tools.ncu),
        "--force-overwrite",
        "--target-processes=all",
        "--nvtx",
        "--nvtx-include=contract_map.profile/",
        "--metrics",
        ",".join(_NCU_METRICS),
        "--csv",
        "--page=raw",
        "--log-file",
        str(csv_path),
        "--export",
        str(report_path),
        *worker,
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_worker_environment(directory / "xla_dump"),
    )
    if completed.returncode != 0 or not result.is_file() or not csv_path.is_file() or not report_path.is_file():
        raise RuntimeError(f"Nsight Compute worker failed: {command}: {completed.stdout}\n{completed.stderr}")
    source_export = (
        str(config.tools.ncu),
        "--import",
        str(report_path),
        "--page",
        "source",
        "--print-source",
        "sass",
        "--log-file",
        str(sass_source_path),
    )
    _run_retained(source_export)
    if not sass_source_path.is_file() or not sass_source_path.read_text().strip():
        raise RuntimeError("Nsight Compute produced no public SASS/source export")
    worker_result = json.loads(result.read_text())
    return NcuProfileEvidence(
        metrics=parse_ncu_metrics(csv_path),
        report_path=str(report_path),
        report_sha256=file_sha256(report_path),
        sass_source_path=str(sass_source_path),
        sass_source_sha256=file_sha256(sass_source_path),
        final_hlo=str(worker_result["final_hlo"]),
    )


def merge_trace_timing(plan: Any, case_result: Mapping[str, Any], traces: tuple[TraceRange, ...]) -> tuple[Any, Any]:
    from tile_lifetime.h100_contract_map_benchmark import BackendVariant, MeasurementBoundary  # noqa: PLC0415

    trace_by_name = {trace.name: trace for trace in traces}
    sequence_by_backend: dict[str, tuple[str, ...]] = {}
    copies_by_backend = {
        backend.value: {
            "device_to_device_count": 0,
            "device_to_device_bytes": 0,
            "host_to_device_count": 0,
            "host_to_device_bytes": 0,
            "unexpected_copy_count": 0,
        }
        for backend in BackendVariant
    }
    raw_samples = []
    for schedule, worker_row in zip(plan.timing.steady_state_schedule, case_result["raw_samples"], strict=True):
        if worker_row["sample_index"] != schedule.sample_index or worker_row["backend_order"] != [
            backend.value for backend in schedule.backend_order
        ]:
            raise ValueError("case worker timing rows do not match the reviewed schedule")
        measurements = {
            backend.value: {
                MeasurementBoundary.KERNEL_ONLY.value: 0,
                MeasurementBoundary.LOGICAL_TRAINING_STEP.value: int(
                    worker_row["logical_training_step_ns"][backend.value]
                ),
            }
            for backend in BackendVariant
        }
        for backend in schedule.backend_order:
            range_name = f"contract_map.steady.{schedule.sample_index}.{backend.value}"
            trace = trace_by_name[range_name]
            iterations = plan.timing.iterations_per_sample
            if len(trace.ordered_kernel_names) % iterations:
                raise ValueError(f"trace range {range_name!r} cannot be partitioned into {iterations} executions")
            launches = len(trace.ordered_kernel_names) // iterations
            chunks = tuple(
                trace.ordered_kernel_names[index * launches : (index + 1) * launches] for index in range(iterations)
            )
            if not chunks or any(chunk != chunks[0] for chunk in chunks[1:]):
                raise ValueError(f"trace range {range_name!r} changed kernel order between iterations")
            previous = sequence_by_backend.setdefault(backend.value, chunks[0])
            if previous != chunks[0]:
                raise ValueError(f"backend {backend.value!r} changed launch topology across scheduled rows")
            if trace.kernel_duration_ns <= 0:
                raise ValueError(f"trace range {range_name!r} lacks positive kernel timing")
            measurements[backend.value][MeasurementBoundary.KERNEL_ONLY.value] = max(
                1,
                round(trace.kernel_duration_ns / iterations),
            )
            copies = copies_by_backend[backend.value]
            copies["device_to_device_count"] += trace.device_to_device_count
            copies["device_to_device_bytes"] += trace.device_to_device_bytes
            copies["host_to_device_count"] += trace.host_to_device_count
            copies["host_to_device_bytes"] += trace.host_to_device_bytes
            copies["unexpected_copy_count"] += (
                trace.unexpected_copy_count + trace.device_to_device_count + trace.host_to_device_count
            )
        raw_samples.append(
            {
                "sample_index": schedule.sample_index,
                "backend_order": [backend.value for backend in schedule.backend_order],
                "measurements_ns": measurements,
            }
        )
    for backend, copies in copies_by_backend.items():
        if copies["unexpected_copy_count"]:
            raise ValueError(f"steady-state CUDA trace found unexpected copies for {backend}: {copies}")
    return raw_samples, {
        backend.value: {
            "ordered_kernel_names": list(sequence_by_backend[backend.value]),
            "launch_count": len(sequence_by_backend[backend.value]),
            "copies": copies_by_backend[backend.value],
        }
        for backend in BackendVariant
    }


def _retain_backend_artifacts(
    case: Any,
    candidates: Mapping[tuple[str, str], Any],
    generated_artifacts: Mapping[tuple[str, str], GeneratedArtifact],
    ncu_records: Mapping[str, NcuProfileEvidence],
    trace_summary: Mapping[str, Mapping[str, Any]],
    final_hlo_by_backend: Mapping[str, str],
    directory: Path,
    ordinary_dump_directory: Path,
) -> dict[str, Any]:
    from tile_lifetime.command_buffer_capture import derive_capture_site_manifest  # noqa: PLC0415
    from tile_lifetime.h100_contract_map_benchmark import BackendVariant  # noqa: PLC0415

    retained: dict[str, Any] = {}
    hlo_directory = directory / "final_hlo"
    manifest_directory = directory / "custom_call_manifests"
    hlo_directory.mkdir()
    manifest_directory.mkdir()
    ordinary_cuda = retain_ordinary_xla_cuda_artifacts(
        dump_directory=ordinary_dump_directory,
        retained_directory=directory / "ordinary_xla_cuda",
    )
    ordinary_evidence = derive_ordinary_xla_executable_evidence(
        final_hlo_by_backend[BackendVariant.ORDINARY_XLA.value],
        rows=case.rows,
        reduction=case.reduction,
        features=case.features,
        profiled_launches=trace_summary[BackendVariant.ORDINARY_XLA.value]["ordered_kernel_names"],
    )
    for backend in BackendVariant:
        hlo_path = hlo_directory / f"{backend.value}.txt"
        hlo_path.write_text(final_hlo_by_backend[backend.value])
        if backend is BackendVariant.ORDINARY_XLA:
            capture_manifest = ordinary_evidence.manifest
            profile = ncu_records[backend.value]
            artifact = ordinary_cuda
            compiler_flags = [
                "jax.jit(...).lower(...).compile()",
                "--xla_dump_hlo_as_text=true",
                "--xla_dump_hlo_as_proto=true",
                "--xla_gpu_dump_llvmir=true",
            ]
            kernel_records = ordinary_kernel_records(profile, artifact)
        else:
            candidate = candidates[(case.case_id, backend.value)]
            generated = generated_artifacts[(case.case_id, backend.value)]
            capture_manifest = derive_capture_site_manifest(
                backend.value,
                final_hlo_by_backend[backend.value],
                {
                    candidate.generated.forward_target: candidate.generated.forward_call_count_symbol,
                    candidate.generated.reverse_target: candidate.generated.reverse_call_count_symbol,
                },
                expected_target_occurrences={
                    candidate.generated.forward_target: 1,
                    candidate.generated.reverse_target: 1,
                },
            ).to_json()
            artifact = {
                "source_path": generated.source_path,
                "source_sha256": generated.source_sha256,
                "shared_library_path": generated.shared_library_path,
                "shared_library_sha256": generated.shared_library_sha256,
                "ptx_path": generated.ptx_path,
                "ptx_sha256": generated.ptx_sha256,
                "cubin_path": generated.cubin_path,
                "cubin_sha256": generated.cubin_sha256,
                "cubin_sass_path": generated.cubin_sass_path,
                "cubin_sass_sha256": generated.cubin_sass_sha256,
                "loaded_image_sass_path": generated.loaded_image_sass_path,
                "loaded_image_sass_sha256": generated.loaded_image_sass_sha256,
            }
            compiler_flags = list(generated.compiler_flags)
            kernel_records = generated_kernel_records(candidate, generated, ncu_records[backend.value].metrics)
        traced_names = tuple(trace_summary[backend.value]["ordered_kernel_names"])
        recorded_names = tuple(record["name"] for record in kernel_records)
        if traced_names != recorded_names:
            raise ValueError(f"Nsight Systems/Nsight Compute launch topology disagrees for {backend.value}")
        capture_manifest["cuda_artifacts"] = artifact
        profile = ncu_records[backend.value]
        capture_manifest["profiler_artifacts"] = {
            "ncu_report_path": profile.report_path,
            "ncu_report_sha256": profile.report_sha256,
            "sass_source_path": profile.sass_source_path,
            "sass_source_sha256": profile.sass_source_sha256,
        }
        manifest_path = manifest_directory / f"{backend.value}.json"
        manifest_path.write_text(json.dumps(capture_manifest, indent=2, sort_keys=True) + "\n")
        retained[backend.value] = {
            "artifacts": {
                "final_optimized_hlo_path": str(hlo_path),
                "final_optimized_hlo_sha256": file_sha256(hlo_path),
                "custom_call_manifest_path": str(manifest_path),
                "custom_call_manifest_sha256": file_sha256(manifest_path),
            },
            "resources": {
                "kernel_records": kernel_records,
                "launch_count": len(kernel_records),
                "ordered_kernel_names": [record["name"] for record in kernel_records],
            },
            "compiler_flags": compiler_flags,
        }
        if backend is BackendVariant.ORDINARY_XLA:
            retained[backend.value]["logical_boundaries"] = {
                "kernel_only": ordinary_evidence.kernel_only_boundary,
                "logical_training_step": ordinary_evidence.logical_training_step_boundary,
            }
    return retained


def retain_ordinary_xla_cuda_artifacts(
    *,
    dump_directory: Path,
    retained_directory: Path,
) -> dict[str, Any]:
    from tile_lifetime.h100_contract_map_benchmark import (  # noqa: PLC0415
        CubinAvailability,
        CubinUnavailableReason,
    )

    ptx_files = tuple(path for path in dump_directory.rglob("*.ptx") if path.is_file() and path.stat().st_size)
    cubin_files = tuple(path for path in dump_directory.rglob("*.cubin") if path.is_file() and path.stat().st_size)
    if len(ptx_files) != 1 or len(cubin_files) > 1:
        raise RuntimeError(
            "pinned jaxlib must dump exactly one ordinary-XLA PTX and at most one public cubin; "
            f"found PTX={ptx_files}, cubin={cubin_files}"
        )
    retained_directory.mkdir()
    ptx_path = retained_directory / "ordinary_xla.ptx"
    shutil.copy2(ptx_files[0], ptx_path)
    if cubin_files:
        cubin_path = retained_directory / "ordinary_xla.cubin"
        shutil.copy2(cubin_files[0], cubin_path)
        cubin = {
            "availability": CubinAvailability.AVAILABLE.value,
            "path": str(cubin_path),
            "sha256": file_sha256(cubin_path),
        }
    else:
        cubin = {
            "availability": CubinAvailability.UNAVAILABLE.value,
            "unavailable_reason": CubinUnavailableReason.PUBLIC_XLA_DUMP_OMITS_CUBIN.value,
        }
    return {
        "ptx_path": str(ptx_path),
        "ptx_sha256": file_sha256(ptx_path),
        "cubin": cubin,
    }


def generated_kernel_records(
    candidate: Any,
    artifact: GeneratedArtifact,
    metrics: tuple[NcuKernelMetrics, ...],
) -> list[dict[str, Any]]:
    if cuda_sass_kernel_names(Path(artifact.loaded_image_sass_path).read_text()) != tuple(
        candidate.generated.kernel_names
    ):
        raise ValueError("loaded shared-library SASS kernel identities changed after compilation")
    metric_by_name = _align_generated_metrics(candidate.generated.kernel_names, metrics)
    resources = {record["kernel_name"]: record for record in artifact.ptxas_resources}
    if set(resources) != set(candidate.generated.kernel_names):
        raise ValueError("ptxas resources do not cover every generated kernel")
    records = []
    for name in candidate.generated.kernel_names:
        metric = metric_by_name[name]
        resource = resources[name]
        if resource["registers_per_thread"] != metric.registers_per_thread:
            raise ValueError(f"ptxas/ncu register count disagrees for {name!r}")
        if resource["static_shared_bytes"] != metric.static_shared_memory_bytes:
            raise ValueError(f"ptxas/ncu static shared memory disagrees for {name!r}")
        records.append(
            _kernel_record(
                name,
                artifact.ptx_path,
                artifact.ptx_sha256,
                {
                    "availability": "available",
                    "path": artifact.cubin_path,
                    "sha256": artifact.cubin_sha256,
                },
                artifact.loaded_image_sass_path,
                artifact.loaded_image_sass_sha256,
                metric,
                spill_load_bytes=resource["spill_load_bytes"],
                spill_store_bytes=resource["spill_store_bytes"],
            )
        )
    return records


def _align_generated_metrics(
    expected_names: tuple[str, ...], metrics: tuple[NcuKernelMetrics, ...]
) -> dict[str, NcuKernelMetrics]:
    normalized_metrics: dict[str, list[NcuKernelMetrics]] = {}
    for metric in metrics:
        normalized_metrics.setdefault(normalize_cuda_kernel_name(metric.name), []).append(metric)
    aligned = {}
    for expected in expected_names:
        normalized_expected = normalize_cuda_kernel_name(expected)
        matches = tuple(normalized_metrics.get(normalized_expected, ()))
        if len(matches) != 1:
            raise ValueError(f"Nsight Compute must report generated kernel {expected!r} exactly once")
        aligned[expected] = matches[0]
    if len(metrics) != len(expected_names):
        raise ValueError("Nsight Compute reported unexpected generated kernel launches")
    return aligned


def ordinary_kernel_records(
    profile: NcuProfileEvidence,
    artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sass = Path(profile.sass_source_path).read_text()
    if any(opcode in sass for opcode in (" LDL", " STL")):
        raise RuntimeError("ordinary-XLA profiler SASS contains local-memory spills without byte evidence")
    return [
        _kernel_record(
            normalize_cuda_kernel_name(metric.name),
            artifact["ptx_path"],
            artifact["ptx_sha256"],
            artifact["cubin"],
            profile.sass_source_path,
            profile.sass_source_sha256,
            metric,
            spill_load_bytes=0,
            spill_store_bytes=0,
        )
        for metric in profile.metrics
    ]


def _kernel_record(
    name: str,
    ptx_path: str,
    ptx_sha256: str,
    cubin: Mapping[str, Any],
    sass_path: str,
    sass_sha256: str,
    metric: NcuKernelMetrics,
    *,
    spill_load_bytes: int,
    spill_store_bytes: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "ptx_path": ptx_path,
        "ptx_sha256": ptx_sha256,
        "cubin": dict(cubin),
        "sass_path": sass_path,
        "sass_sha256": sass_sha256,
        "registers_per_thread": metric.registers_per_thread,
        "spill_load_bytes": spill_load_bytes,
        "spill_store_bytes": spill_store_bytes,
        "static_shared_memory_bytes": metric.static_shared_memory_bytes,
        "dynamic_shared_memory_bytes": metric.dynamic_shared_memory_bytes,
        "block_size": list(metric.block_size),
        "active_blocks_per_sm": metric.active_blocks_per_sm,
        "limiting_occupancy_resource": metric.limiting_occupancy_resource,
        "achieved_occupancy": metric.achieved_occupancy,
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse coordinator and internal worker arguments without importing JAX."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run the reviewed H100 evidence protocol.")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--require-jax-version", required=True)
    parser.add_argument("--git", type=Path)
    parser.add_argument("--nvidia-smi", type=Path)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--ptxas", type=Path)
    parser.add_argument("--cuobjdump", type=Path)
    parser.add_argument("--ncu", type=Path)
    parser.add_argument("--nsys", type=Path)
    parser.add_argument("--worker", type=WorkerMode, choices=tuple(WorkerMode))
    parser.add_argument("--case-id")
    parser.add_argument("--backend")
    parser.add_argument("--generated-manifest", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--cache-kind", choices=("none", "compile", "cold", "hit"), default="none")
    return parser.parse_args(argv)


def _required_argument(value: Any, name: str) -> Any:
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _resolved_tool(explicit: Path | None, name: str) -> Path:
    candidate = str(explicit) if explicit is not None else shutil.which(name)
    if candidate is None:
        raise ValueError(f"required tool is not on PATH and has no explicit path: {name}")
    return Path(candidate).resolve()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    if args.worker is not None:
        for name in ("case_id", "backend", "generated_manifest", "json_output"):
            _required_argument(getattr(args, name), f"--{name.replace('_', '-')}")
        args.nvcc = args.nvcc.resolve()
        run_worker(args)
        return
    if not args.execute:
        raise ValueError("coordinator requires --execute; use the staging manifest CLI for plan-only output")
    source_root = _required_argument(args.source_root, "--source-root").resolve()
    artifact_directory = _required_argument(args.artifact_directory, "--artifact-directory").resolve()
    nvcc = args.nvcc.resolve()
    tools = ToolPaths(
        git=_resolved_tool(args.git, "git"),
        nvidia_smi=_resolved_tool(args.nvidia_smi, "nvidia-smi"),
        nvcc=nvcc,
        ptxas=_resolved_tool(args.ptxas or nvcc.with_name("ptxas"), "ptxas"),
        cuobjdump=_resolved_tool(args.cuobjdump or nvcc.with_name("cuobjdump"), "cuobjdump"),
        ncu=_resolved_tool(args.ncu, "ncu"),
        nsys=_resolved_tool(args.nsys, "nsys"),
    )
    output = run_coordinator(
        RunnerConfig(
            source_root=source_root,
            source_sha=_required_argument(args.source_sha, "--source-sha"),
            artifact_directory=artifact_directory,
            tools=tools,
            require_jax_version=args.require_jax_version,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
