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
import stat
import string
import subprocess
import sys
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from tile_lifetime.bfloat16_metrics import bfloat16_ulp_distance
from tile_lifetime.h100_contract_map_benchmark import NumericalFloorError
from tile_lifetime.nvtx_range import NvtxRange

_ARCHITECTURE = "sm_90a"
_COMPUTE_CAPABILITY = "9.0"
_NSYS_TRACE_APIS = "cuda,nvtx"
_NSYS_EXPORT_LAZY = "true"
_NSYS_CUDA_GRAPH_TRACE = "node"
_NSYS_PROFILE_ARGS = (
    f"--trace={_NSYS_TRACE_APIS}",
    f"--cuda-graph-trace={_NSYS_CUDA_GRAPH_TRACE}",
    "--capture-range=cudaProfilerApi",
    "--capture-range-end=stop",
)
_NSYS_EXPORT_ARGS = ("--type=sqlite", f"--lazy={_NSYS_EXPORT_LAZY}")
_MAX_NSYS_NO_KERNEL_DIAGNOSTIC_CHARS = 4096
_MAX_CACHE_IDENTITY_DIAGNOSTIC_CHARS = 4096
_MAX_PERSISTENT_CACHE_ENTRY_BYTES = 256 * 1024 * 1024
_MAX_PERSISTENT_CACHE_FILES = 1024
_MAX_PERSISTENT_CACHE_ROOT_BYTES = 1024 * 1024 * 1024
_MAX_SERIALIZED_EXECUTABLE_BYTES = 1024 * 1024 * 1024
_CACHE_COMPRESSION = "zlib"
_CACHE_FILE_PATTERN = re.compile(r"(?P<key>.+-[0-9a-f]{64})-cache")
_TARGET_CACHE_FILE_PATTERN = re.compile(r"(?P<key>jit_step-[0-9a-f]{64})-cache")
_CACHE_EVENT_NAMES = (
    "/jax/compilation_cache/compile_requests_use_cache",
    "/jax/compilation_cache/cache_hits",
    "/jax/compilation_cache/cache_misses",
)
_CACHE_DIAGNOSTIC_CLASS_FIELDS = (
    "equality_partition",
    "cache_key_digest",
    "serialized_executable_sha256",
)
_CACHE_DIAGNOSTIC_ROOT_FIELDS = (
    "phase",
    "index",
    "equality_partition",
    "persistent_cache_file_count",
    "persistent_cache_total_bytes",
    "final_hlo_sha256",
)
_NSYS_RELEVANT_TABLES = (
    "CUDA_GRAPH_EVENTS",
    "CUDA_GRAPH_NODE_EVENTS",
    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "NVTX_EVENTS",
    "StringIds",
    "TARGET_INFO_GPU",
)
_OUTPUT_NAMES = ("forward", "dx", "dw0", "dw1")
_MAX_NUMERICAL_WORST_PAIR_DIAGNOSTIC_CHARS = 2048
_CACHE_ENVIRONMENT = {
    "JAX_COMPILATION_CACHE_CHECK_CONTENTS": "false",
    "JAX_COMPILATION_CACHE_INCLUDE_METADATA_IN_KEY": "false",
    "JAX_COMPILATION_CACHE_MAX_SIZE": "-1",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
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
_NCU_IDENTITY_FIELDS = (
    "ID",
    "Process ID",
    "Process Name",
    "Host Name",
    "Kernel Name",
    "Context",
    "Stream",
    "Block Size",
    "Grid Size",
    "Device",
    "CC",
)
_NCU_METRIC_UNITS = {
    "launch__block_size": "",
    "launch__registers_per_thread": "register/thread",
    "launch__shared_mem_per_block_static": "byte/block",
    "launch__shared_mem_per_block_dynamic": "byte/block",
    "launch__occupancy_limit_blocks": "block",
    "launch__occupancy_limit_registers": "block",
    "launch__occupancy_limit_shared_mem": "block",
    "launch__occupancy_limit_warps": "block",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "%",
}
_MAX_NCU_CSV_BYTES = 1 << 20
_SASS_OPCODE_BASES = frozenset(
    {
        "ATOM",
        "B2R",
        "BAR",
        "BMOV",
        "BRA",
        "BREAK",
        "BRX",
        "BSYNC",
        "BSSY",
        "CALL",
        "CCTL",
        "CCTLL",
        "CP",
        "CS2R",
        "DADD",
        "DEPBAR",
        "DFMA",
        "DMUL",
        "DSETP",
        "ERRBAR",
        "EXIT",
        "F2F",
        "F2I",
        "FADD",
        "FCHK",
        "FFMA",
        "FLO",
        "FMNMX",
        "FMUL",
        "FSEL",
        "FSET",
        "FSETP",
        "HADD2",
        "HFMA2",
        "HMMA",
        "I2F",
        "IABS",
        "IADD",
        "IADD3",
        "IMAD",
        "IMMA",
        "IMNMX",
        "IMUL",
        "ISETP",
        "LD",
        "LDC",
        "LDG",
        "LDL",
        "LDS",
        "LEA",
        "LOP3",
        "MATCH",
        "MEMBAR",
        "MOV",
        "MUFU",
        "NANOSLEEP",
        "NOP",
        "P2R",
        "PLOP3",
        "POPC",
        "PRMT",
        "QSPC",
        "R2B",
        "RED",
        "RET",
        "S2R",
        "SEL",
        "SHF",
        "SHFL",
        "SHL",
        "SHR",
        "ST",
        "STG",
        "STL",
        "STS",
        "SUATOM",
        "SULD",
        "SURED",
        "SUST",
        "TEX",
        "TLD",
        "UIADD3",
        "UIMAD",
        "UISETP",
        "ULDC",
        "ULEA",
        "ULOP3",
        "ULDP",
        "UMOV",
        "USEL",
        "UTMALDG",
        "UTMASTG",
        "VABSDIFF",
        "VADD",
        "VIMNMX",
        "VSET",
        "WARPSYNC",
        "YIELD",
    }
)
_NCU_SASS_SECTION_PATTERN = re.compile(r"^Kernel Name {8}(?P<name>[A-Za-z_][A-Za-z0-9_]{25}) {62}$")
_NCU_SASS_HEADER = "Address Source"
_NCU_SASS_SEPARATOR = "------------------ " + "-" * 60 + " ------ ------ ------ ------"
_NCU_SASS_INSTRUCTION_PATTERN = re.compile(
    r"^\s*(?:/\*)?(?P<address>(?:0x)?[0-9A-Fa-f]{4,16})(?:\*/)?(?:\s+|\s*:\s*)"
    r"(?:@!?P[0-9]+(?:\.[A-Z0-9_]+)?\s+)?(?P<mnemonic>[A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)\b"
)
_NCU_SASS_FAILURE_PATTERN = re.compile(
    r"(?:^|\b)(?:warning|error)(?::|\b)|(?:source|sass)\s+(?:is\s+)?(?:not|un)available|\bN/A\b",
    flags=re.IGNORECASE,
)
_NCU_SASS_STATUS_PATTERN = re.compile(
    r'^==PROF== (?:(?:Connected to|Disconnected from) process [0-9]+(?: \(.+\))?|Profiling ".+"(?: .*)?)$'
)
_NCU_SASS_PUBLIC_WORD_PATTERN = re.compile(r"\w+")
_NCU_SASS_PUBLIC_WORDS = ("Kernel", "Name", "Address", "Source", "Section", "Function")
_MAX_NCU_SASS_BYTES = 1 << 20
_MAX_NCU_SASS_LINE_CHARS = 1024
_MAX_NCU_SASS_DIAGNOSTIC_BYTES = 2048
_NCU_SASS_DIAGNOSTIC_PREFIX = "unrecognized Nsight Compute SASS export record: "
_CUOBJDUMP_FUNCTION_PATTERN = re.compile(r"^\s*Function\s*:\s*(?P<name>[A-Za-z_.$][A-Za-z0-9_.$]*)\s*$")
_CUOBJDUMP_FUNCTION_PREFIX = re.compile(r"^\s*Function\b")
_CUOBJDUMP_INSTRUCTION_PATTERN = re.compile(
    r"^\s*/\*(?P<address>[0-9A-Fa-f]{4,16})\*/\s+"
    r"(?:(?:@!?[A-Z][A-Z0-9.]*)\s+)?"
    r"(?P<mnemonic>[A-Z][A-Z0-9]*(?:\.[A-Z0-9]+)*)\b"
    r".*;\s*(?:/\*\s*0x[0-9A-Fa-f]+\s*\*/)?\s*$"
)
_CUOBJDUMP_COMMENT_PREFIX = re.compile(r"^\s*/\*")
_CUOBJDUMP_ENCODING_CONTINUATION = re.compile(r"^\s*/\*\s*0x[0-9A-Fa-f]{16,32}\s*\*/\s*$")


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
    source_tree: str | None = None
    source_capsule_manifest: Path | None = None
    source_capsule_manifest_sha256: str | None = None


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
    source_tree: str | None
    source_capsule_manifest_sha256: str | None


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
class NcuSassInstruction:
    """One validated instruction row from an Nsight Compute source page."""

    address: int
    mnemonic: str


@dataclass(frozen=True)
class NcuSassKernel:
    """One exact Nsight Compute kernel section and its validated SASS rows."""

    name: str
    instructions: tuple[NcuSassInstruction, ...]


@dataclass(frozen=True)
class NcuProfileEvidence:
    """Metrics and retained profiler source/SASS export for one execution."""

    metrics: tuple[NcuKernelMetrics, ...]
    report_path: str
    report_sha256: str
    sass_source_path: str
    sass_source_sha256: str
    final_hlo: str
    persistent_cache: Mapping[str, Any] = field(default_factory=dict)


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
    device_to_host_count: int
    device_to_host_bytes: int
    unexpected_copy_count: int


@dataclass(frozen=True)
class _NsysRange:
    name: str
    start: int
    end: int
    event_type: int
    domain_id: int
    global_tid: int
    end_global_tid: int | None


@dataclass(frozen=True)
class _NsysKernel:
    start: int
    end: int
    name: str
    device_id: int
    correlation_id: int


@dataclass(frozen=True)
class _NsysRuntime:
    start: int
    end: int
    global_tid: int
    correlation_id: int | None


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
    """Return exact normalized entry names from closed cuobjdump sections."""
    if not sass.strip():
        raise ValueError("cuobjdump SASS is empty")
    if "\0" in sass:
        raise ValueError("cuobjdump SASS contains NUL")
    if _NCU_SASS_FAILURE_PATTERN.search(sass) is not None:
        raise ValueError("cuobjdump SASS contains a warning, error, or unavailable source")

    names: list[str] = []
    current_name: str | None = None
    addresses: list[int] = []
    previous_was_instruction = False

    def finish_section() -> None:
        nonlocal current_name, addresses
        if current_name is None:
            return
        if not addresses:
            raise ValueError(f"cuobjdump SASS function {current_name!r} contains no valid instructions")
        if addresses != sorted(set(addresses)):
            raise ValueError(f"cuobjdump SASS function {current_name!r} has repeated or reordered addresses")
        names.append(current_name)
        current_name = None
        addresses = []

    for line in sass.splitlines():
        function = _CUOBJDUMP_FUNCTION_PATTERN.fullmatch(line)
        if function is not None:
            finish_section()
            current_name = normalize_cuda_kernel_name(function.group("name"))
            previous_was_instruction = False
            continue
        if _CUOBJDUMP_FUNCTION_PREFIX.match(line) is not None:
            raise ValueError(f"cuobjdump SASS contains a malformed function identity: {line!r}")

        instruction = _CUOBJDUMP_INSTRUCTION_PATTERN.fullmatch(line)
        if instruction is not None:
            if current_name is None:
                raise ValueError("cuobjdump SASS contains an instruction outside a function section")
            addresses.append(int(instruction.group("address"), 16))
            previous_was_instruction = True
            continue

        if _CUOBJDUMP_ENCODING_CONTINUATION.fullmatch(line) is not None:
            if current_name is None or not previous_was_instruction:
                raise ValueError("cuobjdump SASS contains a standalone instruction encoding")
            previous_was_instruction = False
            continue
        if _CUOBJDUMP_COMMENT_PREFIX.match(line) is not None:
            raise ValueError(f"cuobjdump SASS contains a malformed address-bearing instruction: {line!r}")
        previous_was_instruction = False

    finish_section()
    if not names:
        raise ValueError("cuobjdump SASS contains no function identities")
    if len(set(names)) != len(names):
        raise ValueError("cuobjdump SASS repeats a function identity")
    return tuple(names)


def validate_cuda_sass_kernel_topology(sass: str, expected_names: Sequence[str]) -> tuple[str, ...]:
    """Require exact unique kernel coverage independent of tool emission order."""
    expected = tuple(expected_names)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected cuobjdump kernel topology must be unique and nonempty")
    if any(normalize_cuda_kernel_name(name) != name for name in expected):
        raise ValueError("expected cuobjdump kernel topology must use canonical CUDA symbols")
    actual = cuda_sass_kernel_names(sass)
    missing = tuple(name for name in expected if name not in actual)
    unexpected = tuple(name for name in actual if name not in expected)
    if missing or unexpected or len(actual) != len(expected):
        raise ValueError(
            "cuobjdump SASS kernel coverage differs from the generated topology: "
            f"missing={missing}, unexpected={unexpected}, actual={actual}"
        )
    return actual


def _load_source_capsule_manifest(config: RunnerConfig) -> dict[str, Any]:
    manifest_path = config.source_capsule_manifest
    manifest_sha256 = config.source_capsule_manifest_sha256
    source_tree = config.source_tree
    if manifest_path is None or manifest_sha256 is None or source_tree is None:
        raise ValueError("capsule source preflight requires manifest path, manifest SHA-256, and source tree")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
        raise ValueError("source capsule manifest SHA-256 must be full lowercase hex")
    if not re.fullmatch(r"[0-9a-f]{40}", source_tree):
        raise ValueError("source tree must be a full lowercase Git SHA")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("source capsule manifest must be a regular file")
    raw = manifest_path.read_bytes()
    if len(raw) > 4 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("source capsule manifest differs from the trusted launch identity")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("source capsule manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != {"archive", "members", "schema_version", "source"}:
        raise ValueError("source capsule manifest must use the closed schema")
    canonical = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if raw != canonical:
        raise ValueError("source capsule manifest must use canonical JSON encoding")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("source capsule schema_version must be 1")
    if manifest["source"] != {"commit": config.source_sha, "tree": source_tree}:
        raise ValueError("source capsule commit or tree differs from runner configuration")
    archive = manifest["archive"]
    if (
        not isinstance(archive, dict)
        or set(archive) != {"filename", "sha256"}
        or archive["filename"] != "h100-evidence-source-capsule.zip"
        or not isinstance(archive["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", archive["sha256"]) is None
    ):
        raise ValueError("source capsule archive identity is malformed")
    members = manifest["members"]
    if not isinstance(members, list) or not members or len(members) > 10_000:
        raise ValueError("source capsule members must be a bounded nonempty list")
    records: dict[str, dict[str, Any]] = {}
    total_size = 0
    for record in members:
        if not isinstance(record, dict) or set(record) != {"mode", "path", "sha256", "size", "type"}:
            raise ValueError("source capsule member must use the closed schema")
        source_path = record["path"]
        if not isinstance(source_path, str):
            raise ValueError("source capsule path must be a string")
        path = Path(source_path)
        if not source_path or path.is_absolute() or path.as_posix() != source_path or ".." in path.parts:
            raise ValueError(f"source capsule path is not normalized and relative: {source_path!r}")
        if source_path in records:
            raise ValueError(f"source capsule repeats path: {source_path}")
        if record["type"] == "file" and record["mode"] not in {"100644", "100755"}:
            raise ValueError(f"source capsule file has invalid mode: {record}")
        if record["type"] == "symlink" and record["mode"] != "120000":
            raise ValueError(f"source capsule symlink has invalid mode: {record}")
        if record["type"] not in {"file", "symlink"}:
            raise ValueError(f"source capsule member has invalid type: {record}")
        if type(record["size"]) is not int or not 0 <= record["size"] <= 8 * 1024 * 1024:
            raise ValueError(f"source capsule member has invalid size: {record}")
        if not isinstance(record["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
            raise ValueError(f"source capsule member has invalid SHA-256: {record}")
        total_size += record["size"]
        if total_size > 32 * 1024 * 1024:
            raise ValueError("source capsule expands beyond the reviewed bound")
        records[source_path] = record
    if list(records) != sorted(records):
        raise ValueError("source capsule members must be in canonical path order")
    actual_paths = {
        path.relative_to(config.source_root).as_posix()
        for path in config.source_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_paths != set(records):
        raise ValueError("source capsule file set differs from its manifest")
    for source_path, record in records.items():
        path = config.source_root / source_path
        if record["type"] == "symlink":
            if not path.is_symlink():
                raise ValueError(f"source capsule member is not the required symlink: {source_path}")
            contents = os.readlink(path).encode()
            target = Path(os.readlink(path))
            resolved = Path(os.path.normpath(path.parent / target))
            if target.is_absolute() or not resolved.is_relative_to(config.source_root):
                raise ValueError(f"source capsule symlink escapes source root: {source_path}")
        else:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"source capsule member is not a regular file: {source_path}")
            expected_mode = 0o755 if record["mode"] == "100755" else 0o644
            if stat.S_IMODE(path.stat().st_mode) != expected_mode:
                raise ValueError(f"source capsule member mode differs from manifest: {source_path}")
            contents = path.read_bytes()
        if len(contents) != record["size"] or hashlib.sha256(contents).hexdigest() != record["sha256"]:
            raise ValueError(f"source capsule member content differs from manifest: {source_path}")
    return manifest


def audit_imported_local_modules(config: RunnerConfig, modules: Mapping[str, Any] | None = None) -> None:
    """Require every imported Marin-local module to match a capsule member."""
    if config.source_capsule_manifest is None:
        return
    manifest = _load_source_capsule_manifest(config)
    records = {record["path"]: record for record in manifest["members"]}
    local_prefixes = ("tile_lifetime", "lib.tile_lifetime")
    imported_modules = sys.modules if modules is None else modules
    for name, module in tuple(imported_modules.items()):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            continue
        path = Path(module_file).resolve()
        try:
            relative = path.relative_to(config.source_root).as_posix()
        except ValueError:
            if name in local_prefixes or name.startswith(tuple(f"{prefix}." for prefix in local_prefixes)):
                raise ValueError(f"local module {name} loaded outside the source capsule: {path}") from None
            continue
        record = records.get(relative)
        if record is None or record["type"] != "file" or file_sha256(path) != record["sha256"]:
            raise ValueError(f"imported local module {name} is not an exact source capsule member: {relative}")


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

    if config.source_capsule_manifest is None:
        if config.source_tree is not None or config.source_capsule_manifest_sha256 is not None:
            raise ValueError("capsule source preflight arguments must be supplied together")
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
    else:
        _load_source_capsule_manifest(config)
        head = config.source_sha

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
        source_tree=config.source_tree,
        source_capsule_manifest_sha256=config.source_capsule_manifest_sha256,
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
    """Parse the pinned wide raw-page Nsight Compute CSV contract."""
    metadata = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= _MAX_NCU_CSV_BYTES:
        raise ValueError("Nsight Compute CSV must be a nonempty regular file within the reviewed byte bound")
    with path.open("rb") as stream:
        payload = stream.read(_MAX_NCU_CSV_BYTES + 1)
    if len(payload) != metadata.st_size or len(payload) > _MAX_NCU_CSV_BYTES:
        raise ValueError("Nsight Compute CSV changed or exceeded its reviewed byte bound")
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Nsight Compute CSV must be valid UTF-8") from error
    if "\x00" in source:
        raise ValueError("Nsight Compute CSV must not contain NUL bytes")
    lines = tuple(line for line in source.splitlines() if not line.startswith("=="))
    reader = csv.DictReader(lines)
    fieldnames = reader.fieldnames
    if fieldnames is None or not fieldnames or any(field is None or not field for field in fieldnames):
        raise ValueError("Nsight Compute output has no valid CSV header")
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError("Nsight Compute output repeats CSV columns")
    required = (*_NCU_IDENTITY_FIELDS, *_NCU_METRICS)
    missing = tuple(field for field in required if field not in fieldnames)
    if missing:
        raise ValueError(f"Nsight Compute output omits required columns: {missing}")
    rows = tuple(reader)
    if not rows:
        raise ValueError("Nsight Compute output contains no units row")
    if any(None in row for row in rows):
        raise ValueError("Nsight Compute output has values beyond the declared CSV columns")

    units = rows[0]
    exact_units = all(units[field] == "" for field in _NCU_IDENTITY_FIELDS) and all(
        units[metric] == unit for metric, unit in _NCU_METRIC_UNITS.items()
    )
    if not exact_units:
        raise ValueError("Nsight Compute first data row must be the exact units row")
    if len(rows) < 2:
        raise ValueError("Nsight Compute output contains no kernel rows after the units row")

    records: list[NcuKernelMetrics] = []
    seen: set[tuple[str, str]] = set()
    for row in rows[1:]:
        repeats_units = all(row[metric] == unit for metric, unit in _NCU_METRIC_UNITS.items())
        if all(row[field] == "" for field in _NCU_IDENTITY_FIELDS) or repeats_units:
            raise ValueError("Nsight Compute units row may appear only once and only first")
        name = normalize_cuda_kernel_name(_csv_field(row, "Kernel Name"))
        identifier = _csv_field(row, "ID")
        key = (identifier, name)
        if key in seen:
            raise ValueError(f"Nsight Compute repeats wide metric row for kernel {name!r}")
        seen.add(key)
        metrics = {metric: _csv_field(row, metric) for metric in _NCU_METRICS}
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


def _unrecognized_ncu_sass_record(line_number: int, line: str) -> ValueError:
    line_bytes = line.encode("utf-8")
    metadata = {
        "line_number": line_number,
        "line_sha256": hashlib.sha256(line_bytes).hexdigest(),
        "line_structure": _ncu_sass_line_structure(line),
        "line_utf8_bytes": len(line_bytes),
    }
    detail = json.dumps(metadata, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    message = _NCU_SASS_DIAGNOSTIC_PREFIX + detail
    if len(message.encode("utf-8")) > _MAX_NCU_SASS_DIAGNOSTIC_BYTES:
        return ValueError("unrecognized Nsight Compute SASS export record; diagnostic exceeds reviewed bound")
    return ValueError(message)


def _ncu_sass_line_structure(line: str) -> dict[str, object]:
    ascii_classes = {
        "control": 0,
        "digit": 0,
        "lowercase": 0,
        "punctuation": 0,
        "uppercase": 0,
        "whitespace": 0,
    }
    for character in line:
        codepoint = ord(character)
        if codepoint >= 128:
            continue
        if "A" <= character <= "Z":
            character_class = "uppercase"
        elif "a" <= character <= "z":
            character_class = "lowercase"
        elif "0" <= character <= "9":
            character_class = "digit"
        elif character in " \t\r\n\v\f":
            character_class = "whitespace"
        elif character in string.punctuation:
            character_class = "punctuation"
        elif codepoint < 32 or codepoint == 127:
            character_class = "control"
        else:
            raise AssertionError("ASCII character classification is incomplete")
        ascii_classes[character_class] += 1

    tokens = re.findall(r"\S+", line)
    public_words = frozenset(_NCU_SASS_PUBLIC_WORD_PATTERN.findall(line))
    return {
        "ascii_classes": ascii_classes,
        "delimiters": {
            "colon": line.count(":"),
            "comma": line.count(","),
            "hyphen": line.count("-"),
            "pipe": line.count("|"),
        },
        "leading_spaces": len(line) - len(line.lstrip(" ")),
        "non_ascii_codepoints": sum(ord(character) >= 128 for character in line),
        "public_patterns": {
            "header": line == _NCU_SASS_HEADER,
            "instruction": _NCU_SASS_INSTRUCTION_PATTERN.match(line) is not None,
            "section": _NCU_SASS_SECTION_PATTERN.fullmatch(line) is not None,
            "separator": line == _NCU_SASS_SEPARATOR,
            "status": _NCU_SASS_STATUS_PATTERN.fullmatch(line) is not None,
        },
        "public_vocabulary": {word: word in public_words for word in _NCU_SASS_PUBLIC_WORDS},
        "spaces": line.count(" "),
        "tabs": line.count("\t"),
        "token_count": len(tokens),
        "token_max_utf8_bytes": max((len(token.encode("utf-8")) for token in tokens), default=0),
        "trailing_spaces": len(line) - len(line.rstrip(" ")),
    }


def parse_ncu_sass(source: str, expected_names: Sequence[str]) -> tuple[NcuSassKernel, ...]:
    """Parse a closed Nsight Compute SASS source-page export."""
    if not source or len(source.encode("utf-8")) > _MAX_NCU_SASS_BYTES or "\x00" in source:
        raise ValueError("Nsight Compute SASS export violates its reviewed text bound")
    expected = tuple(normalize_cuda_kernel_name(name) for name in expected_names)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected Nsight Compute SASS kernel identities must be unique and nonempty")
    if _NCU_SASS_FAILURE_PATTERN.search(source):
        raise ValueError("Nsight Compute SASS export contains a warning, error, or unavailable source")

    lines = source.splitlines()
    for line_number, line in enumerate(lines, start=1):
        if len(line) > _MAX_NCU_SASS_LINE_CHARS:
            raise ValueError(f"Nsight Compute SASS export line {line_number} exceeds its reviewed bound")
    if not lines or lines[0] != _NCU_SASS_SEPARATOR:
        raise ValueError("Nsight Compute SASS export omits its exact line-1 table separator")

    sections: list[NcuSassKernel] = []
    current_name: str | None = None
    instructions: list[NcuSassInstruction] = []
    header_seen = False
    separator_seen = False

    def finish_section() -> None:
        nonlocal current_name, instructions, header_seen, separator_seen
        if current_name is None:
            return
        if not header_seen or not separator_seen or not instructions:
            raise ValueError(f"Nsight Compute SASS section {current_name!r} is structurally incomplete")
        sections.append(NcuSassKernel(name=current_name, instructions=tuple(instructions)))
        current_name = None
        instructions = []
        header_seen = False
        separator_seen = False

    for line_number, line in enumerate(lines[1:], start=2):
        section = _NCU_SASS_SECTION_PATTERN.fullmatch(line)
        if section is not None:
            finish_section()
            current_name = normalize_cuda_kernel_name(section.group("name"))
            continue
        if not line.strip():
            continue
        if line == _NCU_SASS_HEADER:
            if current_name is None or header_seen or separator_seen or instructions:
                raise ValueError(f"misplaced Nsight Compute SASS header at line {line_number}")
            header_seen = True
            continue
        if line == _NCU_SASS_SEPARATOR:
            if current_name is None or not header_seen or separator_seen or instructions:
                raise ValueError(f"misplaced Nsight Compute SASS separator at line {line_number}")
            separator_seen = True
            continue
        instruction = _NCU_SASS_INSTRUCTION_PATTERN.match(line)
        if instruction is None or current_name is None or not separator_seen:
            raise _unrecognized_ncu_sass_record(line_number, line)
        mnemonic = instruction.group("mnemonic")
        if mnemonic.split(".", maxsplit=1)[0] not in _SASS_OPCODE_BASES:
            raise ValueError(f"unrecognized SASS instruction mnemonic: {mnemonic!r}")
        address_text = instruction.group("address")
        instructions.append(
            NcuSassInstruction(
                address=int(address_text.removeprefix("0x"), 16),
                mnemonic=mnemonic,
            )
        )
    finish_section()

    actual = tuple(section.name for section in sections)
    if len(set(actual)) != len(actual):
        raise ValueError("Nsight Compute SASS export repeats a kernel section")
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError(f"Nsight Compute SASS kernel coverage differs: expected {expected}, got {actual}")
    by_name = {section.name: section for section in sections}
    return tuple(by_name[name] for name in expected)


def _parse_ncu_sass_file(path: Path, expected_names: Sequence[str]) -> tuple[NcuSassKernel, ...]:
    metadata = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= _MAX_NCU_SASS_BYTES:
        raise ValueError("Nsight Compute SASS export must be a bounded regular file")
    with path.open("rb") as stream:
        payload = stream.read(_MAX_NCU_SASS_BYTES + 1)
    if len(payload) != metadata.st_size or len(payload) > _MAX_NCU_SASS_BYTES:
        raise ValueError("Nsight Compute SASS export changed or exceeded its reviewed byte bound")
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Nsight Compute SASS export must be valid UTF-8") from error
    return parse_ncu_sass(source, expected_names)


def _csv_field(row: Mapping[str | None, str | list[str] | None], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Nsight Compute row omits required field {name!r}")
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


def parse_nsys_sqlite(
    path: Path,
    expected_ranges: tuple[str, ...],
    *,
    report_path: Path | None = None,
) -> tuple[TraceRange, ...]:
    """Read exactly associated CUDA kernels and copies for each required range."""
    try:
        with sqlite3.connect(path) as database:
            tables = {str(row[0]) for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            required = {
                "NVTX_EVENTS",
                "StringIds",
                "CUPTI_ACTIVITY_KIND_KERNEL",
                "CUPTI_ACTIVITY_KIND_RUNTIME",
            }
            if not required.issubset(tables):
                raise ValueError(
                    f"Nsight Systems SQLite export omits required trace tables: {tuple(sorted(required - tables))}"
                )
            ranges = _nsys_ranges(database)
            kernels = _nsys_kernels(database)
            runtimes = _nsys_runtimes(database)
            runtime_by_correlation = _runtime_by_kernel_correlation(kernels, runtimes)
            if tuple(ranges) != expected_ranges:
                raise ValueError("Nsight Systems NVTX ranges do not match the exact steady-state schedule")
            if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
                copies = _nsys_copies(database)
            else:
                _validate_lazy_memcpy_absence(database, kernels)
                copies = ()

            records = []
            for name in expected_ranges:
                trace_range = ranges[name]
                contained_kernels = tuple(
                    kernel
                    for kernel in kernels
                    if _kernel_is_associated(trace_range, kernel, runtime_by_correlation[kernel.correlation_id])
                )
                contained_copies = tuple(
                    copy for copy in copies if trace_range.start <= copy[0] and copy[1] <= trace_range.end
                )
                if not contained_kernels:
                    diagnostic = _no_kernel_diagnostic(
                        database,
                        tables,
                        path,
                        report_path,
                        trace_range,
                        kernels,
                        runtime_by_correlation,
                    )
                    serialized = json.dumps(diagnostic, sort_keys=True, separators=(",", ":"))
                    message = (
                        f"Nsight Systems range {name!r} contains no associated CUDA kernels diagnostic={serialized}"
                    )
                    if len(message) > _MAX_NSYS_NO_KERNEL_DIAGNOSTIC_CHARS:
                        raise AssertionError("Nsight Systems no-kernel diagnostic exceeds its reviewed bound")
                    raise ValueError(message)
                d2d = tuple(copy for copy in contained_copies if copy[3] == "device_to_device")
                h2d = tuple(copy for copy in contained_copies if copy[3] == "host_to_device")
                d2h = tuple(copy for copy in contained_copies if copy[3] == "device_to_host")
                unexpected = tuple(
                    copy
                    for copy in contained_copies
                    if copy[3] not in {"device_to_device", "host_to_device", "device_to_host"}
                )
                records.append(
                    TraceRange(
                        name=name,
                        ordered_kernel_names=tuple(kernel.name for kernel in contained_kernels),
                        kernel_duration_ns=sum(kernel.end - kernel.start for kernel in contained_kernels),
                        device_to_device_count=len(d2d),
                        device_to_device_bytes=sum(copy[2] for copy in d2d),
                        host_to_device_count=len(h2d),
                        host_to_device_bytes=sum(copy[2] for copy in h2d),
                        device_to_host_count=len(d2h),
                        device_to_host_bytes=sum(copy[2] for copy in d2h),
                        unexpected_copy_count=len(unexpected),
                    )
                )
            return tuple(records)
    except sqlite3.DatabaseError as error:
        raise ValueError(f"Nsight Systems SQLite export is unreadable: {error}") from error


def _nsys_ranges(database: sqlite3.Connection) -> dict[str, _NsysRange]:
    columns = _table_columns(database, "NVTX_EVENTS")
    required = {"start", "end", "eventType", "text", "globalTid", "endGlobalTid", "domainId"}
    if not required.issubset(columns):
        raise ValueError(f"NVTX_EVENTS omits required columns: {tuple(sorted(required - columns))}")
    records: dict[str, _NsysRange] = {}
    query = (
        "SELECT start, end, eventType, text, globalTid, endGlobalTid, domainId "
        "FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start, end"
    )
    for start, end, event_type, text, global_tid, end_global_tid, domain_id in database.execute(query):
        if not isinstance(text, str) or not text.startswith("contract_map.steady."):
            continue
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise ValueError(f"Nsight Systems has an invalid steady-state NVTX range {text!r}")
        if type(event_type) is not int or event_type != 59:
            raise ValueError(f"Nsight Systems steady range {text!r} is not an NVTX push/pop range")
        if type(domain_id) is not int or domain_id != 0:
            raise ValueError(f"Nsight Systems steady range {text!r} is not in the default NVTX domain")
        if type(global_tid) is not int or global_tid < 0:
            raise ValueError(f"Nsight Systems steady range {text!r} has an invalid start thread")
        if end_global_tid is not None and (type(end_global_tid) is not int or end_global_tid != global_tid):
            raise ValueError(f"Nsight Systems steady range {text!r} does not end on the same thread")
        if text in records:
            raise ValueError(f"Nsight Systems repeats steady-state NVTX range {text!r}")
        records[text] = _NsysRange(text, start, end, event_type, domain_id, global_tid, end_global_tid)
    ordered = tuple(records.values())
    for previous, current in itertools.pairwise(ordered):
        if current.start < previous.end:
            raise ValueError("Nsight Systems steady ranges do not follow strict source order without overlap")
    return records


def _nsys_kernels(database: sqlite3.Connection) -> tuple[_NsysKernel, ...]:
    columns = _table_columns(database, "CUPTI_ACTIVITY_KIND_KERNEL")
    name_column = "demangledName" if "demangledName" in columns else "shortName"
    if not {"start", "end", name_column, "deviceId", "correlationId"}.issubset(columns):
        raise ValueError("CUPTI kernel table omits time, device, correlation, or kernel-name identity")
    query = (
        f"SELECT kernel.start, kernel.end, strings.value, kernel.deviceId, kernel.correlationId "
        f"FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel LEFT JOIN StringIds AS strings "
        f"ON kernel.{name_column} = strings.id ORDER BY kernel.start, kernel.end"
    )
    records = []
    rows = tuple(database.execute(query))
    kernel_count = database.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    if type(kernel_count) is not int or len(rows) != kernel_count:
        raise ValueError("CUPTI kernel identities do not resolve exactly once through StringIds")
    previous_start: int | None = None
    for start, end, name, device_id, correlation_id in rows:
        if (
            type(start) is not int
            or type(end) is not int
            or type(device_id) is not int
            or type(correlation_id) is not int
            or start < 0
            or end <= start
            or device_id < 0
            or correlation_id < 0
            or not isinstance(name, str)
            or not name.strip()
        ):
            raise ValueError("CUPTI kernel table contains an invalid activity record")
        if previous_start == start:
            raise ValueError("CUPTI kernel launch order is ambiguous because records have equal start timestamps")
        previous_start = start
        records.append(_NsysKernel(start, end, normalize_cuda_kernel_name(name), device_id, correlation_id))
    return tuple(records)


def _nsys_runtimes(database: sqlite3.Connection) -> tuple[_NsysRuntime, ...]:
    columns = _table_columns(database, "CUPTI_ACTIVITY_KIND_RUNTIME")
    required = {"start", "end", "globalTid", "correlationId"}
    if not required.issubset(columns):
        raise ValueError(f"CUPTI runtime table omits required columns: {tuple(sorted(required - columns))}")
    records = []
    for start, end, global_tid, correlation_id in database.execute(
        "SELECT start, end, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start, end"
    ):
        if (
            type(start) is not int
            or type(end) is not int
            or type(global_tid) is not int
            or start < 0
            or end <= start
            or global_tid < 0
            or (correlation_id is not None and (type(correlation_id) is not int or correlation_id < 0))
        ):
            raise ValueError("CUPTI runtime table contains an invalid activity record")
        records.append(_NsysRuntime(start, end, global_tid, correlation_id))
    return tuple(records)


def _runtime_by_kernel_correlation(
    kernels: tuple[_NsysKernel, ...],
    runtimes: tuple[_NsysRuntime, ...],
) -> dict[int, _NsysRuntime]:
    by_correlation: dict[int, list[_NsysRuntime]] = {}
    for runtime in runtimes:
        if runtime.correlation_id is not None:
            by_correlation.setdefault(runtime.correlation_id, []).append(runtime)
    resolved = {}
    for correlation_id in {kernel.correlation_id for kernel in kernels}:
        matches = by_correlation.get(correlation_id, [])
        if len(matches) != 1:
            raise ValueError(
                f"CUPTI kernel correlation {correlation_id} does not resolve exactly once through runtime activity"
            )
        resolved[correlation_id] = matches[0]
    return resolved


def _kernel_is_associated(trace_range: _NsysRange, kernel: _NsysKernel, runtime: _NsysRuntime) -> bool:
    return (
        trace_range.start <= kernel.start
        and kernel.end <= trace_range.end
        and trace_range.start <= runtime.start
        and runtime.end <= trace_range.end
        and runtime.global_tid == trace_range.global_tid
    )


def _interval_counts(start: int, end: int, records: Sequence[tuple[int, int]]) -> dict[str, int]:
    counts = {"before": 0, "contained": 0, "overlap": 0, "after": 0}
    for record_start, record_end in records:
        if record_end <= start:
            counts["before"] += 1
        elif record_start >= end:
            counts["after"] += 1
        elif start <= record_start and record_end <= end:
            counts["contained"] += 1
        else:
            counts["overlap"] += 1
    return counts


def _graph_trace_intervals(database: sqlite3.Connection, tables: set[str]) -> tuple[tuple[int, int], ...]:
    table = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
    if table not in tables:
        return ()
    columns = _table_columns(database, table)
    required = {"start", "end", "deviceId", "correlationId", "graphId", "graphExecId"}
    if not required.issubset(columns):
        raise ValueError(f"CUPTI graph trace table omits required columns: {tuple(sorted(required - columns))}")
    intervals = []
    for start, end in database.execute(f"SELECT start, end FROM {table} ORDER BY start, end"):
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise ValueError("CUPTI graph trace table contains an invalid activity record")
        intervals.append((start, end))
    return tuple(intervals)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_identity(path: Path | None) -> dict[str, int | str] | None:
    if path is None:
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"retained Nsight Systems artifact is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"retained Nsight Systems artifact is empty: {path}")
    return {"bytes": size, "sha256": file_sha256(path)}


def _no_kernel_diagnostic(
    database: sqlite3.Connection,
    tables: set[str],
    sqlite_path: Path,
    report_path: Path | None,
    trace_range: _NsysRange,
    kernels: tuple[_NsysKernel, ...],
    runtime_by_correlation: Mapping[int, _NsysRuntime],
) -> dict[str, Any]:
    kernel_intervals = tuple((kernel.start, kernel.end) for kernel in kernels)
    graph_intervals = _graph_trace_intervals(database, tables)
    names = tuple(kernel.name for kernel in kernels)
    relevant_tables = {}
    for table in _NSYS_RELEVANT_TABLES:
        if table not in tables:
            relevant_tables[table] = {"present": False}
            continue
        columns = tuple(sorted(_table_columns(database, table)))
        row_count = database.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if type(row_count) is not int or row_count < 0:
            raise ValueError(f"Nsight Systems table {table!r} has an invalid row count")
        relevant_tables[table] = {
            "column_count": len(columns),
            "columns_sha256": _canonical_sha256(columns),
            "present": True,
            "row_count": row_count,
        }
    runtime_rows = tuple(runtime_by_correlation[kernel.correlation_id] for kernel in kernels)
    return {
        "schema": "shuttle.nsys_no_kernel_diagnostic.v1",
        "range": {
            "domain_id": trace_range.domain_id,
            "duration_ns": trace_range.end - trace_range.start,
            "end_global_tid": trace_range.end_global_tid,
            "end_ns": trace_range.end,
            "event_type": trace_range.event_type,
            "global_tid": trace_range.global_tid,
            "name": trace_range.name,
            "start_ns": trace_range.start,
        },
        "kernels": {
            "end_max_ns": max((kernel.end for kernel in kernels), default=None),
            "interval_counts": _interval_counts(trace_range.start, trace_range.end, kernel_intervals),
            "name_count": len(names),
            "ordered_names_sha256": _canonical_sha256(names),
            "nearest_next_start_offset_ns": min(
                (kernel.start - trace_range.end for kernel in kernels if kernel.start >= trace_range.end),
                default=None,
            ),
            "nearest_previous_end_offset_ns": min(
                (trace_range.start - kernel.end for kernel in kernels if kernel.end <= trace_range.start),
                default=None,
            ),
            "row_count": len(kernels),
            "start_min_ns": min((kernel.start for kernel in kernels), default=None),
            "unique_name_count": len(set(names)),
            "unique_names_sha256": _canonical_sha256(tuple(sorted(set(names)))),
        },
        "runtime_correlation": {
            "associated_kernel_count": sum(
                _kernel_is_associated(
                    trace_range,
                    kernel,
                    runtime_by_correlation[kernel.correlation_id],
                )
                for kernel in kernels
            ),
            "resolved_kernel_count": len(runtime_rows),
            "same_thread_contained_count": sum(
                trace_range.start <= runtime.start
                and runtime.end <= trace_range.end
                and runtime.global_tid == trace_range.global_tid
                for runtime in runtime_rows
            ),
        },
        "graph_trace": {
            "interval_counts": _interval_counts(trace_range.start, trace_range.end, graph_intervals),
            "row_count": len(graph_intervals),
        },
        "sqlite": _file_identity(sqlite_path),
        "report": _file_identity(report_path),
        "database": {
            "application_id": database.execute("PRAGMA application_id").fetchone()[0],
            "schema_version": database.execute("PRAGMA schema_version").fetchone()[0],
            "table_count": len(tables),
            "table_names_sha256": _canonical_sha256(tuple(sorted(tables))),
            "user_version": database.execute("PRAGMA user_version").fetchone()[0],
        },
        "relevant_tables": relevant_tables,
        "profile_args": list(_NSYS_PROFILE_ARGS),
        "export_args": list(_NSYS_EXPORT_ARGS),
    }


def _validate_lazy_memcpy_absence(
    database: sqlite3.Connection,
    kernels: tuple[_NsysKernel, ...],
) -> None:
    """Prove a lazy export omitted an empty memcpy table rather than CUDA trace data."""
    if not kernels:
        raise ValueError("Nsight Systems cannot treat a missing memcpy table as empty without CUDA kernels")
    columns = _table_columns(database, "TARGET_INFO_GPU")
    if not {"id", "name"}.issubset(columns):
        raise ValueError("Nsight Systems cannot prove CUDA trace provenance without TARGET_INFO_GPU identity")
    devices: dict[int, str] = {}
    for device_id, name in database.execute("SELECT id, name FROM TARGET_INFO_GPU"):
        if type(device_id) is not int or device_id < 0 or not isinstance(name, str) or not name.strip():
            raise ValueError("TARGET_INFO_GPU contains an invalid CUDA device identity")
        if device_id in devices:
            raise ValueError(f"TARGET_INFO_GPU repeats CUDA device id {device_id}")
        devices[device_id] = name
    if not devices:
        raise ValueError("TARGET_INFO_GPU contains no CUDA device identity")
    missing_devices = tuple(sorted({kernel.device_id for kernel in kernels} - devices.keys()))
    if missing_devices:
        raise ValueError(f"CUPTI kernels reference unknown CUDA device ids: {missing_devices}")


def _nsys_copies(database: sqlite3.Connection) -> tuple[tuple[int, int, int, str], ...]:
    columns = _table_columns(database, "CUPTI_ACTIVITY_KIND_MEMCPY")
    if not {"start", "end", "bytes", "copyKind"}.issubset(columns):
        raise ValueError("CUPTI memcpy table omits start, end, bytes, or copyKind")
    kinds = {1: "host_to_device", 2: "device_to_host", 8: "device_to_device"}
    records = []
    for start, end, size, kind in database.execute(
        "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"
    ):
        if (
            type(start) is not int
            or type(end) is not int
            or type(size) is not int
            or type(kind) is not int
            or start < 0
            or end <= start
            or size < 0
            or kind < 0
        ):
            raise ValueError("CUPTI memcpy table contains an invalid activity record")
        records.append((start, end, size, kinds.get(kind, f"copy_kind_{kind}")))
    return tuple(records)


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
        loaded_image_sass_path.write_text(loaded_image_sass.stdout)
        try:
            validate_cuda_sass_kernel_topology(loaded_image_sass.stdout, candidate.generated.kernel_names)
        except ValueError as error:
            raise RuntimeError(
                f"loaded shared-library SASS does not contain the exact generated kernel topology: {error}"
            ) from error
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
    source_config = _worker_source_config(args)
    if source_config.source_capsule_manifest is not None:
        _load_source_capsule_manifest(source_config)
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
    audit_imported_local_modules(source_config)
    if args.worker is WorkerMode.COMPILE:
        result = _run_compile_worker(args, context, jax=jax)
    elif args.worker is WorkerMode.CASE:
        result = _run_case_worker(args, context, jax=jax)
    elif args.worker is WorkerMode.PROFILE:
        result = _run_profile_worker(args, context, jax=jax)
    else:
        raise ValueError(f"unsupported worker mode: {args.worker}")
    audit_imported_local_modules(source_config)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _worker_source_config(args: argparse.Namespace) -> RunnerConfig:
    source_root = _required_argument(args.source_root, "--source-root").resolve()
    nvcc = args.nvcc.resolve()
    tools = ToolPaths(
        git=nvcc,
        nvidia_smi=nvcc,
        nvcc=nvcc,
        ptxas=nvcc,
        cuobjdump=nvcc,
        ncu=nvcc,
        nsys=nvcc,
    )
    return RunnerConfig(
        source_root=source_root,
        source_sha=_required_argument(args.source_sha, "--source-sha"),
        artifact_directory=args.json_output.parent,
        tools=tools,
        require_jax_version=args.require_jax_version,
        source_tree=args.source_tree,
        source_capsule_manifest=args.source_capsule_manifest,
        source_capsule_manifest_sha256=args.source_capsule_manifest_sha256,
    )


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

    compiled = jax.jit(step).lower(*context.inputs).compile()
    compile_done_monotonic_ns = time.monotonic_ns()
    return compiled, compile_done_monotonic_ns


@dataclass(frozen=True)
class _PersistentCacheTarget:
    cache_key: str
    compile_time: int
    compressed_entry_sha256: str
    serialized_executable_sha256: str


@dataclass(frozen=True)
class _PersistentCacheSnapshot:
    root_identity: str
    file_count: int
    total_bytes: int


def _require_pinned_cache_compression_runtime() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("H100 cache evidence requires the pinned Python 3.12 zlib cache runtime")
    try:
        importlib.metadata.version("zstandard")
    except importlib.metadata.PackageNotFoundError:
        return
    raise RuntimeError("H100 cache evidence requires zstandard to be absent")


def _bounded_file_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    metadata = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("persistent-cache entry must be a regular file")
    if not 0 < metadata.st_size <= maximum_bytes:
        raise RuntimeError("persistent-cache entry has an invalid bounded size")
    with path.open("rb") as stream:
        payload = stream.read(maximum_bytes + 1)
    if len(payload) != metadata.st_size or len(payload) > maximum_bytes:
        raise RuntimeError("persistent-cache entry changed or exceeded its reviewed bound")
    return payload


def _persistent_cache_files(cache_directory: Path) -> tuple[tuple[str, bytes], ...]:
    entries = tuple(itertools.islice(cache_directory.iterdir(), _MAX_PERSISTENT_CACHE_FILES + 1))
    if not entries:
        raise RuntimeError("compile worker produced no persistent-cache artifact")
    if len(entries) > _MAX_PERSISTENT_CACHE_FILES:
        raise RuntimeError("unbounded JAX persistent cache exceeds its reviewed file-count bound")
    entries = tuple(sorted(entries, key=lambda path: path.name))
    if any(path.is_symlink() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode) for path in entries):
        raise RuntimeError("unbounded JAX persistent cache must contain only flat regular cache entries")
    cache_files: list[tuple[str, bytes]] = []
    total_bytes = 0
    for path in entries:
        if _CACHE_FILE_PATTERN.fullmatch(path.name) is None:
            raise RuntimeError("unbounded JAX persistent cache contains an unexpected entry")
        payload = _bounded_file_bytes(path, maximum_bytes=_MAX_PERSISTENT_CACHE_ENTRY_BYTES)
        total_bytes += len(payload)
        if total_bytes > _MAX_PERSISTENT_CACHE_ROOT_BYTES:
            raise RuntimeError("unbounded JAX persistent cache exceeds its reviewed total-byte bound")
        cache_files.append((path.name, payload))
    return tuple(cache_files)


def _persistent_cache_snapshot(cache_files: Sequence[tuple[str, bytes]]) -> _PersistentCacheSnapshot:
    if not cache_files:
        raise RuntimeError("persistent cache snapshot must be nonempty")
    root_identity = hashlib.sha256()
    for name, payload in cache_files:
        encoded_name = name.encode()
        root_identity.update(len(encoded_name).to_bytes(4, byteorder="big"))
        root_identity.update(encoded_name)
        root_identity.update(hashlib.sha256(payload).digest())
    return _PersistentCacheSnapshot(
        root_identity=root_identity.hexdigest(),
        file_count=len(cache_files),
        total_bytes=sum(len(payload) for _, payload in cache_files),
    )


def _persistent_cache_target(
    cache_directory: Path,
    expected_cache_key: str | None = None,
) -> tuple[_PersistentCacheTarget, tuple[tuple[str, bytes], ...]]:
    cache_files = _persistent_cache_files(cache_directory)
    target_names = tuple(
        name
        for name, _ in cache_files
        if _TARGET_CACHE_FILE_PATTERN.fullmatch(name) is not None
        and (expected_cache_key is None or name == f"{expected_cache_key}-cache")
    )
    if len(target_names) != 1:
        qualifier = "" if expected_cache_key is None else " matching the expected key"
        raise RuntimeError(f"persistent cache must contain exactly one jit_step target entry{qualifier}")
    if expected_cache_key is None:
        all_target_names = tuple(
            name for name, _ in cache_files if _TARGET_CACHE_FILE_PATTERN.fullmatch(name) is not None
        )
        if len(all_target_names) != 1:
            raise RuntimeError("persistent cache must contain exactly one jit_step target entry")

    target_name = target_names[0]
    compressed = next(payload for name, payload in cache_files if name == target_name)
    try:
        decompressor = zlib.decompressobj()
        executable_and_time = decompressor.decompress(compressed, _MAX_SERIALIZED_EXECUTABLE_BYTES + 5)
    except zlib.error as error:
        raise RuntimeError("jit_step cache entry is not the pinned zlib cache format") from error
    if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise RuntimeError("jit_step cache entry is not one complete bounded zlib stream")
    if not 4 < len(executable_and_time) <= _MAX_SERIALIZED_EXECUTABLE_BYTES + 4:
        raise RuntimeError("jit_step cache entry has an invalid decompressed size")
    match = _TARGET_CACHE_FILE_PATTERN.fullmatch(target_name)
    assert match is not None
    return (
        _PersistentCacheTarget(
            cache_key=match.group("key"),
            compile_time=int.from_bytes(executable_and_time[:4], byteorder="big"),
            compressed_entry_sha256=hashlib.sha256(compressed).hexdigest(),
            serialized_executable_sha256=hashlib.sha256(executable_and_time[4:]).hexdigest(),
        ),
        cache_files,
    )


def _compile_with_cache_events(
    context: _WorkerCaseContext,
    backend: str,
    *,
    jax: Any,
) -> tuple[Any, int, dict[str, int]]:
    event_counts = {name: 0 for name in _CACHE_EVENT_NAMES}

    def listener(event: str, **metadata: str | int) -> None:
        del metadata
        if event in event_counts:
            event_counts[event] += 1

    jax.monitoring.register_event_listener(listener)
    try:
        compiled, compile_done_monotonic_ns = _compiled_backend(context, backend, jax=jax)
    finally:
        jax.monitoring.unregister_event_listener(listener)
    return compiled, compile_done_monotonic_ns, event_counts


def _load_worker_cache_contract(args: argparse.Namespace) -> dict[str, Any] | None:
    path = getattr(args, "cache_contract", None)
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {"backends", "schema_version", "snapshot"}:
        raise ValueError("worker cache contract has an invalid closed schema")
    if payload["schema_version"] != 1:
        raise ValueError("worker cache contract has an unsupported schema version")
    snapshot = payload["snapshot"]
    if not isinstance(snapshot, dict) or set(snapshot) != {"file_count", "root_identity", "total_bytes"}:
        raise ValueError("worker cache contract has an invalid snapshot schema")
    if not isinstance(payload["backends"], dict) or not payload["backends"]:
        raise ValueError("worker cache contract must contain backend identities")
    return payload


def _validated_worker_cache_snapshot(
    cache_directory: Path,
    contract: Mapping[str, Any],
) -> tuple[tuple[str, bytes], ...]:
    files = _persistent_cache_files(cache_directory)
    observed = _persistent_cache_snapshot(files)
    expected = contract["snapshot"]
    if (
        expected.get("root_identity") != observed.root_identity
        or expected.get("file_count") != observed.file_count
        or expected.get("total_bytes") != observed.total_bytes
    ):
        raise ValueError("worker persistent-cache snapshot differs from its sealed source")
    return files


def _cache_hit_executable(
    args: argparse.Namespace,
    context: _WorkerCaseContext,
    backend: str,
    contract: Mapping[str, Any],
    *,
    jax: Any,
) -> tuple[Any, dict[str, Any]]:
    expected = contract["backends"].get(backend)
    if not isinstance(expected, dict) or set(expected) != {
        "cache_key",
        "compressed_entry_sha256",
        "final_hlo_sha256",
        "serialized_executable_sha256",
    }:
        raise ValueError(f"worker cache contract omits a closed identity for {backend}")
    cache_directory = Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", ""))
    if not cache_directory.is_dir():
        raise RuntimeError("cached worker requires an existing isolated JAX_COMPILATION_CACHE_DIR")
    _validated_worker_cache_snapshot(cache_directory, contract)
    compiled, _, events = _compile_with_cache_events(context, backend, jax=jax)
    files = _validated_worker_cache_snapshot(cache_directory, contract)
    target, _ = _persistent_cache_target(cache_directory, str(expected["cache_key"]))
    if (
        target.serialized_executable_sha256 != expected["serialized_executable_sha256"]
        or target.compressed_entry_sha256 != expected["compressed_entry_sha256"]
    ):
        raise ValueError(f"cached worker loaded a noncanonical serialized executable for {backend}")
    final_hlo = compiled.as_text()
    if hashlib.sha256(final_hlo.encode()).hexdigest() != expected["final_hlo_sha256"]:
        raise ValueError(f"cached worker final HLO differs from the canonical executable for {backend}")
    expected_events = dict(zip(_CACHE_EVENT_NAMES, (1, 1, 0), strict=True))
    if events != expected_events:
        raise ValueError(f"cached worker did not prove one public persistent-cache hit for {backend}")
    snapshot = _persistent_cache_snapshot(files)
    return compiled, {
        "persistent_cache_entry_sha256": target.compressed_entry_sha256,
        "persistent_cache_events": events,
        "persistent_cache_key": target.cache_key,
        "persistent_cache_root_identity": snapshot.root_identity,
        "persistent_cache_serialized_executable_sha256": target.serialized_executable_sha256,
    }


def _run_compile_worker(args: argparse.Namespace, context: _WorkerCaseContext, *, jax: Any) -> dict[str, Any]:
    _require_pinned_cache_compression_runtime()
    cache_contract = _load_worker_cache_contract(args)
    if cache_contract is not None:
        _validated_worker_cache_snapshot(
            Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", "")),
            cache_contract,
        )
    compiled, compile_done_monotonic_ns, cache_events = _compile_with_cache_events(context, args.backend, jax=jax)
    started = time.perf_counter_ns()
    output = compiled(*context.inputs)
    jax.block_until_ready(output)
    first_execution_ns = time.perf_counter_ns() - started
    cache_directory = Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", ""))
    if not cache_directory.is_dir():
        raise RuntimeError("compile worker requires an existing isolated JAX_COMPILATION_CACHE_DIR")
    expected_cache_key = None
    if cache_contract is not None:
        expected_backend = cache_contract["backends"].get(args.backend)
        if not isinstance(expected_backend, dict):
            raise ValueError(f"worker cache contract omits {args.backend}")
        expected_cache_key = str(expected_backend.get("cache_key"))
        _validated_worker_cache_snapshot(cache_directory, cache_contract)
    target, cache_files = _persistent_cache_target(cache_directory, expected_cache_key)
    snapshot = _persistent_cache_snapshot(cache_files)
    if cache_contract is not None and (
        target.serialized_executable_sha256 != expected_backend.get("serialized_executable_sha256")
        or target.compressed_entry_sha256 != expected_backend.get("compressed_entry_sha256")
    ):
        raise ValueError("compile worker cached consumer differs from its canonical executable")
    contract_identity = hashlib.sha256()
    contract_identity.update(target.cache_key.encode())
    contract_identity.update(bytes.fromhex(target.serialized_executable_sha256))
    return {
        "case_id": args.case_id,
        "backend": args.backend,
        "cache_kind": args.cache_kind,
        "compile_done_monotonic_ns": compile_done_monotonic_ns,
        "first_execution_ns": first_execution_ns,
        "persistent_cache_compression": _CACHE_COMPRESSION,
        "persistent_cache_compile_time": target.compile_time,
        "persistent_cache_entry_sha256": target.compressed_entry_sha256,
        "persistent_cache_file_count": len(cache_files),
        "persistent_cache_identity": contract_identity.hexdigest(),
        "persistent_cache_key": target.cache_key,
        "persistent_cache_root_identity": snapshot.root_identity,
        "persistent_cache_serialized_executable_sha256": target.serialized_executable_sha256,
        "persistent_cache_total_bytes": sum(len(payload) for _, payload in cache_files),
        "persistent_cache_events": cache_events,
        "final_hlo": compiled.as_text(),
    }


def _run_profile_worker(args: argparse.Namespace, context: _WorkerCaseContext, *, jax: Any) -> dict[str, Any]:
    contract = _load_worker_cache_contract(args)
    if contract is None:
        raise ValueError("profile worker requires a sealed persistent-cache contract")
    compiled, cache_evidence = _cache_hit_executable(args, context, args.backend, contract, jax=jax)
    jax.block_until_ready(compiled(*context.inputs))
    with _NvtxRange("contract_map.profile", args.nvcc):
        output = compiled(*context.inputs)
        jax.block_until_ready(output)
    return {
        "case_id": args.case_id,
        "backend": args.backend,
        "profiled": True,
        "persistent_cache": cache_evidence,
        "final_hlo": compiled.as_text(),
    }


def _run_case_worker(args: argparse.Namespace, context: _WorkerCaseContext, *, jax: Any) -> dict[str, Any]:
    from tile_lifetime.h100_contract_map_benchmark import (  # noqa: PLC0415
        BackendVariant,
        default_h100_contract_map_benchmark_plan,
    )

    contract = _load_worker_cache_contract(args)
    if contract is None:
        raise ValueError("case worker requires a sealed persistent-cache contract")
    executables = {}
    cache_evidence = {}
    for backend in BackendVariant:
        executable, evidence = _cache_hit_executable(args, context, backend.value, contract, jax=jax)
        executables[backend.value] = executable
        cache_evidence[backend.value] = evidence

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
        "persistent_cache": cache_evidence,
        "final_hlo": final_hlo,
        "numerical": numerical,
        "warmup_samples_ns": warmups,
        "raw_samples": raw_samples,
    }


class _NvtxRange(NvtxRange):
    def __init__(self, name: str, nvcc: Path):
        from tile_lifetime.cuda_toolchain import cuda_toolkit_shared_library  # noqa: PLC0415

        super().__init__(name, cuda_toolkit_shared_library(nvcc, "nvToolsExt"))


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
        REVIEWED_NUMERICAL_FLOORS,
        REVIEWED_NUMERICAL_FLOORS_SHA256,
        BackendVariant,
        MeasurementBoundary,
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
        floor = next(reviewed for reviewed in REVIEWED_NUMERICAL_FLOORS if reviewed.backend is backend)
        output_evidence = {}
        worst_pairs = {}
        for index, name in enumerate(_OUTPUT_NAMES):
            output_evidence[name] = _output_numerical_evidence(index, repeats, reference[index])
            worst_pairs[name] = _worst_pair_diagnostic(
                repeats[0][index],
                reference[index],
                absolute_threshold=floor.output_floor(name).maximum_absolute_error,
            )
        try:
            validate_backend_numerical_evidence(
                backend,
                output_evidence,
                case_id=context.case.case_id,
                measurement_boundary=MeasurementBoundary.LOGICAL_TRAINING_STEP,
            )
        except NumericalFloorError as error:
            raise _with_worst_pair_diagnostic(error, worst_pairs[error.output_name]) from None
        evidence[backend.value] = {
            "reviewed_floors_sha256": REVIEWED_NUMERICAL_FLOORS_SHA256,
            "floors_passed_before_timing": True,
            "outputs": output_evidence,
        }
    return evidence


@dataclass(frozen=True)
class _Bfloat16ScalarDiagnostic:
    hexadecimal: str
    value: float
    sign: str
    exponent: int | None
    classification: str


@dataclass(frozen=True)
class _WorstPairDiagnostic:
    index: tuple[int, ...]
    actual: _Bfloat16ScalarDiagnostic
    reference: _Bfloat16ScalarDiagnostic
    absolute_error: float
    ulp_distance: int
    finite_values: int
    exact_mismatches: int
    one_ulp_mismatches: int
    absolute_threshold: float
    absolute_mismatches: int


def _bfloat16_scalar_diagnostic(value: Any) -> _Bfloat16ScalarDiagnostic:
    import numpy as np  # noqa: PLC0415
    from ml_dtypes import bfloat16  # noqa: PLC0415

    scalar = np.asarray(value, dtype=bfloat16).reshape(())
    bits = int(scalar.view(np.uint16).item())
    exponent_bits = (bits >> 7) & 0xFF
    fraction = bits & 0x7F
    if exponent_bits == 0:
        classification = "zero" if fraction == 0 else "subnormal"
        exponent = -126
    elif exponent_bits == 0xFF:
        classification = "infinity" if fraction == 0 else "nan"
        exponent = None
    else:
        classification = "normal"
        exponent = exponent_bits - 127
    return _Bfloat16ScalarDiagnostic(
        hexadecimal=f"0x{bits:04x}",
        value=float(scalar.astype(np.float32)),
        sign="negative" if bits & 0x8000 else "positive",
        exponent=exponent,
        classification=classification,
    )


def _worst_pair_diagnostic(
    actual: Any,
    reference: Any,
    *,
    absolute_threshold: float,
) -> _WorstPairDiagnostic | None:
    import numpy as np  # noqa: PLC0415
    from ml_dtypes import bfloat16  # noqa: PLC0415

    if not math.isfinite(absolute_threshold) or absolute_threshold < 0.0:
        raise ValueError("worst-pair absolute threshold must be finite and nonnegative")
    actual_array = np.asarray(actual)
    reference_array = np.asarray(reference)
    if actual_array.dtype.name != "bfloat16":
        raise TypeError("worst-pair actual output must have BF16 dtype")
    if actual_array.shape != reference_array.shape:
        raise ValueError("worst-pair actual output and reference must have identical shapes")
    finite = np.isfinite(actual_array) & np.isfinite(reference_array)
    finite_flat_indices = np.flatnonzero(finite.reshape(-1))
    if finite_flat_indices.size == 0:
        return None
    actual_finite = actual_array[finite]
    reference_finite = reference_array[finite]
    ulp = bfloat16_ulp_distance(actual_finite, reference_finite)
    absolute = np.abs(actual_finite.astype(np.float32) - reference_finite.astype(np.float32))
    finite_index = int(np.argmax(ulp))
    flat_index = int(finite_flat_indices[finite_index])
    index = tuple(int(coordinate) for coordinate in np.unravel_index(flat_index, actual_array.shape))
    return _WorstPairDiagnostic(
        index=index,
        actual=_bfloat16_scalar_diagnostic(np.asarray(actual_finite, dtype=bfloat16)[finite_index]),
        reference=_bfloat16_scalar_diagnostic(np.asarray(reference_finite, dtype=bfloat16)[finite_index]),
        absolute_error=float(absolute[finite_index]),
        ulp_distance=int(ulp[finite_index]),
        finite_values=int(finite_flat_indices.size),
        exact_mismatches=int(np.count_nonzero(ulp != 0)),
        one_ulp_mismatches=int(np.count_nonzero(ulp > 1)),
        absolute_threshold=absolute_threshold,
        absolute_mismatches=int(np.count_nonzero(absolute > absolute_threshold)),
    )


def _with_worst_pair_diagnostic(
    error: NumericalFloorError, diagnostic: _WorstPairDiagnostic | None
) -> NumericalFloorError:
    if diagnostic is None:
        return error
    fields = (
        ("worst_index", ",".join(str(coordinate) for coordinate in diagnostic.index)),
        ("worst_actual_hex", diagnostic.actual.hexadecimal),
        ("worst_actual", repr(diagnostic.actual.value)),
        ("worst_actual_sign", diagnostic.actual.sign),
        ("worst_actual_exponent", "none" if diagnostic.actual.exponent is None else str(diagnostic.actual.exponent)),
        ("worst_actual_class", diagnostic.actual.classification),
        ("worst_reference_hex", diagnostic.reference.hexadecimal),
        ("worst_reference", repr(diagnostic.reference.value)),
        ("worst_reference_sign", diagnostic.reference.sign),
        (
            "worst_reference_exponent",
            "none" if diagnostic.reference.exponent is None else str(diagnostic.reference.exponent),
        ),
        ("worst_reference_class", diagnostic.reference.classification),
        ("worst_absolute_error", repr(diagnostic.absolute_error)),
        ("worst_ulp_distance", str(diagnostic.ulp_distance)),
        ("finite_values", str(diagnostic.finite_values)),
        ("exact_mismatches", str(diagnostic.exact_mismatches)),
        ("one_ulp_mismatches", str(diagnostic.one_ulp_mismatches)),
        ("absolute_threshold", repr(diagnostic.absolute_threshold)),
        ("absolute_mismatches", str(diagnostic.absolute_mismatches)),
    )
    message = f"{error} " + " ".join(f"{name}={value}" for name, value in fields)
    if len(message) > _MAX_NUMERICAL_WORST_PAIR_DIAGNOSTIC_CHARS:
        return NumericalFloorError(
            "numerical worst-pair diagnostic exceeded the closed 2048-character bound",
            output_name=error.output_name,
        )
    return NumericalFloorError(message, output_name=error.output_name)


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
    for repeat_index, actual in enumerate(actual_repeats):
        if actual.dtype.name != "bfloat16":
            raise TypeError(f"numerical repeat {repeat_index} must have BF16 dtype")
        if actual.shape != first.shape:
            raise ValueError("numerical repeats must have identical shapes")
    if expected.shape != first.shape:
        raise ValueError("numerical output and reference must have identical shapes")
    finite = np.isfinite(first) & np.isfinite(expected)
    nonfinite = sum(int(np.count_nonzero(~np.isfinite(actual))) for actual in actual_repeats)
    nonfinite += int(np.count_nonzero(~np.isfinite(expected)))
    finite_difference = np.abs(first[finite].astype(np.float32) - expected[finite].astype(np.float32))
    ulp = bfloat16_ulp_distance(first[finite], expected[finite])
    pairwise = []
    for left, right in itertools.combinations(range(len(actual_repeats)), 2):
        drift_finite = np.isfinite(actual_repeats[left]) & np.isfinite(actual_repeats[right])
        finite_drift = np.abs(
            actual_repeats[left][drift_finite].astype(np.float32)
            - actual_repeats[right][drift_finite].astype(np.float32)
        )
        drift_ulp = bfloat16_ulp_distance(actual_repeats[left][drift_finite], actual_repeats[right][drift_finite])
        pairwise.append(
            {
                "left_repeat_index": left,
                "right_repeat_index": right,
                "maximum_absolute_error": float(finite_drift.max(initial=0.0)),
                "mean_absolute_error": float(finite_drift.mean()) if finite_drift.size else 0.0,
                "maximum_ulp_distance": int(drift_ulp.max(initial=0)),
                "mean_ulp_distance": float(drift_ulp.mean()) if drift_ulp.size else 0.0,
            }
        )
    return {
        "maximum_absolute_error": float(finite_difference.max(initial=0.0)),
        "mean_absolute_error": float(finite_difference.mean()) if finite_difference.size else 0.0,
        "maximum_ulp_distance": int(ulp.max(initial=0)),
        "mean_ulp_distance": float(ulp.mean()) if ulp.size else 0.0,
        "nonfinite_values": nonfinite,
        "repeat_hashes": [hashlib.sha256(value.tobytes(order="C")).hexdigest() for value in actual_repeats],
        "pairwise_drift": pairwise,
    }


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
    audit_imported_local_modules(config)
    config.artifact_directory.mkdir(parents=True)
    (config.artifact_directory / "preflight.json").write_text(
        json.dumps(asdict(preflight), indent=2, sort_keys=True) + "\n"
    )
    generated_artifacts = compile_generated_candidates(config)
    audit_imported_local_modules(config)
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
        canonical_preparations = {
            backend.value: _prepare_canonical_cache(
                config,
                case.case_id,
                backend.value,
                generated_manifest,
                case_directory / "canonical" / backend.value,
            )
            for backend in BackendVariant
        }
        canonical_cache_root = case_directory / "canonical_cache_snapshot"
        canonical_cache_snapshot = _merge_canonical_target_snapshots(
            tuple(Path(canonical_preparations[backend.value]["canonical_cache_root"]) for backend in BackendVariant),
            canonical_cache_root,
        )
        canonical_cache_contract = _write_worker_cache_contract(
            case_directory / "canonical_cache_contract.json",
            canonical_cache_root,
            {backend.value: canonical_preparations[backend.value]["record"] for backend in BackendVariant},
        )
        case_result, trace_records = _run_profiled_case(
            config,
            case.case_id,
            generated_manifest,
            case_directory,
            canonical_cache_root,
            canonical_cache_contract,
        )
        compile_records = {
            backend.value: _run_cache_protocol(
                config,
                case.case_id,
                backend.value,
                generated_manifest,
                case_directory / "cache" / backend.value,
                Path(canonical_preparations[backend.value]["canonical_cache_root"]),
                canonical_preparations[backend.value]["record"],
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
                canonical_cache_root,
                canonical_cache_contract,
            )
            for backend in BackendVariant
        }
        if _persistent_cache_snapshot(_persistent_cache_files(canonical_cache_root)) != canonical_cache_snapshot:
            raise ValueError("canonical case cache snapshot changed during measurement-worker reads")
        validate_measurement_cache_consumers(
            compile_records,
            case_result,
            ncu_records,
            canonical_root_identity=canonical_cache_snapshot.root_identity,
        )
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
            case_directory / "canonical" / BackendVariant.ORDINARY_XLA.value / "canonical_dump",
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
                        "source_tree": config.source_tree,
                        "source_capsule_manifest_sha256": config.source_capsule_manifest_sha256,
                        "persistent_cache_identity": compiled["persistent_cache_identity"],
                        "canonical_cache_root_identity": canonical_cache_snapshot.root_identity,
                        "fresh_compile_serialized_executable_sha256": compiled[
                            "fresh_compile_serialized_executable_sha256"
                        ],
                        "fresh_compile_final_hlo_sha256": compiled["fresh_compile_final_hlo_sha256"],
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
    audit_imported_local_modules(config)
    bundle = {
        "schema": "shuttle.h100_contract_map_executed_bundle.v5",
        "architecture_status": ArchitectureStatus.NONCONFORMING.value,
        "source_sha": config.source_sha,
        "source_tree": config.source_tree,
        "source_capsule_manifest_sha256": config.source_capsule_manifest_sha256,
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
    cache_contract: Path | None = None,
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
        "--source-root",
        str(config.source_root),
        "--source-sha",
        config.source_sha,
        "--nvcc",
        str(config.tools.nvcc),
        "--require-jax-version",
        config.require_jax_version,
        "--cache-kind",
        cache_kind,
        *(("--cache-contract", str(cache_contract)) if cache_contract is not None else ()),
        *(
            (
                "--source-tree",
                config.source_tree,
                "--source-capsule-manifest",
                str(config.source_capsule_manifest),
                "--source-capsule-manifest-sha256",
                config.source_capsule_manifest_sha256,
            )
            if config.source_capsule_manifest is not None
            and config.source_tree is not None
            and config.source_capsule_manifest_sha256 is not None
            else ()
        ),
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


def _write_cache_snapshot(destination: Path, files: Sequence[tuple[str, bytes]]) -> _PersistentCacheSnapshot:
    if destination.exists():
        raise ValueError("persistent-cache snapshot destination must be fresh")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise ValueError("persistent-cache snapshot temporary destination must be fresh")
    temporary.mkdir(parents=True)
    for name, payload in files:
        if _CACHE_FILE_PATTERN.fullmatch(name) is None or Path(name).name != name:
            raise ValueError("persistent-cache snapshot has an invalid closed member name")
        (temporary / name).write_bytes(payload)
    written = _persistent_cache_files(temporary)
    if tuple(written) != tuple(files):
        raise ValueError("persistent-cache snapshot changed while being written")
    snapshot = _persistent_cache_snapshot(written)
    temporary.replace(destination)
    return snapshot


def _seal_canonical_target_snapshot(
    source: Path,
    destination: Path,
    record: Mapping[str, Any],
) -> _PersistentCacheSnapshot:
    cache_key = record["persistent_cache_key"]
    target, files = _persistent_cache_target(source, cache_key)
    source_snapshot = _persistent_cache_snapshot(files)
    if (
        target.serialized_executable_sha256 != record["persistent_cache_serialized_executable_sha256"]
        or target.compressed_entry_sha256 != record["persistent_cache_entry_sha256"]
        or source_snapshot.root_identity != record["persistent_cache_root_identity"]
        or source_snapshot.file_count != record["persistent_cache_file_count"]
        or source_snapshot.total_bytes != record["persistent_cache_total_bytes"]
    ):
        raise ValueError("canonical cache root changed after its compile worker")
    return _write_cache_snapshot(destination, files)


def _merge_canonical_target_snapshots(
    sources: Sequence[Path],
    destination: Path,
) -> _PersistentCacheSnapshot:
    if not sources:
        raise ValueError("canonical cache merge requires at least one backend snapshot")
    merged = dict(_persistent_cache_files(sources[0]))
    for source in sources[1:]:
        files = _persistent_cache_files(source)
        targets = tuple(
            (name, payload) for name, payload in files if _TARGET_CACHE_FILE_PATTERN.fullmatch(name) is not None
        )
        if len(targets) != 1:
            raise ValueError("canonical backend snapshot must contain exactly one target entry")
        for name, payload in targets:
            previous = merged.setdefault(name, payload)
            if previous != payload:
                raise ValueError("canonical target snapshots collide with different bytes")
    return _write_cache_snapshot(destination, tuple(sorted(merged.items())))


def _clone_cache_snapshot(source: Path, destination: Path) -> _PersistentCacheSnapshot:
    files = _persistent_cache_files(source)
    source_snapshot = _persistent_cache_snapshot(files)
    cloned = _write_cache_snapshot(destination, files)
    if cloned != source_snapshot:
        raise ValueError("cloned persistent-cache snapshot differs from its source")
    return cloned


def _write_worker_cache_contract(
    path: Path,
    snapshot_directory: Path,
    canonical_records: Mapping[str, Mapping[str, Any]],
) -> Path:
    files = _persistent_cache_files(snapshot_directory)
    snapshot = _persistent_cache_snapshot(files)
    backends = {}
    for backend, record in canonical_records.items():
        backends[backend] = {
            "cache_key": record["persistent_cache_key"],
            "compressed_entry_sha256": record["persistent_cache_entry_sha256"],
            "final_hlo_sha256": hashlib.sha256(str(record["final_hlo"]).encode()).hexdigest(),
            "serialized_executable_sha256": record["persistent_cache_serialized_executable_sha256"],
        }
    expected_targets = {f"{record['cache_key']}-cache" for record in backends.values()}
    actual_targets = {name for name, _ in files if _TARGET_CACHE_FILE_PATTERN.fullmatch(name) is not None}
    if len(expected_targets) != len(backends) or actual_targets != expected_targets:
        raise ValueError("canonical cache snapshot does not contain one distinct target for every backend")
    payload = {
        "backends": backends,
        "schema_version": 1,
        "snapshot": {
            "file_count": snapshot.file_count,
            "root_identity": snapshot.root_identity,
            "total_bytes": snapshot.total_bytes,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


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
    now: Callable[[], int] = time.monotonic_ns,
) -> dict[str, Any]:
    """Measure coordinator process spawn through worker compile completion."""
    started = now()
    completed = run(command, check=False, capture_output=True, text=True, env=environment)
    worker_exited = now()
    if completed.returncode != 0:
        raise RuntimeError(
            f"compile worker failed with {completed.returncode}: {command}: {completed.stdout}\n{completed.stderr}"
        )
    if not json_output.is_file():
        raise RuntimeError(f"compile worker succeeded without structured output: {json_output}")
    result = json.loads(json_output.read_text())
    compile_done = result.get("compile_done_monotonic_ns")
    if type(compile_done) is not int or not started <= compile_done <= worker_exited:
        raise RuntimeError("compile worker completion timestamp must fall between coordinator spawn and exit")
    compile_ns = compile_done - started
    if compile_ns <= 0:
        raise RuntimeError("compile worker elapsed time must be positive")
    result["compile_ns"] = compile_ns
    result["postcompile_ns"] = worker_exited - compile_done
    return result


def validated_cache_protocol_identity(
    compile_records: Sequence[Mapping[str, Any]],
    cold_records: Sequence[Mapping[str, Any]],
    hit_records: Sequence[Mapping[str, Any]],
    *,
    canonical_record: Mapping[str, Any] | None = None,
    case_id: str,
    backend: str,
    required_processes: int,
) -> str:
    """Bind fresh compile samples and cloned cache hits to one canonical entry."""
    if canonical_record is None:
        if not compile_records:
            raise ValueError("cache protocol requires a canonical record or fresh compile sample")
        canonical_record = compile_records[0]
    groups = {
        "canonical": (canonical_record,),
        "compile": compile_records,
        "cold": cold_records,
        "hit": hit_records,
    }
    for name, records in groups.items():
        if name != "canonical" and len(records) != required_processes:
            raise ValueError(f"cache protocol requires {required_processes} {name} roots")
    if re.fullmatch(r"[a-z0-9_]{1,128}", case_id) is None:
        raise ValueError("cache identity diagnostic requires a canonical case id")
    if re.fullmatch(r"[a-z0-9_]{1,64}", backend) is None:
        raise ValueError("cache identity diagnostic requires a canonical backend")

    partitions: dict[tuple[str, str], str] = {}
    roots = []
    for phase, records in groups.items():
        for index, record in enumerate(records):
            compression = record.get("persistent_cache_compression")
            if compression != _CACHE_COMPRESSION:
                raise ValueError(f"cache protocol {phase}[{index}] has an invalid compression contract")
            cache_key = record.get("persistent_cache_key")
            if not isinstance(cache_key, str) or _TARGET_CACHE_FILE_PATTERN.fullmatch(f"{cache_key}-cache") is None:
                raise ValueError(f"cache protocol {phase}[{index}] has an invalid target cache key")
            executable_identity = record.get("persistent_cache_serialized_executable_sha256")
            if not isinstance(executable_identity, str) or re.fullmatch(r"[0-9a-f]{64}", executable_identity) is None:
                raise ValueError(f"cache protocol {phase}[{index}] has an invalid serialized executable identity")
            identity = record.get("persistent_cache_identity")
            if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
                raise ValueError(f"cache protocol {phase}[{index}] has an invalid content identity")
            expected_identity = hashlib.sha256(cache_key.encode() + bytes.fromhex(executable_identity)).hexdigest()
            if identity != expected_identity:
                raise ValueError(f"cache protocol {phase}[{index}] has an inconsistent content identity")
            entry_identity = record.get("persistent_cache_entry_sha256")
            root_identity = record.get("persistent_cache_root_identity")
            if not isinstance(entry_identity, str) or re.fullmatch(r"[0-9a-f]{64}", entry_identity) is None:
                raise ValueError(f"cache protocol {phase}[{index}] has an invalid compressed entry identity")
            if not isinstance(root_identity, str) or re.fullmatch(r"[0-9a-f]{64}", root_identity) is None:
                raise ValueError(f"cache protocol {phase}[{index}] has an invalid root identity")
            compile_time = record.get("persistent_cache_compile_time")
            if type(compile_time) is not int or not 0 <= compile_time <= 2**32 - 1:
                raise ValueError(f"cache protocol {phase}[{index}] has an invalid cached compile time")
            file_count = record.get("persistent_cache_file_count")
            total_bytes = record.get("persistent_cache_total_bytes")
            if type(file_count) is not int or not 0 < file_count <= 2**63 - 1:
                raise ValueError(f"cache protocol {phase}[{index}] has an invalid file count")
            if type(total_bytes) is not int or not 0 < total_bytes <= 2**63 - 1:
                raise ValueError(f"cache protocol {phase}[{index}] has an invalid byte total")
            final_hlo = record.get("final_hlo")
            if not isinstance(final_hlo, str) or not final_hlo.strip():
                raise ValueError(f"cache protocol {phase}[{index}] has an invalid final HLO")
            events = record.get("persistent_cache_events")
            expected_events = dict(
                zip(
                    _CACHE_EVENT_NAMES,
                    (1, 0, 1) if phase in {"canonical", "compile"} else (1, 1, 0),
                    strict=True,
                )
            )
            if events != expected_events:
                raise ValueError(f"cache protocol {phase}[{index}] has invalid public cache events")
            partition = partitions.setdefault((cache_key, executable_identity), f"class_{len(partitions)}")
            if phase != "canonical":
                roots.append(
                    (
                        phase,
                        index,
                        partition,
                        file_count,
                        total_bytes,
                        hashlib.sha256(final_hlo.encode()).hexdigest(),
                    )
                )
    canonical_key = str(canonical_record["persistent_cache_key"])
    canonical_executable = str(canonical_record["persistent_cache_serialized_executable_sha256"])
    canonical_entry = str(canonical_record["persistent_cache_entry_sha256"])
    canonical_root = str(canonical_record["persistent_cache_root_identity"])
    keys = {str(record["persistent_cache_key"]) for records in groups.values() for record in records}
    cached_executables = {
        str(record["persistent_cache_serialized_executable_sha256"])
        for phase in ("cold", "hit")
        for record in groups[phase]
    }
    cached_entries = {
        str(record["persistent_cache_entry_sha256"]) for phase in ("cold", "hit") for record in groups[phase]
    }
    cached_roots = {
        str(record["persistent_cache_root_identity"]) for phase in ("cold", "hit") for record in groups[phase]
    }
    if (
        keys != {canonical_key}
        or cached_executables != {canonical_executable}
        or cached_entries != {canonical_entry}
        or cached_roots != {canonical_root}
    ):
        classes = [
            (label, cache_key.removeprefix("jit_step-"), executable_identity)
            for (cache_key, executable_identity), label in partitions.items()
        ]
        diagnostic = {
            "backend": backend,
            "case_id": case_id,
            "class_fields": _CACHE_DIAGNOSTIC_CLASS_FIELDS,
            "classes": classes,
            "canonical_equality_partition": partitions[(canonical_key, canonical_executable)],
            "expected_cached_equality_partitions": 1,
            "fresh_compile_equality_partitions": len(
                {
                    (
                        str(record["persistent_cache_key"]),
                        str(record["persistent_cache_serialized_executable_sha256"]),
                    )
                    for record in compile_records
                }
            ),
            "observed_equality_partitions": len(partitions),
            "root_fields": _CACHE_DIAGNOSTIC_ROOT_FIELDS,
            "roots": roots,
            "schema_version": 3,
        }
        serialized = json.dumps(diagnostic, sort_keys=True, separators=(",", ":"))
        message = (
            "all roots must share one target key and every cloned cache consumer must use the canonical executable "
            f"diagnostic={serialized}"
        )
        if len(message) > _MAX_CACHE_IDENTITY_DIAGNOSTIC_CHARS:
            raise AssertionError("cache identity diagnostic exceeds its reviewed bound")
        raise ValueError(message)
    return hashlib.sha256(canonical_key.encode() + bytes.fromhex(canonical_executable)).hexdigest()


def validated_executable_hlo(
    backend: str,
    *,
    case_worker_hlo: str,
    cache_protocol: Mapping[str, Any],
    profile_worker_hlo: str,
) -> str:
    """Bind every canonical-cache consumer to the setup worker's exact executable HLO."""
    compile_records = tuple(cache_protocol["compile"])
    if not compile_records:
        raise ValueError(f"{backend} executable evidence has no compile worker")
    authoritative = str(cache_protocol.get("canonical", compile_records[0])["final_hlo"])
    observed = [case_worker_hlo, profile_worker_hlo]
    for group in ("cold", "hit"):
        observed.extend(str(record["final_hlo"]) for record in cache_protocol[group])
    if not authoritative.strip() or any(value != authoritative for value in observed):
        raise ValueError(f"{backend} canonical final HLO differs across cache, timing, or profile workers")
    return authoritative


def validate_measurement_cache_consumers(
    cache_protocols: Mapping[str, Mapping[str, Any]],
    case_worker: Mapping[str, Any],
    profile_workers: Mapping[str, NcuProfileEvidence],
    *,
    canonical_root_identity: str,
) -> None:
    expected_backends = tuple(cache_protocols)
    case_evidence = case_worker.get("persistent_cache")
    if not isinstance(case_evidence, dict) or set(case_evidence) != set(expected_backends):
        raise ValueError("case worker cache evidence does not cover exactly every backend")
    expected_events = dict(zip(_CACHE_EVENT_NAMES, (1, 1, 0), strict=True))
    for backend, protocol in cache_protocols.items():
        canonical = protocol["canonical"]
        for role, evidence in (
            ("case", case_evidence[backend]),
            ("profile", profile_workers[backend].persistent_cache),
        ):
            if not isinstance(evidence, dict) or set(evidence) != {
                "persistent_cache_entry_sha256",
                "persistent_cache_events",
                "persistent_cache_key",
                "persistent_cache_root_identity",
                "persistent_cache_serialized_executable_sha256",
            }:
                raise ValueError(f"{role} worker has malformed cache evidence for {backend}")
            if (
                evidence["persistent_cache_events"] != expected_events
                or evidence["persistent_cache_entry_sha256"] != canonical["persistent_cache_entry_sha256"]
                or evidence["persistent_cache_key"] != canonical["persistent_cache_key"]
                or evidence["persistent_cache_serialized_executable_sha256"]
                != canonical["persistent_cache_serialized_executable_sha256"]
                or evidence["persistent_cache_root_identity"] != canonical_root_identity
            ):
                raise ValueError(f"{role} worker did not execute the canonical cached executable for {backend}")


def _run_profiled_case(
    config: RunnerConfig,
    case_id: str,
    generated_manifest: Path,
    directory: Path,
    cache_source: Path,
    cache_contract: Path,
) -> tuple[dict[str, Any], tuple[TraceRange, ...]]:
    result_path = directory / "case_result.json"
    worker = _worker_base_command(
        config,
        worker=WorkerMode.CASE,
        case_id=case_id,
        backend="all",
        generated_manifest=generated_manifest,
        json_output=result_path,
        cache_contract=cache_contract,
    )
    worker_cache = directory / "case_worker_cache"
    source_snapshot = _persistent_cache_snapshot(_persistent_cache_files(cache_source))
    if _clone_cache_snapshot(cache_source, worker_cache) != source_snapshot:
        raise ValueError("case worker cache clone differs from its canonical source")
    report_base = directory / "steady_trace"
    command = (
        str(config.tools.nsys),
        "profile",
        "--force-overwrite=true",
        *_NSYS_PROFILE_ARGS,
        "--output",
        str(report_base),
        *worker,
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_worker_environment(directory / "xla_dump", worker_cache),
    )
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"Nsight Systems case worker failed: {command}: {completed.stdout}\n{completed.stderr}")
    if _persistent_cache_snapshot(_persistent_cache_files(worker_cache)) != source_snapshot:
        raise ValueError("case worker changed its cloned canonical cache snapshot")
    report = report_base.with_suffix(".nsys-rep")
    sqlite_path = directory / "steady_trace.sqlite"
    export = (
        str(config.tools.nsys),
        "export",
        "--force-overwrite=true",
        *_NSYS_EXPORT_ARGS,
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
    return result, parse_nsys_sqlite(sqlite_path, expected_ranges, report_path=report)


def _prepare_canonical_cache(
    config: RunnerConfig,
    case_id: str,
    backend: str,
    generated_manifest: Path,
    directory: Path,
) -> dict[str, Any]:
    directory.mkdir(parents=True)
    populated_root = directory / "populated_root"
    result_path = directory / "canonical.json"
    command = _worker_base_command(
        config,
        worker=WorkerMode.COMPILE,
        case_id=case_id,
        backend=backend,
        generated_manifest=generated_manifest,
        json_output=result_path,
        cache_kind="canonical",
    )
    record = _run_worker_command(
        command,
        environment=_worker_environment(directory / "canonical_dump", populated_root),
        json_output=result_path,
    )
    expected_events = dict(zip(_CACHE_EVENT_NAMES, (1, 0, 1), strict=True))
    if record.get("persistent_cache_events") != expected_events:
        raise ValueError(f"canonical cache preparation did not prove one public miss for {case_id}/{backend}")
    sealed_root = directory / "sealed_root"
    snapshot = _seal_canonical_target_snapshot(populated_root, sealed_root, record)
    return {
        "canonical_cache_root": str(sealed_root),
        "canonical_cache_root_identity": snapshot.root_identity,
        "record": record,
    }


def _run_cache_protocol(
    config: RunnerConfig,
    case_id: str,
    backend: str,
    generated_manifest: Path,
    directory: Path,
    canonical_cache_root: Path,
    canonical_record: Mapping[str, Any],
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
    canonical_root = canonical_cache_root
    canonical_snapshot = _persistent_cache_snapshot(_persistent_cache_files(canonical_root))
    if canonical_snapshot.root_identity != canonical_record.get("persistent_cache_root_identity"):
        raise ValueError(f"canonical cache root identity differs for {case_id}/{backend}")
    cache_contract = _write_worker_cache_contract(
        directory / "canonical_cache_contract.json",
        canonical_root,
        {backend: canonical_record},
    )
    cold_records = []
    hit_records = []
    for index in range(protocol.persistent_cache_cold_processes):
        root = directory / "paired_roots" / str(index)
        cloned = _clone_cache_snapshot(canonical_root, root)
        if cloned != canonical_snapshot:
            raise ValueError(f"persistent cache clone differs for {case_id}/{backend}")
        cold_result = directory / f"cold_{index}.json"
        cold_command = _worker_base_command(
            config,
            worker=WorkerMode.COMPILE,
            case_id=case_id,
            backend=backend,
            generated_manifest=generated_manifest,
            json_output=cold_result,
            cache_kind="cold",
            cache_contract=cache_contract,
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
            cache_contract=cache_contract,
        )
        hit = run_timed_compile_worker_command(
            hit_command,
            environment=_worker_environment(directory / f"hit_dump_{index}", root),
            json_output=hit_result,
        )
        if cold["persistent_cache_root_identity"] != hit["persistent_cache_root_identity"]:
            raise ValueError(f"persistent cache root changed between cold and hit for {case_id}/{backend}")
        if cold["persistent_cache_root_identity"] != canonical_snapshot.root_identity:
            raise ValueError(f"persistent cache clone changed from its canonical snapshot for {case_id}/{backend}")
        cold_records.append(cold)
        hit_records.append(hit)
    if _persistent_cache_snapshot(_persistent_cache_files(canonical_root)) != canonical_snapshot:
        raise ValueError(f"canonical persistent cache snapshot changed during cloned reads for {case_id}/{backend}")
    identity = validated_cache_protocol_identity(
        compile_records,
        cold_records,
        hit_records,
        canonical_record=canonical_record,
        case_id=case_id,
        backend=backend,
        required_processes=protocol.compile_processes,
    )
    return {
        "canonical": canonical_record,
        "canonical_cache_root": str(canonical_root),
        "canonical_cache_root_identity": canonical_snapshot.root_identity,
        "compile": compile_records,
        "cold": cold_records,
        "fresh_compile_final_hlo_sha256": [
            hashlib.sha256(str(record["final_hlo"]).encode()).hexdigest() for record in compile_records
        ],
        "fresh_compile_serialized_executable_sha256": [
            record["persistent_cache_serialized_executable_sha256"] for record in compile_records
        ],
        "hit": hit_records,
        "persistent_cache_identity": identity,
    }


def _run_ncu_profile(
    config: RunnerConfig,
    case_id: str,
    backend: str,
    generated_manifest: Path,
    directory: Path,
    cache_source: Path,
    cache_contract: Path,
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
        cache_contract=cache_contract,
    )
    worker_cache = directory / "profile_worker_cache"
    source_snapshot = _persistent_cache_snapshot(_persistent_cache_files(cache_source))
    if _clone_cache_snapshot(cache_source, worker_cache) != source_snapshot:
        raise ValueError("profile worker cache clone differs from its canonical source")
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
        env=_worker_environment(directory / "xla_dump", worker_cache),
    )
    if completed.returncode != 0 or not result.is_file() or not csv_path.is_file() or not report_path.is_file():
        raise RuntimeError(f"Nsight Compute worker failed: {command}: {completed.stdout}\n{completed.stderr}")
    if _persistent_cache_snapshot(_persistent_cache_files(worker_cache)) != source_snapshot:
        raise ValueError("profile worker changed its cloned canonical cache snapshot")
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
    if not sass_source_path.is_file():
        raise RuntimeError("Nsight Compute produced no public SASS/source export")
    worker_result = json.loads(result.read_text())
    metrics = parse_ncu_metrics(csv_path)
    _parse_ncu_sass_file(sass_source_path, tuple(metric.name for metric in metrics))
    return NcuProfileEvidence(
        metrics=metrics,
        report_path=str(report_path),
        report_sha256=file_sha256(report_path),
        sass_source_path=str(sass_source_path),
        sass_source_sha256=file_sha256(sass_source_path),
        persistent_cache=dict(worker_result["persistent_cache"]),
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
            "device_to_host_count": 0,
            "device_to_host_bytes": 0,
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
            copies["device_to_host_count"] += trace.device_to_host_count
            copies["device_to_host_bytes"] += trace.device_to_host_bytes
            copies["unexpected_copy_count"] += (
                trace.unexpected_copy_count
                + trace.device_to_device_count
                + trace.host_to_device_count
                + trace.device_to_host_count
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
    shared_library_path = Path(artifact.shared_library_path)
    if file_sha256(shared_library_path) != artifact.shared_library_sha256:
        raise ValueError("generated shared-library content changed before evidence collection")
    loaded_image_sass_path = Path(artifact.loaded_image_sass_path)
    if file_sha256(loaded_image_sass_path) != artifact.loaded_image_sass_sha256:
        raise ValueError("loaded shared-library SASS content changed before evidence collection")
    validate_cuda_sass_kernel_topology(loaded_image_sass_path.read_text(), candidate.generated.kernel_names)
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
    sass_kernels = parse_ncu_sass(
        Path(profile.sass_source_path).read_text(), tuple(metric.name for metric in profile.metrics)
    )
    if any(
        instruction.mnemonic.split(".", maxsplit=1)[0] in {"LDL", "STL"}
        for kernel in sass_kernels
        for instruction in kernel.instructions
    ):
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
    parser.add_argument("--source-tree")
    parser.add_argument("--source-capsule-manifest", type=Path)
    parser.add_argument("--source-capsule-manifest-sha256")
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
    parser.add_argument("--cache-contract", type=Path)
    parser.add_argument("--cache-kind", choices=("none", "canonical", "compile", "cold", "hit"), default="none")
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
            source_tree=args.source_tree,
            source_capsule_manifest=(
                args.source_capsule_manifest.resolve() if args.source_capsule_manifest is not None else None
            ),
            source_capsule_manifest_sha256=args.source_capsule_manifest_sha256,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
