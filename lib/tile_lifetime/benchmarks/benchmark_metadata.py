# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproducibility metadata shared by standalone GPU benchmarks."""

import hashlib
import json
import os
import shlex
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "pci.bus_id",
    "driver_version",
    "pstate",
    "persistence_mode",
    "power.draw",
    "power.limit",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.applications.graphics",
    "clocks.applications.memory",
    "clocks.max.graphics",
    "clocks.max.sm",
    "clocks.max.memory",
    "temperature.gpu",
)

RECORDED_ENVIRONMENT_VARIABLES = (
    "CUDA_HOME",
    "CUDA_CCCL_INCLUDE",
    "DEEPEP_BUILD_INTRANODE_ONLY",
    "DEEPEP_DISABLE_NVSHMEM",
    "MAX_JOBS",
    "NCCL_DEBUG",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "TORCH_CUDA_ARCH_LIST",
)


def file_sha256(path: Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value using stable key ordering and separators."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def framed_tensor_sha256(dtype: str, shape: tuple[int, ...], payload: bytes) -> str:
    """Hash a tensor dtype, shape, and C-order byte payload."""
    digest = hashlib.sha256()
    encoded_dtype = dtype.encode()
    digest.update(struct.pack("<Q", len(encoded_dtype)))
    digest.update(encoded_dtype)
    digest.update(struct.pack("<Q", len(shape)))
    for dimension in shape:
        digest.update(struct.pack("<Q", dimension))
    digest.update(payload)
    return digest.hexdigest()


def command_record() -> dict[str, Any]:
    """Record the exact Python command and selected non-secret environment."""
    return {
        "argv": sys.argv,
        "shell": shlex.join(sys.argv),
        "working_directory": str(Path.cwd()),
        "environment": {name: os.environ[name] for name in RECORDED_ENVIRONMENT_VARIABLES if name in os.environ},
    }


def command_output(arguments: list[str]) -> dict[str, Any]:
    """Run a metadata command without hiding unsupported-tool errors."""
    completed = subprocess.run(arguments, check=False, capture_output=True, text=True)
    return {
        "arguments": arguments,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def nvidia_smi_snapshot() -> dict[str, Any]:
    """Capture topology and per-GPU clock, power, and identity fields."""
    field_records: dict[str, Any] = {}
    gpu_count = 0
    for field in GPU_QUERY_FIELDS:
        result = command_output(
            [
                "nvidia-smi",
                f"--query-gpu={field}",
                "--format=csv,noheader,nounits",
            ]
        )
        values = result["stdout"].splitlines() if result["returncode"] == 0 else []
        gpu_count = max(gpu_count, len(values))
        field_records[field] = {
            "values": values,
            "returncode": result["returncode"],
            "stderr": result["stderr"],
        }
    gpus = []
    for gpu_index in range(gpu_count):
        gpus.append(
            {
                field: record["values"][gpu_index] if gpu_index < len(record["values"]) else None
                for field, record in field_records.items()
            }
        )
    return {
        "gpus": gpus,
        "field_errors": {
            field: {"returncode": record["returncode"], "stderr": record["stderr"]}
            for field, record in field_records.items()
            if record["returncode"] != 0
        },
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
    }


def toolchain_snapshot(nvcc: str) -> dict[str, Any]:
    """Capture host and CUDA compiler versions used by a benchmark."""
    return {
        "python": sys.version,
        "platform": command_output(["uname", "-a"]),
        "nvcc": command_output([nvcc, "--version"]),
        "ptxas": command_output([str(Path(nvcc).with_name("ptxas")), "--version"]),
    }
