# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect and validate the NCCL library loaded by the current process."""

from __future__ import annotations

import ctypes
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

NCCL_RUNTIME_VERSION_ENV = "GRUG_EXPECT_NCCL_RUNTIME_VERSION"
NCCL_LIBRARY_PATH_ENV = "GRUG_EXPECT_NCCL_LIBRARY_PATH"


@dataclass(frozen=True)
class NcclRuntime:
    version: int
    mapped_libraries: tuple[Path, ...]


def nccl_runtime() -> NcclRuntime:
    """Return the NCCL version and mapped library paths for this process."""
    library = ctypes.CDLL("libnccl.so.2")
    version = ctypes.c_int()
    result = library.ncclGetVersion(ctypes.byref(version))
    if result != 0:
        raise RuntimeError(f"ncclGetVersion failed with status {result}")

    mapped_libraries = tuple(
        sorted(
            {
                Path(line.rsplit(maxsplit=1)[-1])
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libnccl.so" in line and line.rsplit(maxsplit=1)[-1].startswith("/")
            }
        )
    )
    return NcclRuntime(version=version.value, mapped_libraries=mapped_libraries)


def validate_nccl_runtime(actual: NcclRuntime, expected: Mapping[str, str]) -> None:
    """Fail when the loaded NCCL runtime differs from explicit expectations."""
    expected_version = expected.get(NCCL_RUNTIME_VERSION_ENV)
    if expected_version is not None:
        try:
            parsed_version = int(expected_version)
        except ValueError as error:
            raise ValueError(f"{NCCL_RUNTIME_VERSION_ENV} must be an integer, got {expected_version!r}") from error
        if actual.version != parsed_version:
            raise RuntimeError(f"NCCL runtime version mismatch: expected {parsed_version}, got {actual.version}")

    expected_library = expected.get(NCCL_LIBRARY_PATH_ENV)
    if expected_library is not None:
        resolved_expected = Path(expected_library).resolve()
        resolved_mapped = {path.resolve() for path in actual.mapped_libraries}
        if resolved_mapped != {resolved_expected}:
            raise RuntimeError(
                f"NCCL runtime library mismatch: expected {resolved_expected}, "
                f"mapped {sorted(str(path) for path in resolved_mapped)}"
            )
