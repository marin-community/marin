# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve CUDA toolkit link paths from an explicitly selected compiler."""

from pathlib import Path


def cuda_toolkit_library_directories(nvcc: Path) -> tuple[Path, ...]:
    """Return existing library directories beside an NVCC installation."""
    if not nvcc.is_file():
        raise ValueError(f"CUDA compiler does not exist: {nvcc}")
    toolkit = nvcc.resolve().parent.parent
    return tuple(path for path in (toolkit / "lib64", toolkit / "lib") if path.is_dir())


def cuda_toolkit_shared_library(nvcc: Path, name: str) -> Path:
    """Resolve an unversioned or versioned CUDA shared library beside NVCC."""
    filename = f"lib{name}.so"
    directories = cuda_toolkit_library_directories(nvcc)
    for directory in directories:
        unversioned = directory / filename
        if unversioned.is_file():
            return unversioned
        versioned = sorted(directory.glob(f"{filename}.*"), reverse=True)
        if versioned:
            return versioned[0]
    searched = ", ".join(str(directory / filename) for directory in directories)
    raise ValueError(f"CUDA shared library {filename} does not exist beside {nvcc}; searched {searched}")


def cuda_toolkit_link_flags(nvcc: Path, *, runtime_search_path: bool) -> tuple[str, ...]:
    """Return NVCC flags for pip and system toolkit library layouts."""
    directories = cuda_toolkit_library_directories(nvcc)
    library_flags = tuple(flag for directory in directories for flag in ("-L", str(directory)))
    if not runtime_search_path:
        return library_flags
    runtime_flags = tuple(
        flag for directory in directories for flag in ("-Xlinker", "-rpath", "-Xlinker", str(directory))
    )
    return (*library_flags, *runtime_flags)
