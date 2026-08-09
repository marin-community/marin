# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tile_lifetime.cuda_toolchain import (
    cuda_toolkit_library_directories,
    cuda_toolkit_link_flags,
    cuda_toolkit_shared_library,
    cuda_toolkit_shared_library_link_flags,
)


def test_cuda_toolkit_link_flags_cover_pip_layout(tmp_path: Path) -> None:
    toolkit = tmp_path / "nvidia" / "cu13"
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    library = toolkit / "lib"
    library.mkdir()

    assert cuda_toolkit_library_directories(nvcc) == (library,)
    assert cuda_toolkit_link_flags(nvcc, runtime_search_path=False) == ("-L", str(library))
    assert cuda_toolkit_link_flags(nvcc, runtime_search_path=True) == (
        "-L",
        str(library),
        "-Xlinker",
        "-rpath",
        "-Xlinker",
        str(library),
    )


def test_cuda_toolkit_link_flags_cover_split_pip_wheels(tmp_path: Path) -> None:
    package_root = tmp_path / "site-packages" / "nvidia"
    nvcc = package_root / "cu13" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    runtime_library = package_root / "cuda_runtime" / "lib"
    runtime_library.mkdir(parents=True)
    cudart = runtime_library / "libcudart.so.13"
    cudart.touch()

    assert cuda_toolkit_library_directories(nvcc) == (runtime_library,)
    assert cuda_toolkit_shared_library(nvcc, "cudart") == cudart


def test_cuda_toolkit_link_flags_preserve_supplied_symlink_layout(tmp_path: Path) -> None:
    toolkit = tmp_path / "site-packages" / "nvidia" / "cu13"
    installed_nvcc = tmp_path / "site-packages" / "bin" / "nvcc"
    installed_nvcc.parent.mkdir(parents=True)
    installed_nvcc.touch()
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.symlink_to(installed_nvcc)
    library = toolkit / "lib" / "libcudart.so.13"
    library.parent.mkdir()
    library.touch()

    assert cuda_toolkit_shared_library(nvcc, "cudart") == library


def test_cuda_toolkit_link_flags_reject_missing_compiler(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CUDA compiler does not exist"):
        cuda_toolkit_link_flags(tmp_path / "bin" / "nvcc", runtime_search_path=False)


def test_cuda_toolkit_shared_library_resolves_versioned_pip_library(tmp_path: Path) -> None:
    toolkit = tmp_path / "nvidia" / "cu13"
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    older = toolkit / "lib" / "libcublas.so.9"
    versioned = toolkit / "lib" / "libcublas.so.13"
    versioned.parent.mkdir()
    older.touch()
    versioned.touch()

    assert cuda_toolkit_shared_library(nvcc, "cublas") == versioned


def test_cuda_toolkit_shared_library_resolves_versioned_runtime(tmp_path: Path) -> None:
    toolkit = tmp_path / "nvidia" / "cu13"
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    library_directory = toolkit / "lib"
    library_directory.mkdir()
    versioned = library_directory / "libcudart.so.13"
    versioned.touch()

    assert cuda_toolkit_shared_library(nvcc, "cudart") == versioned


def test_cuda_toolkit_shared_library_link_flags_wrap_exact_versioned_paths(tmp_path: Path) -> None:
    toolkit = tmp_path / "nvidia" / "cu13"
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    library_directory = toolkit / "lib"
    library_directory.mkdir()
    cublas = library_directory / "libcublas.so.13"
    cudart = library_directory / "libcudart.so.13"
    cublas.touch()
    cudart.touch()

    assert cuda_toolkit_shared_library_link_flags(nvcc, ("cublas", "cudart")) == (
        "-Xlinker",
        str(cublas),
        "-Xlinker",
        str(cudart),
    )


def test_cuda_toolkit_shared_library_prefers_unversioned_library(tmp_path: Path) -> None:
    toolkit = tmp_path / "cuda"
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    library = toolkit / "lib64" / "libcudart.so"
    library.parent.mkdir()
    library.touch()
    (library.parent / "libcudart.so.13").touch()

    assert cuda_toolkit_shared_library(nvcc, "cudart") == library


def test_cuda_toolkit_shared_library_rejects_missing_library(tmp_path: Path) -> None:
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    (tmp_path / "cuda" / "lib").mkdir()

    with pytest.raises(ValueError, match=r"libcublas\.so does not exist"):
        cuda_toolkit_shared_library(nvcc, "cublas")
