# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from importlib.resources import files

import pytest

import levanter.kernels.deepep
from levanter.kernels.deepep.availability import (
    DEEPEP_CACHE_ENV,
    DEEPEP_CUDA_ARCH_ENV,
    DEEPEP_SRC_ENV,
    TRANSPORT_REQUIRED_FILES,
    deepep_cache_root,
    deepep_cuda_arch_flag,
    deepep_source_root,
)


_DEEPEP_FFI_SOURCES = (
    "deepep_layout_ffi.cu",
    "deepep_transport_ffi.cu",
    "deepep_transport_pyext.cc",
)


def _write_deepep_skeleton(root: Path, files: tuple[str, ...] = TRANSPORT_REQUIRED_FILES) -> None:
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// test\n")


def test_deepep_source_root_accepts_required_source_tree(tmp_path, monkeypatch):
    _write_deepep_skeleton(tmp_path)
    monkeypatch.setenv(DEEPEP_SRC_ENV, str(tmp_path))

    assert deepep_source_root(required_files=TRANSPORT_REQUIRED_FILES, purpose="test") == tmp_path.resolve()


def test_deepep_source_root_reports_missing_files(tmp_path, monkeypatch):
    _write_deepep_skeleton(tmp_path, files=("csrc/kernels/layout.cu",))
    monkeypatch.setenv(DEEPEP_SRC_ENV, str(tmp_path))

    with pytest.raises(RuntimeError, match="runtime.cu"):
        deepep_source_root(required_files=TRANSPORT_REQUIRED_FILES, purpose="test")


def test_deepep_cuda_arch_flag_supports_b200(monkeypatch):
    monkeypatch.setenv(DEEPEP_CUDA_ARCH_ENV, "sm_100")

    assert deepep_cuda_arch_flag() == ["-gencode=arch=compute_100,code=sm_100"]


def test_deepep_cuda_arch_flag_rejects_unknown_arch(monkeypatch):
    monkeypatch.setenv(DEEPEP_CUDA_ARCH_ENV, "compute_100")

    with pytest.raises(RuntimeError, match=DEEPEP_CUDA_ARCH_ENV):
        deepep_cuda_arch_flag()


def test_deepep_cache_root_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv(DEEPEP_CACHE_ENV, str(tmp_path))

    assert deepep_cache_root("deepep_transport_ffi") == tmp_path.resolve() / "deepep_transport_ffi"


def test_deepep_ffi_sources_are_package_resources():
    csrc = files(levanter.kernels.deepep) / "csrc"

    for source in _DEEPEP_FFI_SOURCES:
        assert (csrc / source).is_file()
