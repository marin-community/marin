# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tile_lifetime.cuda_toolchain import cuda_toolkit_library_directories, cuda_toolkit_link_flags


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


def test_cuda_toolkit_link_flags_reject_missing_compiler(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CUDA compiler does not exist"):
        cuda_toolkit_link_flags(tmp_path / "bin" / "nvcc", runtime_search_path=False)
