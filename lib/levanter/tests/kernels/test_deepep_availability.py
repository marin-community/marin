# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from levanter.kernels.deepep import transport_ffi
from levanter.kernels.deepep.availability import (
    DEEPEP_CUDA_ARCH_ENV,
    DEEPEP_KNOWN_GOOD_COMMIT,
    DEEPEP_SRC_ENV,
    TRANSPORT_REQUIRED_FILES,
    deepep_layout_source,
    deepep_preflight_status,
    deepep_source_root,
)


def _write(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("// test source\n")
    return path


def _write_transport_sources(root: Path) -> None:
    for relative in TRANSPORT_REQUIRED_FILES:
        _write(root, relative)


def test_deepep_layout_source_accepts_legacy_layout_path(tmp_path: Path) -> None:
    root = tmp_path / "DeepEP"
    layout_source = _write(root, "csrc/kernels/legacy/layout.cu")

    assert deepep_layout_source(root) == layout_source


def test_deepep_source_root_accepts_legacy_layout_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "DeepEP"
    _write_transport_sources(root)
    _write(root, "csrc/kernels/legacy/layout.cu")
    monkeypatch.setenv(DEEPEP_SRC_ENV, str(root))

    assert (
        deepep_source_root(
            required_files=TRANSPORT_REQUIRED_FILES,
            purpose="test",
            requires_layout_source=True,
        )
        == root.resolve()
    )


def test_deepep_preflight_reports_missing_layout_source_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "DeepEP"
    _write_transport_sources(root)
    monkeypatch.setenv(DEEPEP_SRC_ENV, str(root))
    monkeypatch.setenv(DEEPEP_CUDA_ARCH_ENV, "sm_100")

    status = deepep_preflight_status(required_files=TRANSPORT_REQUIRED_FILES)

    assert any("layout source" in error and "csrc/kernels/legacy/layout.cu" in error for error in status.errors)


def test_deepep_preflight_warns_for_unverified_source_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "DeepEP"
    _write_transport_sources(root)
    _write(root, "csrc/kernels/legacy/layout.cu")
    monkeypatch.setenv(DEEPEP_SRC_ENV, str(root))
    monkeypatch.setenv(DEEPEP_CUDA_ARCH_ENV, "sm_100")

    status = deepep_preflight_status(required_files=TRANSPORT_REQUIRED_FILES)

    assert status.source_revision is None
    assert any(DEEPEP_KNOWN_GOOD_COMMIT in warning for warning in status.warnings)


def test_prepare_intranode_source_compiles_generated_launch_patch(tmp_path: Path) -> None:
    root = tmp_path / "DeepEP"
    intranode_source = root / "csrc" / "kernels" / "intranode.cu"
    intranode_source.parent.mkdir(parents=True, exist_ok=True)
    intranode_source.write_text(
        "\nvoid dispatch() {\n"
        "    constexpr int kNumThreads = 768;\n"
        "    auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \\\n"
        "    SET_SHARED_MEMORY_FOR_TMA(kernel); \\\n"
        "}\n"
        "\nvoid combine() {\n"
        "    auto kernel = combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>; \\\n"
        "    SET_SHARED_MEMORY_FOR_TMA(kernel); \\\n"
        "}\n"
    )

    prepared = transport_ffi._prepare_intranode_source(tmp_path / "build", root)

    assert prepared == tmp_path / "build" / "generated" / "intranode.cu"
    prepared_text = prepared.read_text()
    assert "SET_SHARED_MEMORY_FOR_TMA((dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>));" in prepared_text
    assert "SET_SHARED_MEMORY_FOR_TMA((combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>));" in prepared_text
    assert "SET_SHARED_MEMORY_FOR_TMA(kernel);" not in prepared_text
