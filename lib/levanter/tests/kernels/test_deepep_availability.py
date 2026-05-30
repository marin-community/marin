# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from levanter.kernels.deepep.availability import (
    DEEPEP_CUDA_ARCH_ENV,
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
