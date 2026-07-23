# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

BUILD_PACKAGE_PATH = Path(__file__).parents[1] / "rust" / "build_package.py"


def _load_build_package():
    spec = importlib.util.spec_from_file_location("iris_native_build_package", BUILD_PACKAGE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nightly_version_advances_latest_stable(monkeypatch):
    build_package = _load_build_package()
    monkeypatch.setattr(build_package, "_read_declared_version", lambda: "0.1.0")
    monkeypatch.setattr(build_package, "_latest_pypi_stable", lambda: "0.1.2")

    resolved = build_package.resolve_version("nightly", None)

    base, timestamp = resolved.split("-dev.")
    assert base == "0.1.3"
    assert len(timestamp) == 12
    assert timestamp.isdigit()


def test_build_version_is_written_to_all_manifests(tmp_path, monkeypatch):
    build_package = _load_build_package()
    paths = tuple(tmp_path / name for name in ("pyproject.toml", "Cargo.toml", "pyext-Cargo.toml"))
    for path in paths:
        path.write_text('[package]\nversion = "0.1.0"\n')
    monkeypatch.setattr(build_package, "VERSION_PATHS", paths)

    build_package._write_versions("0.1.1-dev.202607230600")

    assert [path.read_text() for path in paths] == [
        '[package]\nversion = "0.1.1-dev.202607230600"\n',
        '[package]\nversion = "0.1.1-dev.202607230600"\n',
        '[package]\nversion = "0.1.1-dev.202607230600"\n',
    ]
