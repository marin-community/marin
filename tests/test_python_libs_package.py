# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/python_libs_package.py.

Focused on the PyPI gating logic: the script must skip a package only when
the exact (name, version) tuple is already on PyPI, not when the name has
ever been published. Regression coverage for #5867 (marin-finelog==0.99 was
silently skipped because marin-finelog==0.1.0 already existed on PyPI).
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "python_libs_package.py"


@pytest.fixture(scope="module")
def libs_module():
    spec = importlib.util.spec_from_file_location("python_libs_package", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def _fake_urlopen(version_map: dict[str, list[str] | None]):
    """Return a urlopen replacement that serves PyPI JSON from version_map.

    A None entry raises HTTPError(404), matching PyPI's response for an
    unregistered project. A list of strings is served as a /pypi/<name>/json
    payload with those release keys.
    """

    def fake(url: str, timeout: float = 0):  # noqa: ARG001
        name = url.rsplit("/", 2)[-2]
        versions = version_map.get(name, None)
        if versions is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
        payload = json.dumps({"releases": {v: [] for v in versions}}).encode()
        return io.BytesIO(payload)

    return fake


def test_pypi_has_version_missing_project(libs_module, monkeypatch):
    monkeypatch.setattr(libs_module.urllib.request, "urlopen", _fake_urlopen({}))
    assert libs_module._pypi_has_version("nonexistent-pkg", "0.99") is False


def test_pypi_has_version_other_versions_only(libs_module, monkeypatch):
    # The 5867 case: project exists but the requested version is not published.
    monkeypatch.setattr(
        libs_module.urllib.request,
        "urlopen",
        _fake_urlopen({"marin-finelog": ["0.1.0"]}),
    )
    assert libs_module._pypi_has_version("marin-finelog", "0.99") is False
    assert libs_module._pypi_has_version("marin-finelog", "0.1.0") is True


def test_pypi_has_version_propagates_non_404(libs_module, monkeypatch):
    def boom(url, timeout=0):  # noqa: ARG001
        raise urllib.error.HTTPError(url, 503, "Service Unavailable", hdrs=None, fp=None)

    monkeypatch.setattr(libs_module.urllib.request, "urlopen", boom)
    with pytest.raises(urllib.error.HTTPError):
        libs_module._pypi_has_version("marin-iris", "0.99")


def test_publish_pypi_uploads_missing_version(libs_module, monkeypatch, tmp_path):
    """Existing project at an older version still gets the new version uploaded."""
    monkeypatch.setenv("UV_PUBLISH_TOKEN", "fake-token")
    monkeypatch.setattr(libs_module, "DIST_DIR", tmp_path)
    # PACKAGES is a module-level dict; restrict the loop to a single entry so
    # this test stays hermetic to the script's package list.
    monkeypatch.setattr(libs_module, "PACKAGES", {"marin-finelog": {}})
    # finelog exists on PyPI at 0.1.0 only — the old code skipped, the fix uploads.
    monkeypatch.setattr(
        libs_module.urllib.request,
        "urlopen",
        _fake_urlopen({"marin-finelog": ["0.1.0"]}),
    )

    fake_wheel = tmp_path / "marin_finelog-0.99-py3-none-any.whl"
    fake_wheel.write_bytes(b"")
    fake_sdist = tmp_path / "marin_finelog-0.99.tar.gz"
    fake_sdist.write_bytes(b"")

    calls: list[list[str]] = []

    def fake_run(cmd, check):  # noqa: ARG001
        calls.append(cmd)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(libs_module.subprocess, "run", fake_run)

    libs_module.publish_pypi("0.99")

    assert len(calls) == 1
    assert calls[0][:2] == ["uv", "publish"]
    assert str(fake_wheel) in calls[0]
    assert str(fake_sdist) in calls[0]


def test_publish_pypi_skips_when_version_already_on_pypi(libs_module, monkeypatch, tmp_path):
    monkeypatch.setenv("UV_PUBLISH_TOKEN", "fake-token")
    monkeypatch.setattr(libs_module, "DIST_DIR", tmp_path)
    monkeypatch.setattr(libs_module, "PACKAGES", {"marin-finelog": {}})
    monkeypatch.setattr(
        libs_module.urllib.request,
        "urlopen",
        _fake_urlopen({"marin-finelog": ["0.1.0", "0.99"]}),
    )
    (tmp_path / "marin_finelog-0.99-py3-none-any.whl").write_bytes(b"")

    def fail_run(*args, **kwargs):
        raise AssertionError(f"subprocess.run should not be called when version is present: {args!r}")

    monkeypatch.setattr(libs_module.subprocess, "run", fail_run)

    libs_module.publish_pypi("0.99")
