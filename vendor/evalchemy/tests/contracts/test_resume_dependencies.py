"""Installed dependency fingerprints survive vendored import-path changes."""

import importlib
import sys

import pytest
from eval.contracts.finestore_resume import FineStoreResumeManager
from eval.resume.fingerprint import RunFingerprint
from eval.resume.wiring import installed_dependency_versions
from finestore.store import DataStore


@pytest.fixture(autouse=True)
def fixture_imports():
    yield
    for name in ("resume_fixture_parent", "resume_fixture_core"):
        sys.modules.pop(name, None)


def write_distribution(root, name, version, owned=()):
    metadata = root / f"{name.replace('-', '_')}-{version}.dist-info"
    metadata.mkdir(parents=True)
    path = metadata / "METADATA"
    path.write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    (metadata / "RECORD").write_text("".join(f"{item},,\n" for item in [path.relative_to(root), *owned]))
    return path


def test_vendor_import_and_dependency_updates_preserve_logical_resume(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    installed = write_distribution(site, "resume-control-core", "1.0")
    parent = site / "resume_fixture_parent"
    vendor = parent / "_vendor"
    core = write_distribution(vendor, "resume-control-core", "99.0")
    extra = write_distribution(vendor, "resume-vendor-only", "99.0")
    write_distribution(site, "resume-control-parent", "1.0", [core.relative_to(site), extra.relative_to(site)])
    (parent / "__init__.py").write_text(
        "from pathlib import Path\nimport sys\nsys.path.append(str(Path(__file__).parent / '_vendor'))\n"
    )
    monkeypatch.setattr(sys, "path", [str(site), *sys.path])
    monkeypatch.delitem(sys.modules, "resume_fixture_parent", raising=False)

    def fingerprint():
        return RunFingerprint(inputs={"dependencies": installed_dependency_versions()})

    with DataStore(str(tmp_path / "archive")) as store:
        assert FineStoreResumeManager(store, "run", "task", fingerprint()).decide() == "fresh"
        importlib.import_module("resume_fixture_parent")
        assert FineStoreResumeManager(store, "run", "task", fingerprint()).decide() == "resume"
        installed.write_text("Metadata-Version: 2.1\nName: resume-control-core\nVersion: 2.0\n")
        assert FineStoreResumeManager(store, "run", "task", fingerprint()).decide() == "resume"


def test_independent_overlay_matches_import_precedence(tmp_path, monkeypatch):
    site, overlay = tmp_path / "site-packages", tmp_path / "overlay"
    write_distribution(site, "resume-control-core", "1.0")
    write_distribution(overlay, "resume-control-core", "2.0")
    (site / "resume_fixture_core.py").write_text("__version__ = '1.0'\n")
    (overlay / "resume_fixture_core.py").write_text("__version__ = '2.0'\n")
    monkeypatch.setattr(sys, "path", [str(overlay), str(site), *sys.path])
    monkeypatch.delitem(sys.modules, "resume_fixture_core", raising=False)
    imported = importlib.import_module("resume_fixture_core")
    assert installed_dependency_versions()["resume-control-core"] == imported.__version__ == "2.0"
