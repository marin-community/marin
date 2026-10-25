# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/ci/check_locked_security.py"


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    return tmp_path


def write_lock(root: Path, packages: list[tuple[str, str]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text('[project]\nname = "fixture"\nversion = "0.0.0"\nrequires-python = ">=3.12"\n')
    lock = root / "uv.lock"
    text = 'version = 1\nrevision = 3\nrequires-python = ">=3.12"\n'
    text += '\n[[package]]\nname = "fixture"\nversion = "0.0.0"\nsource = { virtual = "." }\n'
    if packages:
        # The affected entries are only in an optional extra, never selected on the CLI.
        text += "\n[package.optional-dependencies]\nnetwork = [\n"
        text += "".join(f'{{ name = "{name}" }},\n' for name, _ in packages)
        text += "]\n"
    for name, version in packages:
        text += (
            f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n'
            'source = { registry = "https://pypi.org/simple" }\n'
        )
    lock.write_text(text)
    subprocess.run(["git", "-C", str(root), "add", "uv.lock", "pyproject.toml"], check=True)
    return lock


def run_audit(root: Path) -> tuple[int, dict]:
    output = root / "report.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert output.exists(), result.stderr
    return result.returncode, json.loads(output.read_text())


@pytest.mark.parametrize(
    "name,version", [("brotli", "1.2.0"), ("brotlicffi", "1.2.0.2"), ("brotli", "1.2.1rc1"), ("litellm", "1.107.0")]
)
def test_audit_rejects_optional_packages_in_nested_lock(repository: Path, name: str, version: str):
    write_lock(repository, [])
    lock = write_lock(repository / "vendor/agent", [(name, version)])
    before = lock.read_bytes()
    code, report = run_audit(repository)
    assert code == 1
    assert report["lock_count"] == 2
    nested = next(row for row in report["locks"] if row["path"] == "vendor/agent/uv.lock")
    assert nested["findings"] and not nested["errors"]
    assert lock.read_bytes() == before


def test_audit_accepts_fixed_releases_and_empty_project(repository: Path):
    write_lock(repository, [])
    write_lock(repository / "vendor/agent", [("brotli", "1.2.1"), ("brotlicffi", "1.2.1.0")])
    code, report = run_audit(repository)
    assert code == 0
    assert report["passed"] is True
    assert report["lock_count"] == 2


def test_audit_reports_invalid_lock_and_still_checks_other_projects(repository: Path):
    lock = write_lock(repository, [])
    lock.write_text("not a TOML lock\n")
    write_lock(repository / "vendor/agent", [("brotli", "1.2.0")])
    code, report = run_audit(repository)
    assert code == 1
    assert any(row["errors"] for row in report["locks"])
    assert any(row["findings"] for row in report["locks"])


def test_audit_rejects_missing_workspace_manifest(repository: Path):
    write_lock(repository, [("brotli", "1.2.0")])
    (repository / "pyproject.toml").unlink()
    code, report = run_audit(repository)
    assert code == 1
    assert report["locks"][0]["errors"]
    assert report["locks"][0]["findings"]


def test_audit_rejects_empty_coverage(repository: Path):
    code, report = run_audit(repository)
    assert code == 1
    assert report["passed"] is False
    assert report["lock_count"] == 0
