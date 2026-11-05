# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the maintained-app release selector and CI gates."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.ci.marina_app_releases import AppPolicy, changed_apps, drift, release


def fake_uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: dict) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.jsonl"
    executable = bin_dir / "uv"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['FAKE_UV_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "if 'release-status' in args:\n"
        "    print(os.environ['FAKE_STATUS'])\n"
        "elif '--dry-run' in args:\n"
        "    print(json.dumps({'checks': ['package', 'backend_import'], 'file_count': 3}))\n"
        "elif '--base-version' in args:\n"
        "    sys.exit(17)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_UV_CALLS", str(calls))
    monkeypatch.setenv("FAKE_STATUS", json.dumps(status))
    return calls


def test_select_changed_app_and_shared_input_respects_manual_policy():
    policies = {
        "evaldash": AppPolicy("evaldash", True, ("tests/test_applets.py",), ("infra/marina/web",)),
        "echo": AppPolicy("echo", False, ("tests/test_applets.py",), ("infra/marina/web",)),
        "belay": AppPolicy("belay", True, ("tests/test_applets.py",), ()),
    }
    assert changed_apps({"infra/marina/web/nav.ts", "infra/marina/apps/belay/app.py"}, policies) == [
        "belay",
        "evaldash",
    ]
    assert changed_apps({".github/marina-app-releases.toml"}, policies) == []


def test_drift_names_unreleased_source_commit(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    app = tmp_path / "infra/marina/apps/evaldash"
    app.mkdir(parents=True)
    (app / "app.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "Initial app"], cwd=tmp_path, check=True)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, text=True, capture_output=True
    ).stdout.strip()
    (app / "app.py").write_text("value = 2\n")
    subprocess.run(["git", "commit", "-qam", "Fix dashboard score"], cwd=tmp_path, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = {
        "state": "different",
        "version": 6,
        "released": {"commit": base, "path": "infra/marina/apps/evaldash", "inputs": {}},
        "checkout": {"commit": head},
    }
    fake_uv(tmp_path, monkeypatch, status)

    report = drift(AppPolicy("evaldash", True, (), ()), root=tmp_path)

    assert report["state"] == "different"
    assert len(report["commits"]) == 1
    assert report["commits"][0].endswith("Fix dashboard score")


def test_release_conflict_keeps_inspected_base_version_and_does_not_retry(tmp_path, monkeypatch):
    calls = fake_uv(tmp_path, monkeypatch, {"state": "different", "version": 4})
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = \"-m 'not slow and not cluster'\"\n")
    record_path = tmp_path / "release.json"
    policy = AppPolicy("evaldash", True, ("tests/test_applets.py",), ())

    with pytest.raises(subprocess.CalledProcessError):
        release(policy, record_path, root=tmp_path)

    arguments = [json.loads(line) for line in calls.read_text().splitlines()]
    publishes = [args for args in arguments if "--base-version" in args]
    assert len(publishes) == 1
    assert publishes[0][publishes[0].index("--base-version") + 1] == "4"
    test_command = next(args for args in arguments if "pytest" in args)
    assert test_command[test_command.index("-m") + 1] == "not slow and not cluster"
    assert json.loads(record_path.read_text())["action"] == "validated"
