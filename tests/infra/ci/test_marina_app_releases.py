# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the maintained-app release selector and CI gates."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci.marina_app_releases import ROOT, AppPolicy, changed_apps, drift, release


def fake_uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: dict, *, mode: str = "conflict") -> Path:
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
        "calls = [json.loads(line) for line in open(os.environ['FAKE_UV_CALLS'])]\n"
        "publishes = sum('--base-version' in call for call in calls)\n"
        "if 'release-status' in args:\n"
        "    if os.environ['FAKE_MODE'] == 'stale_once' and publishes == 2:\n"
        "        print(json.dumps({'state': 'current', 'version': 5}))\n"
        "    else:\n"
        "        print(os.environ['FAKE_STATUS'])\n"
        "elif '--dry-run' in args:\n"
        "    print(json.dumps({'checks': ['package', 'backend_import'], 'file_count': 3}))\n"
        "elif '--base-version' in args:\n"
        "    if os.environ['FAKE_MODE'] == 'stale_once':\n"
        "        if publishes == 1:\n"
        '            print("stale main checkout", file=sys.stderr)\n'
        "            sys.exit(os.EX_TEMPFAIL)\n"
        "        print(json.dumps({'version': 5}))\n"
        "    else:\n"
        "        sys.exit(17)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_UV_CALLS", str(calls))
    monkeypatch.setenv("FAKE_STATUS", json.dumps(status))
    monkeypatch.setenv("FAKE_MODE", mode)
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


def test_selector_checks_policy_completeness_and_emits_empty_matrix():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ci/marina_app_releases.py",
            "select",
            "--before",
            "HEAD",
            "--after",
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == '{"include":[]}'


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


def test_release_refuses_unrecorded_live_source(tmp_path, monkeypatch):
    calls = fake_uv(tmp_path, monkeypatch, {"state": "unrecorded", "version": 6})

    with pytest.raises(RuntimeError, match="release manually before automation"):
        release(AppPolicy("evaldash", True, (), ()), tmp_path / "release.json", root=tmp_path)

    assert all("--base-version" not in json.loads(line) for line in calls.read_text().splitlines())


def test_release_refreshes_stale_main_and_revalidates_before_publishing(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", remote], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / ".github").mkdir()
    (repo / ".github/marina-app-releases.toml").write_text(
        '[apps.evaldash]\nrelease_on_merge = true\ntests = ["tests/test_applets.py"]\n'
    )
    app = repo / "infra/marina/apps/evaldash"
    app.mkdir(parents=True)
    (app / "applet.toml").write_text('title = "EvalDash"\n')
    tests = repo / "infra/marina/tests"
    tests.mkdir(parents=True)
    (tests / "test_applets.py").write_text("def test_placeholder(): pass\n")
    (repo / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = \"-m 'not slow'\"\n")
    (app / "app.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "Initial app"], cwd=repo, check=True)
    subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=repo, check=True)
    subprocess.run(["git", "push", "-q", "-u", "origin", "main"], cwd=repo, check=True)
    old_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    (app / "app.py").write_text("value = 2\n")
    subprocess.run(["git", "commit", "-qam", "Updated app"], cwd=repo, check=True)
    new_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    subprocess.run(["git", "push", "-q", "origin", "main"], cwd=repo, check=True)
    subprocess.run(["git", "reset", "--hard", "-q", old_head], cwd=repo, check=True)
    calls = fake_uv(tmp_path, monkeypatch, {"state": "different", "version": 4}, mode="stale_once")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    record = release(AppPolicy("evaldash", True, ("tests/test_applets.py",), ()), tmp_path / "release.json", root=repo)

    arguments = [json.loads(line) for line in calls.read_text().splitlines()]
    assert sum("pytest" in args for args in arguments) == 2
    assert sum("--dry-run" in args for args in arguments) == 2
    assert sum("--base-version" in args for args in arguments) == 2
    assert record["verification"] == "passed"
    assert (
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
        == new_head
    )
