# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from scripts.ci import snowball_release


@pytest.fixture
def marina_process(monkeypatch):
    replies = []
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "node":
            return SimpleNamespace(returncode=0)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(stdout=json.dumps(reply))

    monkeypatch.setattr(snowball_release.subprocess, "run", run)
    return replies, commands


def test_unchanged_source_does_not_build_or_publish(marina_process):
    replies, commands = marina_process
    replies.append({"state": "current", "version": 6})

    result = CliRunner().invoke(snowball_release.main)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["action"] == "unchanged"
    assert len(commands) == 1
    assert "release-status" in commands[0]


def test_changed_source_releases_observed_version_and_records_provenance(marina_process):
    replies, commands = marina_process
    before = {"state": "different", "version": 6, "released": {"commit": "old"}}
    after = {"state": "current", "version": 7, "released": {"commit": "main-commit"}}
    published = {"version": 7, "url": "https://public.applets.marina.oa.dev/a/tree/v/7/"}
    replies.extend([before, published, after])

    result = CliRunner().invoke(snowball_release.main)

    assert result.exit_code == 0, result.output
    record = json.loads(result.output)
    assert record == {"action": "released", "before": before, "release": published, "after": after}
    assert commands[2] == [
        "uv",
        "run",
        "--no-sync",
        "marina",
        "release",
        snowball_release.APP_DIR,
        "--base-version",
        "6",
        "--json",
    ]
    assert commands[1][0] == "node"


def test_unrecorded_source_needs_explicit_bootstrap(marina_process):
    replies, commands = marina_process
    replies.append({"state": "unrecorded", "version": 6})

    result = CliRunner().invoke(snowball_release.main)

    assert result.exit_code != 0
    assert "bootstrap_unrecorded=true" in result.output
    assert len(commands) == 1


def test_bootstrap_can_be_previewed_without_publishing(marina_process):
    replies, commands = marina_process
    replies.extend([{"state": "unrecorded", "version": 6}, {"files": ["dist/index.html"]}])

    result = CliRunner().invoke(snowball_release.main, ["--bootstrap-unrecorded", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["action"] == "would_release"
    assert "--dry-run" in commands[-1]
    assert all("--base-version" not in command for command in commands)


def test_concurrent_release_conflict_is_not_retried(marina_process):
    replies, commands = marina_process
    replies.extend([{"state": "different", "version": 6}, subprocess.CalledProcessError(1, "marina release")])

    result = CliRunner().invoke(snowball_release.main)

    assert result.exit_code != 0
    assert isinstance(result.exception, subprocess.CalledProcessError)
    assert sum("release" in command for command in commands) == 1
    assert len(commands) == 3


def test_post_release_selection_change_is_reported(marina_process):
    replies, _ = marina_process
    replies.extend(
        [
            {"state": "different", "version": 6},
            {"version": 7},
            {"state": "different", "version": 8},
        ]
    )

    result = CliRunner().invoke(snowball_release.main)

    assert result.exit_code != 0
    assert "no longer current" in result.output
