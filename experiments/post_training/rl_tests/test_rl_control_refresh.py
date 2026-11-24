# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training import rl_control_refresh as refresh
from experiments.post_training.snowball_rl import SNOWBALL
from infra.ci import rl_control_request


def test_refresh_dry_run_uses_the_requested_merge_and_plans_only_stale_cohorts(
    recorded_runs, tmp_path, monkeypatch, capsys
):
    repository = "marin-community/marin"
    sha = "a" * 40
    pull = {"merged": True, "merge_commit_sha": sha, "base": {"ref": "main", "repo": {"full_name": repository}}}
    monkeypatch.setattr(rl_control_request, "_github", lambda endpoint: pull)
    request = rl_control_request.resolve_request(
        "workflow_dispatch", {"inputs": {"pull_request": "42"}}, repository, set()
    )
    assert request == rl_control_request.RefreshRequest(42, sha, True)

    recorded = recorded_runs("learning-medium", "users/test/refresh-source")
    attempt = refresh.read_stored_skyrl_attempt(str(recorded))
    assert attempt is not None and attempt.config is not None
    configs = {}
    for kind in ("current", "old"):
        config = deepcopy(attempt.config)
        for name, prompt in (("train", kind), ("eval", "shared")):
            path = tmp_path / f"{kind}-{name}.parquet"
            pq.write_table(pa.table({"task_spec": [json.dumps({"id": name, "prompt": prompt})]}), path)
            source = {"identity": name, "uri": str(tmp_path), "relative_path": path.name, "local_path": str(tmp_path)}
            config["inputs"]["train_data" if name == "train" else "validation_data"] = [source]
        configs[kind] = config

    names = ("learning-medium", "speed-medium")
    table = replace(
        SNOWBALL,
        control_pins={
            name: replace(SNOWBALL.control_pins[name], revision=SNOWBALL.by_name[name].revision) for name in names
        },
    )
    monkeypatch.setattr(refresh, "SNOWBALL", table)
    monkeypatch.setattr(refresh, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(
        refresh.Provenance, "capture", lambda _: refresh.Provenance("tree", sha[:10], False, "main", "test")
    )
    monkeypatch.setattr(refresh.rl_gate, "_compose_plans", lambda *args: [configs["current"]])
    stale_preset = None

    def read_attempt(path):
        config = configs["old" if stale_preset is not None and stale_preset in path else "current"]
        return attempt.model_copy(update={"config": config})

    monkeypatch.setattr(refresh, "read_stored_skyrl_attempt", read_attempt)
    monkeypatch.setattr(refresh, "qualify_refresh", lambda *args: pytest.fail("dry run submitted work"))
    args = ("--pull", str(request.pull), "--dry-run")
    refresh.main(args)
    assert capsys.readouterr().out == "All RL controls match this merge. No collection planned.\n"

    stale_preset = "learning-medium"
    refresh.main(args)
    assert capsys.readouterr().out == "Would refresh learning-medium.\n"


def test_comment_requests_wait_for_main_and_require_the_allowlist(monkeypatch, capsys):
    repository = "marin-community/marin"
    request_comment = {"body": "/rl-controls", "user": {"login": "member"}}
    event = {"issue": {"number": 37, "pull_request": {}}, "comment": request_comment}
    pull = {"number": 37, "base": {"ref": "main", "repo": {"full_name": repository}}, "merged": False}

    def github(endpoint, *flags):
        return [[request_comment]] if flags else pull

    monkeypatch.setattr(rl_control_request, "_github", github)
    assert rl_control_request.resolve_request("issue_comment", event, repository, set()) is None
    assert capsys.readouterr().out == ""
    assert rl_control_request.resolve_request("issue_comment", event, repository, {"member"}) is None
    assert "awaits its merge into main" in capsys.readouterr().out

    sha = "b" * 40
    pull.update(merged=True, merge_commit_sha=sha)
    closed = {"pull_request": pull}
    assert rl_control_request.resolve_request(
        "pull_request_target", closed, repository, {"member"}
    ) == rl_control_request.RefreshRequest(37, sha, False)
    pull["base"]["ref"] = "experimental"
    assert rl_control_request.resolve_request("pull_request_target", closed, repository, {"member"}) is None
