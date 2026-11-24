# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

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
    for kind in ("current", "train", "eval"):
        config = deepcopy(attempt.config)
        for name in ("train", "eval"):
            path = tmp_path / f"{kind}-{name}.parquet"
            prompt = "changed" if kind == name else "shared"
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
    stale_source = "current"

    def read_attempt(path):
        config = configs[stale_source if "learning-medium" in path else "current"]
        return attempt.model_copy(update={"config": config})

    monkeypatch.setattr(refresh, "read_stored_skyrl_attempt", read_attempt)
    monkeypatch.setattr(refresh, "qualify_refresh", lambda *args: pytest.fail("dry run submitted work"))
    args = ("--pull", str(request.pull), "--dry-run")
    refresh.main(args)
    assert json.loads(capsys.readouterr().out)["refresh"] == []

    for stale_source in ("train", "eval"):
        refresh.main(args)
        assert json.loads(capsys.readouterr().out)["refresh"] == ["learning-medium"], stale_source


def test_comment_requests_wait_for_main_and_require_the_allowlist(monkeypatch):
    repository = "marin-community/marin"
    request_comment = {"body": "/rl-controls", "user": {"login": "member"}}
    event = {"issue": {"number": 37, "pull_request": {}}, "comment": request_comment}
    pull = {"number": 37, "base": {"ref": "main", "repo": {"full_name": repository}}, "merged": False}

    comments = [dict(request_comment, body="/rl-controls "), dict(request_comment, user={"login": "stranger"})]

    def github(endpoint, *flags):
        return [comments, authorized_page] if flags else pull

    monkeypatch.setattr(rl_control_request, "_github", lambda *args: pytest.fail("unauthorized PR lookup"))
    assert rl_control_request.resolve_request("issue_comment", event, repository, set()) is None
    monkeypatch.setattr(rl_control_request, "_github", github)
    assert rl_control_request.resolve_request("issue_comment", event, repository, {"member"}) is None

    sha = "b" * 40
    pull.update(merged=True, merge_commit_sha=sha)
    closed = {"pull_request": pull}
    authorized_page = []
    assert rl_control_request.resolve_request("pull_request_target", closed, repository, {"member"}) is None
    authorized_page = [request_comment]
    assert rl_control_request.resolve_request(
        "issue_comment", event, repository, {"member"}
    ) == rl_control_request.RefreshRequest(37, sha, False)
    assert rl_control_request.resolve_request(
        "pull_request_target", closed, repository, {"member"}
    ) == rl_control_request.RefreshRequest(37, sha, False)
    pull["base"]["ref"] = "experimental"
    assert rl_control_request.resolve_request("pull_request_target", closed, repository, {"member"}) is None


def test_live_refresh_requires_published_links_and_current_main_before_opening_pins(monkeypatch):
    preset = SNOWBALL.by_name["speed-medium"]
    previous = SNOWBALL.control_pins[preset.name]
    current = replace(previous, commit="a" * 10, revision=preset.revision)
    item = refresh.ControlRefresh(preset, previous, current)
    plan = []
    drafts = []
    submissions = []
    publication_errors = []
    changed_paths = ""
    advance_main = False

    def check_output(args, *, text):
        match args[:2]:
            case ["gh", "api"]:
                return "controls-bot\n"
            case ["git", "rev-parse"]:
                return "a" * 40 + "\n"
            case ["git", "diff"]:
                return changed_paths
            case _:
                pytest.fail(f"unexpected command {args}")

    def command(args, *, check, **_kwargs):
        submissions.append(args[:3])
        assert args[:2] in (["gh", "pr"], ["git", "fetch"], ["git", "config"], ["git", "ls-remote"])
        return SimpleNamespace(returncode=2)

    def publish(outputs, requested_preset, *, table, gate_path):
        nonlocal changed_paths
        assert requested_preset == preset and gate_path == "stored-link"
        assert len(outputs) == current.replicates
        if advance_main:
            changed_paths = "experiments/post_training/snowball_rl.py\n"
        return SimpleNamespace(errors=publication_errors)

    monkeypatch.setattr(refresh, "subprocess", SimpleNamespace(check_output=check_output, run=command))
    monkeypatch.setattr(refresh, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(refresh, "plan_refresh", lambda _: plan)
    monkeypatch.setattr(refresh, "qualify_refresh", lambda *args, **kwargs: (SNOWBALL, ((item, "stored-link"),)))
    monkeypatch.setattr(refresh.rl_publication, "publish_runs", publish)
    monkeypatch.setattr(refresh, "_submit_pin_pr", lambda *args: drafts.append(args))
    monkeypatch.setenv("RL_CONTROLS_BOT_LOGIN", "controls-bot")
    monkeypatch.setenv("RL_CONTROLS_ALLOWED_LOGINS", "reviewer")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    args = ("--pull", "37")
    refresh.main(args)
    assert submissions == [["gh", "pr", "comment"]] and not drafts

    for name, value in (
        ("RL_CONTROLS_MIN_OVERLAP", str(refresh.rl_gate.MIN_SHARED_TASK_FRACTION)),
        ("TOGETHER_API_KEY", "fixture"),
        ("WANDB_API_KEY", "fixture"),
    ):
        monkeypatch.setenv(name, value)
    plan.append(item)

    publication_errors.append("temporary service failure")
    with pytest.raises(RuntimeError, match="temporary service failure"):
        refresh.main(args)
    assert not drafts

    publication_errors.clear()
    advance_main = True
    with pytest.raises(ValueError, match="current main changed Snowball control inputs"):
        refresh.main(args)
    assert not drafts

    advance_main = False
    changed_paths = ""
    refresh.main(args)
    [(_table, reports, source, branch, pull, published)] = drafts
    assert (reports, source, branch, pull, published) == (
        ((item, "stored-link"),),
        "a" * 40,
        "rl-controls/" + "a" * 40,
        37,
        None,
    )
