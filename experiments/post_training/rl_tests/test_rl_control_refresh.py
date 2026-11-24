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


def test_nightly_and_named_dry_runs_plan_revision_and_launch_selected_training_changes(
    recorded_runs, tmp_path, monkeypatch, capsys
):
    recorded = recorded_runs("learning-medium", "users/test/refresh-source")
    attempt = refresh.read_stored_skyrl_attempt(str(recorded))
    assert attempt is not None and attempt.config is not None
    configs = {}
    for changed in (None, "train", "eval"):
        config = deepcopy(attempt.config)
        for name, key in (("train", "train_data"), ("eval", "validation_data")):
            path = tmp_path / f"{changed}-{name}.parquet"
            prompt = "changed" if changed == name else "shared"
            pq.write_table(pa.table({"task_spec": [json.dumps({"id": name, "prompt": prompt})]}), path)
            config["inputs"][key] = [
                {"identity": name, "uri": str(tmp_path), "relative_path": path.name, "local_path": str(tmp_path)}
            ]
        configs[changed] = config

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
        refresh.Provenance, "capture", lambda _: refresh.Provenance("tree", "a" * 10, False, "main", "test")
    )
    monkeypatch.setattr(refresh.rl_gate, "_compose_plans", lambda *args: [configs[None]])
    changed_source = None

    def read_attempt(path):
        assert "learning-medium" in path
        return attempt.model_copy(update={"config": configs[changed_source]})

    monkeypatch.setattr(refresh, "read_stored_skyrl_attempt", read_attempt)
    monkeypatch.setattr(refresh, "qualify_refresh", lambda *args, **kwargs: pytest.fail("dry run submitted work"))

    def planned(*args):
        refresh.main(("--dry-run", *args))
        return json.loads(capsys.readouterr().out)["refresh"]

    assert planned() == []
    changed_source = "eval"
    assert planned() == []
    changed_source = "train"
    assert planned() == ["learning-medium"]
    stale_speed = replace(table.control_pins["speed-medium"], revision="2026.01.01")
    monkeypatch.setattr(
        refresh, "SNOWBALL", replace(table, control_pins={**table.control_pins, "speed-medium": stale_speed})
    )
    assert set(planned()) == {"learning-medium", "speed-medium"}
    assert planned("--preset", "speed-medium") == ["speed-medium"]
    assert planned("--preset", "data-medium") == ["learning-medium"]
    with pytest.raises(ValueError, match="choose a gated preset"):
        planned("--preset", "train-long")


def test_live_refresh_requires_owner_inputs_published_links_and_current_main(monkeypatch):
    preset = SNOWBALL.by_name["speed-medium"]
    previous = SNOWBALL.control_pins[preset.name]
    current = replace(previous, commit="a" * 10, revision=preset.revision)
    item = refresh.ControlRefresh(preset, previous, current)
    plan = []
    drafts = []
    commands = []
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
        commands.append(args[:2])
        assert args[:2] in (["git", "fetch"], ["git", "config"], ["git", "ls-remote"])
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
    monkeypatch.setattr(refresh, "plan_refresh", lambda *args: plan)
    monkeypatch.setattr(refresh, "qualify_refresh", lambda *args, **kwargs: (SNOWBALL, ((item, "stored-link"),)))
    monkeypatch.setattr(refresh.rl_publication, "publish_runs", publish)
    monkeypatch.setattr(refresh, "_submit_pin_pr", lambda *args: drafts.append(args))
    monkeypatch.setenv("RL_CONTROLS_BOT_LOGIN", "controls-bot")
    monkeypatch.setenv("RL_CONTROLS_PIN_MERGE_POLICY", "fixture")
    monkeypatch.delenv("RL_CONTROLS_CODE_CHANGES", raising=False)
    with pytest.raises(ValueError, match="code-change policy, pin merge policy, and bot login"):
        refresh.main(())
    monkeypatch.setenv("RL_CONTROLS_CODE_CHANGES", "fixture")
    monkeypatch.setenv("TOGETHER_API_KEY", "fixture")
    monkeypatch.setenv("WANDB_API_KEY", "fixture")
    refresh.main(())
    assert not commands and not drafts
    plan.append(item)

    publication_errors.append("temporary service failure")
    with pytest.raises(RuntimeError, match="temporary service failure"):
        refresh.main(())
    assert not drafts

    publication_errors.clear()
    advance_main = True
    with pytest.raises(ValueError, match="current main changed Snowball control inputs"):
        refresh.main(())
    assert not drafts

    advance_main = False
    changed_paths = ""
    refresh.main(())
    [(_table, reports, source, branch, published)] = drafts
    assert (reports, source, branch, published) == (((item, "stored-link"),), "a" * 40, "rl-controls/" + "a" * 40, None)
