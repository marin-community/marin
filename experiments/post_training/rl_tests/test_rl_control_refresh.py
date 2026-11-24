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
    monkeypatch.setattr(refresh, "configure_coreweave_s3", lambda: pytest.fail("planner accepted a live invocation"))
    with pytest.raises(SystemExit) as error:
        refresh.main(())
    assert error.value.code == 2 and "--dry-run" in capsys.readouterr().err
    monkeypatch.setattr(refresh, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(
        refresh.Provenance, "capture", lambda _: refresh.Provenance("tree", "a" * 10, False, "main", "test")
    )
    monkeypatch.setattr(refresh.rl_gate, "validate_candidate", lambda *args, **kwargs: configs[None])
    changed_source = None
    last_pin = table.control_pins["learning-medium"]
    last_path = refresh.rl_gate.control_path(
        table.by_name["learning-medium"], last_pin, last_pin.replicates - 1, table=table
    )

    def read_attempt(path):
        assert "learning-medium" in path
        source = None if changed_source == "train" and path != last_path else changed_source
        return attempt.model_copy(update={"config": configs[source]})

    monkeypatch.setattr(refresh, "read_stored_skyrl_attempt", read_attempt)

    def planned(*args):
        refresh.main(("--dry-run", *args))
        report = json.loads(capsys.readouterr().out)
        assert report["commit"] == "a" * 10
        return report["refresh"]

    assert planned() == []
    changed_source = "eval"
    assert planned() == []
    revised = replace(
        table,
        presets=tuple(
            replace(preset, revision="2026.11.24.1") if preset.control_preset == "learning-medium" else preset
            for preset in table.presets
        ),
    )
    monkeypatch.setattr(refresh, "SNOWBALL", revised)
    assert planned("--preset", "learning-medium") == ["learning-medium"]
    mismatched = replace(
        revised,
        presets=tuple(
            table.by_name["data-medium"] if preset.name == "data-medium" else preset for preset in revised.presets
        ),
    )
    monkeypatch.setattr(refresh, "SNOWBALL", mismatched)
    with pytest.raises(ValueError, match="data-medium must share the learning-medium revision"):
        planned("--preset", "data-medium")
    monkeypatch.setattr(refresh, "SNOWBALL", table)
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


def test_generation_refresh_reads_the_sampled_training_set(recorded_runs, tmp_path, monkeypatch):
    root = recorded_runs("data-short", "users/test/refresh-screening")
    attempt = refresh.read_stored_skyrl_attempt(str(root))
    assert attempt is not None and attempt.config is not None
    config = deepcopy(attempt.config)
    assert config["inputs"]["train_data"] == []

    for name, prompt in (("current", "current task"), ("old", "old task")):
        pq.write_table(pa.table({"task_spec": [json.dumps({"id": "task", "prompt": prompt})]}), tmp_path / name)
    current = {
        "identity": "selection/sample@version:fingerprint",
        "uri": str(tmp_path),
        "relative_path": "current",
        "local_path": str(tmp_path),
    }
    config["inputs"]["validation_data"] = [current]
    observed = deepcopy(config)
    preset = SNOWBALL.by_name["data-short"]
    pin = replace(SNOWBALL.control_pins[preset.name], revision=preset.revision)
    table = replace(SNOWBALL, control_pins={preset.name: pin})
    monkeypatch.setattr(
        refresh.Provenance, "capture", lambda _: refresh.Provenance("tree", "a" * 10, False, "main", "test")
    )
    monkeypatch.setattr(refresh.rl_gate, "validate_candidate", lambda *args, **kwargs: config)
    monkeypatch.setattr(refresh, "read_stored_skyrl_attempt", lambda _: attempt.model_copy(update={"config": observed}))

    assert refresh.plan_refresh(table, preset.name)[1] == []
    observed["inputs"]["validation_data"][0]["relative_path"] = "old"
    assert refresh.plan_refresh(table, preset.name)[1] == [preset.name]
    current.update(identity="selection/sample@version:new-fingerprint", relative_path="unbuilt")
    assert refresh.plan_refresh(table, preset.name)[1] == [preset.name]
