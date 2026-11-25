# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest

from experiments.post_training import rl_control_refresh as refresh
from experiments.post_training import snowball_rl
from experiments.post_training.rl_tests.fixtures import controls, edit_receipt

pytestmark = pytest.mark.usefixtures("source_provenance")


def test_nightly_and_named_plans_use_gate_control_validation(
    recorded_runs, monkeypatch, capsys, cache_native_compositions
):
    source = snowball_rl.SNOWBALL
    names = ("learning-medium", "data-medium", "speed-medium", "train-long")
    learning = replace(source.control_pins["learning-medium"], cluster="cw-rno2a")
    table = replace(
        source,
        presets=tuple(source.by_name[name] for name in names),
        control_pins={
            "learning-medium": learning,
            "speed-medium": replace(source.control_pins["speed-medium"], revision="2026.01.01"),
        },
    )
    monkeypatch.setattr(refresh, "SNOWBALL", table)
    monkeypatch.setattr(refresh, "configure_coreweave_s3", lambda: None)
    owner = table.by_name[learning.preset]
    assert owner.recipe.layout.cluster != learning.cluster
    placed = replace(owner, recipe=replace(owner.recipe, layout=replace(owner.recipe.layout, cluster=learning.cluster)))
    placed_table = replace(table, presets=(placed, *table.presets[1:]))
    roots = controls(recorded_runs, learning.preset, table=placed_table)
    cache_native_compositions()

    def planned(*args):
        refresh.main(args)
        report = json.loads(capsys.readouterr().out)
        assert report["commit"] == "commit-one"
        return report["refresh"]

    assert planned() == {"snowball": ["speed-medium"], "mochi": []}
    args = ("--model", "snowball", "--preset", "data-medium")
    assert planned(*args) == {"snowball": []}
    with edit_receipt(roots[-1]) as receipt:
        optimizer = receipt["config"]["skyrl"]["trainer"]["policy"]["optimizer_config"]
        learning_rate = optimizer["lr"]
        optimizer["lr"] *= 2
    assert planned(*args) == {"snowball": ["learning-medium"]}
    assert planned() == {"snowball": ["learning-medium", "speed-medium"], "mochi": []}
    with edit_receipt(roots[-1]) as receipt:
        receipt["config"]["skyrl"]["trainer"]["policy"]["optimizer_config"]["lr"] = learning_rate
    with edit_receipt(roots[-1]) as receipt:
        receipt["result"]["state"] = "failed"
    with pytest.raises(refresh.rl_gate.InvalidComparison, match="expected a successful pinned control receipt"):
        planned(*args)
    roots[-1].rename(roots[-1].with_name("unavailable"))
    with pytest.raises(refresh.rl_gate.InvalidComparison, match="expected a successful pinned control receipt"):
        planned(*args)

    monkeypatch.setattr(refresh, "SNOWBALL", replace(table, control_pins={}))
    assert planned(*args) == {"snowball": ["learning-medium"]}
    with pytest.raises(ValueError, match="choose a gated preset"):
        planned("--model", "snowball", "--preset", "train-long")
    for commit, dirty in (("", False), ("commit-one", True)):
        provenance = refresh.Provenance("tree", commit, dirty, "main", "test")
        monkeypatch.setattr(refresh.Provenance, "capture", lambda _, value=provenance: value)
        with pytest.raises(ValueError, match="needs a clean source commit"):
            planned(*args)

    temporary = replace(owner, temporary_output=True)
    with pytest.raises(ValueError, match="needs a durable control preset"):
        replace(table, presets=(temporary, *table.presets[1:]), control_pins={})
