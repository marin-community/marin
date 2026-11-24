# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest

from experiments.post_training import rl_control_refresh as refresh
from experiments.post_training import snowball_rl
from experiments.post_training.rl_tests.fixtures import cache_native_compositions, controls, edit_receipt

pytestmark = pytest.mark.usefixtures("source_provenance")


def test_nightly_and_named_plans_use_gate_control_validation(recorded_runs, monkeypatch, capsys):
    source = snowball_rl.SNOWBALL
    names = ("learning-medium", "data-medium", "speed-medium", "train-long")
    learning = source.control_pins["learning-medium"]
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
    roots = controls(recorded_runs, "learning-medium")
    cache_native_compositions(monkeypatch)

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
    with edit_receipt(roots[-1]) as receipt:
        receipt["config"]["skyrl"]["trainer"]["policy"]["optimizer_config"]["lr"] = learning_rate
    roots[-1].rename(roots[-1].with_name("unavailable"))
    with pytest.raises(refresh.rl_gate.InvalidComparison, match="expected a successful pinned control receipt"):
        planned(*args)

    monkeypatch.setattr(refresh, "SNOWBALL", replace(table, control_pins={}))
    assert planned(*args) == {"snowball": ["learning-medium"]}
    with pytest.raises(ValueError, match="choose a gated preset"):
        planned("--model", "snowball", "--preset", "train-long")
    monkeypatch.setattr(
        refresh.Provenance, "capture", lambda _: refresh.Provenance("tree", "commit-one", True, "main", "test")
    )
    with pytest.raises(ValueError, match="needs a clean source commit"):
        planned(*args)
