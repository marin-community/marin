# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from experiments.post_training.long_loop.__main__ import validate_graph
from experiments.post_training.long_loop.run_spec import allocation_peaks, resolve_spec
from experiments.post_training.long_loop.spec_pipeline import build_spec_graph

SPEC = Path(__file__).parents[1] / "specs" / "round1.yaml"


def test_production_and_smoke_preserve_data_context_and_quota():
    production = resolve_spec(SPEC, smoke=False)
    smoke = resolve_spec(SPEC, smoke=True)
    assert smoke["datasets"] == production["datasets"]
    assert smoke["sft"]["data"] == production["sft"]["data"]
    actual = hashlib.sha256(
        json.dumps(production["sft"]["data"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert actual == production["datasets"]["sft"]["provenance"]["canonical_data_sha256"]
    assert [step for step, _ in smoke["sft"]["data"]["train_weights"]] == list(range(0, 4750, 250))
    assert smoke["sft"]["stop_after_steps"] < production["sft"]["stop_after_steps"]
    assert smoke["rl"]["config"]["trainer"]["max_steps"] < production["rl"]["config"]["trainer"]["max_steps"]
    assert smoke["sft"]["model"] == production["sft"]["model"]
    assert smoke["rl"]["config"]["context_budget"] == production["rl"]["config"]["context_budget"]
    for field in ("rollout_buffer", "policy", "ref"):
        assert smoke["rl"]["config"]["trainer"][field] == production["rl"]["config"]["trainer"][field]
    assert smoke["blockers"] == production["blockers"]


def test_changed_pin_and_semantic_smoke_override_fail_before_graph(tmp_path):
    root = tmp_path / "specs"
    shutil.copytree(SPEC.parent, root)
    path = root / SPEC.name
    raw = yaml.safe_load(path.read_text())
    raw["smoke_overrides"]["context_length"] = 2048
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Smoke cannot change"):
        resolve_spec(path, smoke=True)
    shutil.copyfile(SPEC, path)
    data = root / "sft-data.yaml"
    data.write_text(data.read_text() + "\n# changed bytes\n")
    with pytest.raises(ValueError, match="Pinned file changed"):
        resolve_spec(path, smoke=False)


def test_actual_sft_graph_constructs_without_admitting_unprepared_rl(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin/")
    resolved = resolve_spec(SPEC, smoke=False)
    terminal = build_spec_graph(resolved, "2026.11.09.12", through="sft")
    validate_graph(terminal, tmp_path)
    rows = json.loads((tmp_path / "graph.json").read_text())
    assert [row["stage"].rsplit("/", 1)[-1] for row in rows] == ["base", "inputs", "sft"]
    assert rows[0]["source"] == resolved["initializer"]["uri"]
    assert terminal.runtime_args["skyrl_execution"].priority == "production"
    with pytest.raises(ValueError, match="Run is not admitted"):
        build_spec_graph(resolved, "2026.11.09.12", through="archive")


@pytest.mark.parametrize("stage", ["coordinator", "rl", "evaluation", "endpoint"])
def test_new_submission_rejects_stale_interactive_priority(stage):
    resolved = resolve_spec(SPEC, smoke=False)
    if stage == "coordinator":
        runtime = resolved["execution"]["coordinator"]
    elif stage == "endpoint":
        runtime = resolved[stage]
    else:
        runtime = resolved[stage]["execution"]
    runtime["priority"] = "interactive"
    with pytest.raises(ValueError, match="production"):
        allocation_peaks(resolved)
