# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
import yaml

from experiments.post_training.long_loop.run_spec import resolve_spec

SPEC = Path(__file__).parents[1] / "specs" / "round1.yaml"


def test_production_and_smoke_preserve_data_context_and_quota():
    production = resolve_spec(SPEC, smoke=False)
    smoke = resolve_spec(SPEC, smoke=True)
    assert smoke["datasets"] == production["datasets"]
    assert smoke["sft"]["data"] == production["sft"]["data"]
    assert [step for step, _ in smoke["sft"]["data"]["train_weights"]] == list(range(0, 4750, 250))
    assert smoke["sft"]["stop_after_steps"] < production["sft"]["stop_after_steps"]
    assert smoke["rl"]["config"]["trainer"]["max_steps"] < production["rl"]["config"]["trainer"]["max_steps"]
    for config in (production, smoke):
        assert config["sft"]["model"]["max_seq_len"] == 262144
        assert config["rl"]["config"]["context_budget"]["request_window_tokens"] == 102400
        trainer = config["rl"]["config"]["trainer"]
        assert trainer["rollout_buffer"]["consumed_route_quotas"] == {"agentic": 32, "nonagentic": 32}
        assert trainer["policy"]["megatron_config"]["context_parallel_size"] == 4
        assert any("agentic manifest" in blocker for blocker in config["blockers"])


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
