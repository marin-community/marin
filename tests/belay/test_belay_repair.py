# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

from experiments.belay.code_change import config_changes
from experiments.belay.submit import (
    FAILED,
    next_stage_runs,
    parse_spec,
    repair_key,
    repairs_used,
    run_id,
    spec_with_repair,
)

SPEC = """experiment: X0236
track: speedrun
runs:
  seed0:
    seed: 0
    variant: fused KDA window
    flags: [--model-set, kda_local_attn_fused_window=64]
  control:
    seed: 0
    variant: speedrun control
    control: true
launched:
  control: uv run ... --run-id belay-X0236-control
  seed0: uv run ... --run-id belay-X0236-seed0
"""


def test_a_repair_replaces_only_the_failed_run_and_keeps_its_control() -> None:
    text = spec_with_repair(SPEC, "seed0", repair_key("seed0", 1), "reshape out_sharding in fused_heads")
    spec = parse_spec(text)
    assert sorted(spec.runs) == ["control", "seed0-r1"]
    relaunch = spec.runs["seed0-r1"]
    assert (relaunch.seed, relaunch.flags) == (0, ("--model-set", "kda_local_attn_fused_window=64"))
    assert run_id(spec, relaunch) == "belay-X0236-seed0-r1"
    assert "&id" not in text
    failed = yaml.safe_load(text)[FAILED]["seed0"]
    assert (failed["replaced_by"], failed["reason"]) == ("seed0-r1", "reshape out_sharding in fused_heads")


def test_a_relaunch_of_a_relaunch_keeps_one_suffix() -> None:
    assert [repair_key("seed0", 1), repair_key("seed0-r1", 2), repair_key("seed0-t-r2", 3)] == [
        "seed0-r1",
        "seed0-r2",
        "seed0-t-r3",
    ]


def test_attempts_count_across_the_ideas_specs_only(tmp_path: Path) -> None:
    once = spec_with_repair(SPEC, "seed0", "seed0-r1", "first")
    twice = spec_with_repair(once, "seed0-r1", "seed0-r2", "second")
    (tmp_path / "X0236.yaml").write_text(twice)
    (tmp_path / "X0236d768.yaml").write_text(once.replace("X0236", "X0236d768"))
    (tmp_path / "X02361.yaml").write_text(once.replace("X0236", "X02361"))  # another idea's id, same prefix
    assert repairs_used("X0236", tmp_path) == 3


def test_a_repair_may_not_touch_the_config_but_needs_no_new_flag() -> None:
    before = {"GrugModelConfig": {"kda_local_attn_fused_window": "0", "qk_mult": "1.3"}}
    assert config_changes(before, before) == []  # a code-only fix: nothing to refuse
    added = {"GrugModelConfig": {**before["GrugModelConfig"], "extra": "False"}}
    changed = {"GrugModelConfig": {**before["GrugModelConfig"], "qk_mult": "1.0"}}
    assert config_changes(before, added) == ["GrugModelConfig.extra"]
    assert config_changes(before, changed) == ["GrugModelConfig.qk_mult"]


def test_the_equal_time_run_repeats_the_variants_settings_at_the_new_step_count() -> None:
    spec = parse_spec(SPEC.replace("seed0-r1", "seed0"))
    runs = next_stage_runs(spec, 2768, "")
    assert list(runs) == ["seed0-t"]
    timed = runs["seed0-t"]
    assert timed["stage"] == "time" and timed["seed"] == 0
    assert timed["flags"] == ["--model-set", "kda_local_attn_fused_window=64", "--num-steps", "2768"]
