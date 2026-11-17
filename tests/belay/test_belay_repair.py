# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

from experiments.belay.submit import FAILED, parse_spec, repair_key, repairs_used, run_id, spec_with_repair

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
