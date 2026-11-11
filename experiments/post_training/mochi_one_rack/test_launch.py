# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from marin.execution.lazy import StepContext
from marin.rl.skyrl import SkyRLRun

from experiments.post_training.mochi_one_rack.launch import build_smoke


def test_resumed_stage_loads_exact_step_five_and_saves_to_its_own_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = build_smoke("2026.11.11.9273")
    prefix = str(tmp_path)
    completed = SkyRLRun(
        path=smoke.four_k.path(prefix),
        hf_model_uri=None,
        global_step=5,
        tokenizer_uri="s3://test/tokenizer",
        tokenizer_revision="test-revision",
        checkpoint_root="s3://retained-four-k/checkpoints",
        draft_checkpoint_root=None,
        terminal_manifest_uri="s3://retained-four-k/terminal.json",
        iris_job_id="/test/completed-four-k",
    )

    def load_completed(cls: type[SkyRLRun], source: str) -> SkyRLRun:
        assert source == completed.path
        return completed

    # Only the completed dependency's metadata read crosses the storage boundary.
    monkeypatch.setattr(SkyRLRun, "raw_load", classmethod(load_completed))
    step = smoke.thirty_two_k
    config = step.build_config(
        StepContext.for_run(step.path(prefix), prefix, runtime_args=step.runtime_args, deps=step.deps)
    )
    resumed = yaml.safe_load(config.launch_config_yaml)
    assert resumed["skyrl"]["trainer"]["resume_mode"] == "from_path"
    assert resumed["skyrl"]["trainer"]["resume_path"] == "s3://retained-four-k/checkpoints/global_step_5"
    assert resumed["artifacts"]["checkpoint_root"] == config.output.checkpoint_root
    assert config.output.checkpoint_root.endswith(f"{step.name}/{step.version}/checkpoints")
    assert config.output.checkpoint_root != completed.checkpoint_root
