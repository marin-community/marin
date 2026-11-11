# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from marin.execution.lazy import StepContext
from marin.rl.skyrl import SkyRLRun

from experiments.post_training.mochi_one_rack.launch import build_smoke


@pytest.mark.parametrize(
    "stage,previous_stage,previous_step", [("four_k", "initial_four_k", 4), ("thirty_two_k", "four_k", 5)]
)
def test_resumed_stage_loads_exact_muonh_checkpoint_and_saves_to_its_own_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, previous_stage: str, previous_step: int
) -> None:
    smoke = build_smoke("2026.11.11.9300", initial_four_k_version="2026.11.11.9298", four_k_version="2026.11.11.9299")
    previous = getattr(smoke, previous_stage)
    prefix = str(tmp_path)
    completed = SkyRLRun(
        path=previous.path(prefix),
        hf_model_uri=None,
        global_step=None,
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
    step = getattr(smoke, stage)
    config = step.build_config(
        StepContext.for_run(step.path(prefix), prefix, runtime_args=step.runtime_args, deps=step.deps)
    )
    resumed = yaml.safe_load(config.launch_config_yaml)
    assert resumed["skyrl"]["trainer"]["resume_mode"] == "from_path"
    assert resumed["skyrl"]["trainer"]["resume_path"] == f"s3://retained-four-k/checkpoints/global_step_{previous_step}"
    assert resumed["artifacts"]["checkpoint_root"] == config.output.checkpoint_root
    assert config.output.checkpoint_root.endswith(f"{step.name}/{step.version}/checkpoints")
    assert config.output.checkpoint_root != completed.checkpoint_root
    checkpoint_callback = next(c for c in resumed["skyrl"]["trainer"]["callbacks"] if c["type"] == "checkpoint")
    assert checkpoint_callback["save_steps"] == previous_step + (1 if stage == "four_k" else 5)
    assert checkpoint_callback["save_on_train_end"]
    assert "mochi-muonh" in config.output.checkpoint_root
