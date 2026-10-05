# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import yaml
from marin.execution.lazy import StepContext

from experiments.post_training import async_rl
from experiments.post_training.curriculum_rl.launch import SNOWBALL_POLICY


def test_launcher_builds_complete_smoke_run(monkeypatch) -> None:
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    monkeypatch.setattr(async_rl, "username_segment", lambda: "alice")

    run = async_rl.build_run(SNOWBALL_POLICY, async_rl.SMOKE_PRESET, version="2026.09.18")

    assert run.rl.name == "users/alice/checkpoints/async-rl/snowball-smoke"
    assert any(dep.name == async_rl.POOL_ARTIFACT_NAME for dep in run.rl.deps)
    assert any(dep.name == SNOWBALL_POLICY.adopted_model.name for dep in run.rl.deps)
    assert run.evaluation.deps == (run.rl,)
    assert run.evaluation.name == "evals/alice-async-rl-snowball-smoke/gsm8k-smoke"

    tuned = async_rl.build_run(
        SNOWBALL_POLICY,
        async_rl.SMOKE_PRESET,
        version="2026.09.18",
        settings=("trainer.policy.optimizer_config.lr=5e-6", "trainer.max_steps=3", "trainer.eval_interval=2"),
    )
    config = tuned.rl.build_config(StepContext.for_fingerprint(tuned.rl.runtime_args, tuned.rl.deps))
    trainer = yaml.safe_load(config.launch_config_yaml)["skyrl"]["trainer"]
    assert trainer["policy"]["optimizer_config"]["lr"] == 5e-6
    assert trainer["ckpt_interval"] == 2
    assert trainer["hf_save_interval"] == 6
    assert tuned.rl.fingerprint() != run.rl.fingerprint()
    with pytest.raises(ValueError, match=r"generator\.n_samples_per_prompt"):
        async_rl.build_run(
            SNOWBALL_POLICY,
            async_rl.SMOKE_PRESET,
            version="2026.09.18",
            settings=("generator.n_samples_per_prompt=8",),
        )
