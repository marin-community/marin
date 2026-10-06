# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
