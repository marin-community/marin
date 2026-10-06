# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from marin.execution.lazy import StepContext

from experiments.post_training import async_rl
from experiments.post_training.curriculum_rl.launch import SNOWBALL_POLICY

COMPOSE_LAUNCH = """
import json
import sys
from cloud.iris.launch_config import load_launch_config

launch = load_launch_config(sys.argv[1])
print(json.dumps({
    "entrypoint": launch.runtime.entrypoint,
    "training_type": launch.runtime.training_type,
    "batch_policy": launch.skyrl.trainer.rollout_buffer.batch_policy,
}))
"""


def test_launcher_builds_complete_smoke_run(monkeypatch) -> None:
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    monkeypatch.setattr(async_rl, "username_segment", lambda: "alice")

    run = async_rl.build_run(SNOWBALL_POLICY, async_rl.SMOKE_PRESET, version="2026.09.18")

    assert run.rl.name == "users/alice/checkpoints/async-rl/snowball-smoke"
    assert any(dep.name == async_rl.POOL_ARTIFACT_NAME for dep in run.rl.deps)
    assert any(dep.name == SNOWBALL_POLICY.adopted_model.name for dep in run.rl.deps)
    assert run.evaluation.deps == (run.rl,)
    assert run.evaluation.name == "evals/alice-async-rl-snowball-smoke/gsm8k-smoke"


def test_rollout_overrides_change_the_run_identity(monkeypatch) -> None:
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    monkeypatch.setattr(async_rl, "username_segment", lambda: "alice")
    run = async_rl.build_run(SNOWBALL_POLICY, async_rl.SMOKE_PRESET, version="2026.09.18")
    capped = async_rl.build_run(
        SNOWBALL_POLICY,
        async_rl.SMOKE_PRESET,
        version="2026.09.18",
        settings=("trainer.rollout_buffer.max_in_flight=1",),
    )
    assert capped.rl.name != run.rl.name
    assert capped.evaluation.name != run.evaluation.name


@pytest.mark.timeout(180)  # A cold cache must install the locked CPU-safe launcher dependencies.
@pytest.mark.parametrize(
    ("preset", "training_type"),
    [(async_rl.DEFAULT, "async"), (async_rl.SMOKE_PRESET, "async"), (async_rl.ON_POLICY, "sync")],
    ids=["default", "smoke", "on-policy"],
)
def test_launcher_composes_pinned_rollout_buffer_config(preset, training_type, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    monkeypatch.setattr(async_rl, "username_segment", lambda: "alice")
    run = async_rl.build_run(SNOWBALL_POLICY, preset, version="2026.09.18")
    launch = run.rl.build_config(StepContext.for_fingerprint(run.rl.runtime_args, run.rl.deps))
    path = tmp_path / "launch.yaml"
    path.write_text(launch.launch_config_yaml)
    project = Path(__file__).resolve().parents[2] / "config/external/MarinSkyRL"
    # Use Marin's isolated lock so the test exercises the same SDK revision as the launcher.
    result = subprocess.run(
        ["uv", "run", "--frozen", "--project", str(project), "python", "-c", COMPOSE_LAUNCH, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "entrypoint": "skyrl_train.entrypoints.main_base",
        "training_type": training_type,
        "batch_policy": "rolling",
    }
