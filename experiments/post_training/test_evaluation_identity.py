# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.models.snowball import SnowballConfig
from marin.execution.lazy import ArtifactStep
from marin.training.training import LevanterCheckpoint

from experiments.evals.evals import core_evals
from experiments.evals.evaluation import eval_step
from experiments.post_training import iceball_micro, skyrl_port_smoke
from experiments.post_training.curriculum_rl import launch
from experiments.post_training.curriculum_sft.math_trial import build_math_trial


def test_post_training_evaluation_identities_stay_pinned(monkeypatch):
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    monkeypatch.setattr(launch, "username_segment", lambda: "alice")
    monkeypatch.setattr("marin.experiment.checkpoints.resolve_lm_config", lambda *_: SnowballConfig())
    version = "2026.10.26"
    arm = launch.build_arms(specs=(launch.ARMS["naive"],), scale="smoke", version=version)["naive"]
    checkpoint = ArtifactStep.adopt("tests/model", version, "/models/checkpoint", kind=LevanterCheckpoint)

    steps = (
        iceball_micro.build_workflow(version=version).evaluation,
        skyrl_port_smoke.build_smoke(version).evaluate,
        arm.evaluation,
        build_math_trial(version, learning_rate=5e-5, warmup=0, data="glm")["after"],
        eval_step(checkpoint, core_evals()[0], version=version),
    )

    assert [(step.name, step.version, step.fingerprint()) for step in steps] == [
        ("evals/iceball-micro/gsm8k-smoke-aime24-smoke", "2026.10.26", "38aa8654"),
        ("evals/skyrl-port-smoke/gsm8k-smoke", "2026.10.26", "3b56f26e"),
        ("evals/alice-curriculum-rl-naive-smoke/gsm8k-smoke", "2026.10.26", "34198f43"),
        ("evals/curriculum-math-sep20-trained/olympiadbench-deterministic-math500", "2026.10.26", "6b3af6fe"),
        ("evaluation/curated/tests/model/core", "2026.10.26", "729d29aa"),
    ]
