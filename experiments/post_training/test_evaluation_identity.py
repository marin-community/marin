# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training import iceball_micro, skyrl_port_smoke
from experiments.post_training.curriculum_rl import launch


def test_post_training_evaluation_identities_stay_pinned(monkeypatch):
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    monkeypatch.setattr(launch, "username_segment", lambda: "alice")
    version = "2026.10.26"
    arm = launch.build_arms(specs=(launch.ARMS["naive"],), scale="smoke", version=version)["naive"]

    steps = (
        iceball_micro.build_workflow(version=version).evaluation,
        skyrl_port_smoke.build_smoke(version).evaluate,
        arm.evaluation,
    )

    # The judged evaluations bind generation plans (the merged entrypoint contract
    # derives the entrypoint from the plan) and the epoch mechanism's store controls,
    # which shifts the pinned fingerprints from the a0268301/3b56f26e/34198f43 set
    # the role-plan form produced.
    assert [(step.name, step.version, step.fingerprint()) for step in steps] == [
        ("evals/iceball-micro/gsm8k-smoke-aime24-smoke", "2026.10.26", "ef8f5cc8"),
        ("evals/skyrl-port-smoke/gsm8k-smoke", "2026.10.26", "db22a44b"),
        ("evals/alice-curriculum-rl-naive-smoke/gsm8k-smoke", "2026.10.26", "7dfc8ee4"),
    ]
