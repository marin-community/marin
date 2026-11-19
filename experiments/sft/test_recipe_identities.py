# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.models.snowball import SnowballConfig

from experiments.post_training.curriculum_sft import math_trial, trial
from experiments.post_training.iceball_micro import build_workflow
from experiments.sft.configs import delphi_1e22, snowball_h100
from experiments.sft.launcher import LevanterSft, resources_from_accelerator, sft_model


def test_sft_recipe_identities():
    tpu = resources_from_accelerator("v6e-64")
    steps = (
        sft_model(snowball_h100.SFT, snowball_h100.RESOURCES).step,
        sft_model(delphi_1e22.SFT, tpu).step,
        sft_model(LevanterSft(delphi_1e22.SPEC_WC50M), tpu).step,
        build_workflow(version="2026.11.19.1").sft,
    )

    assert [(step.name, step.version, step.fingerprint()) for step in steps] == [
        ("checkpoints/snowball-openthoughts-agent-sft", "2026.11.19.1", "d3cfe782"),
        ("checkpoints/delphi-1e22-magpie-warmup-levanter-sft", "2026.07.15", "681c760f"),
        ("checkpoints/delphi-1e22-wc50m-warmup-levanter-sft", "2026.07.15", "cb1e1554"),
        ("checkpoints/iceball-micro-sft", "2026.11.19.1", "03dbbead"),
    ]


def test_curriculum_sft_retains_its_native_snowball_fingerprints(monkeypatch):
    def owned_name(name):
        return f"users/sft-fixture/{name}"

    monkeypatch.setattr(math_trial, "user_owned_name", owned_name)
    monkeypatch.setattr(trial, "user_owned_name", owned_name)
    monkeypatch.setattr("marin.experiment.checkpoints.resolve_lm_config", lambda *args: SnowballConfig(qk_mult=1.75))
    step = math_trial.build_math_trial("2026.11.19.1", 1e-5, 10, "glm")["train"]
    assert (step.name, step.version, step.fingerprint()) == (
        "users/sft-fixture/checkpoints/curriculum-sft/1c980189d90b/snowball",
        "2026.11.19.1",
        "13f7f0db",
    )
    [conversion] = [dep for dep in step.deps if dep.name.startswith("checkpoints/hf-to-levanter/")]
    assert conversion.fingerprint() == "cc7d2ded"
