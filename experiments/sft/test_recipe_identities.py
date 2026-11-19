# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from levanter.models.snowball import SnowballConfig
from levanter.utils.mesh import MeshConfig

from experiments.june_tpu_67b_a2b.moe import sft_67b_a2b_2stage as grug
from experiments.post_training.curriculum_sft import math_trial, trial
from experiments.post_training.iceball_micro import build_workflow
from experiments.sft.configs import delphi_1e22, snowball_h100
from experiments.sft.launcher import LevanterSft, resources_from_accelerator, sft_model


def test_sft_recipe_identities(monkeypatch):
    def owned_name(name):
        return f"users/sft-fixture/{name}"

    monkeypatch.setattr(math_trial, "user_owned_name", owned_name)
    monkeypatch.setattr(trial, "user_owned_name", owned_name)
    monkeypatch.setattr("marin.experiment.checkpoints.resolve_lm_config", lambda *args: SnowballConfig(qk_mult=1.75))
    curriculum = math_trial.build_math_trial("2026.11.19.1", 1e-5, 10, "glm")["train"]
    [conversion] = [dep for dep in curriculum.deps if dep.name.startswith("checkpoints/hf-to-levanter/")]

    old_mesh = MeshConfig(
        axes={"data": 1, "replica": 1, "model": 1, "context": 4, "expert": -1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context"},
    )
    old_snowball = LevanterSft(replace(snowball_h100.SPEC, version="2026.09.23", mesh=old_mesh))
    tpu = resources_from_accelerator("v6e-64")
    steps = (
        sft_model(old_snowball, snowball_h100.RESOURCES).step,
        sft_model(snowball_h100.SFT, snowball_h100.RESOURCES).step,
        sft_model(delphi_1e22.SFT, tpu).step,
        sft_model(LevanterSft(delphi_1e22.SPEC_WC50M), tpu).step,
        build_workflow(version="2026.11.19.1").sft,
        grug.build_job1("2026.11.19.1"),
        curriculum,
        conversion,
    )

    assert [(step.name, step.version, step.fingerprint()) for step in steps] == [
        ("checkpoints/snowball-openthoughts-agent-sft", "2026.09.23", "bf7978a7"),
        ("checkpoints/snowball-openthoughts-agent-sft", "2026.11.19.1", "734e0b40"),
        ("checkpoints/delphi-1e22-magpie-warmup-levanter-sft", "2026.07.15", "681c760f"),
        ("checkpoints/delphi-1e22-wc50m-warmup-levanter-sft", "2026.07.15", "cb1e1554"),
        ("checkpoints/iceball-micro-sft", "2026.11.19.1", "03dbbead"),
        ("grug/grug_67b_a2b_sft_s1_wildchat", "2026.11.19.1", "4ff51899"),
        ("users/sft-fixture/checkpoints/curriculum-sft/1c980189d90b/snowball", "2026.11.19.1", "13f7f0db"),
        (
            "checkpoints/hf-to-levanter/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026-09-20-snowball",
            "2026.09.25.2",
            "cc7d2ded",
        ),
    ]
