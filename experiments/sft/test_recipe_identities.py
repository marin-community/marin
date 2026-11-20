# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from levanter.models.snowball import SnowballConfig
from levanter.utils.mesh import MeshConfig

from experiments.post_training.curriculum_sft import math_trial, trial
from experiments.post_training.iceball_micro import build_workflow
from experiments.post_training.snowball_sft import MIXTURE_262K_32NODE, OPENTHOUGHTS_32K_4NODE, OPENTHOUGHTS_SPEC
from experiments.sft.configs import delphi_1e22
from experiments.sft.launcher import resources_from_accelerator, sft_step


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
    old_snowball = replace(OPENTHOUGHTS_SPEC, version="2026.09.23", mesh=old_mesh, pack=False)
    tpu = resources_from_accelerator("v6e-64")
    steps = (
        sft_step(old_snowball, OPENTHOUGHTS_32K_4NODE.resources),
        OPENTHOUGHTS_32K_4NODE.model().step,
        MIXTURE_262K_32NODE.model().step,
        sft_step(delphi_1e22.SPEC, tpu),
        sft_step(delphi_1e22.SPEC_WC50M, tpu),
        build_workflow(version="2026.11.19.1").sft,
        curriculum,
        conversion,
    )

    assert [(step.name, step.version, step.fingerprint()) for step in steps] == [
        ("checkpoints/snowball-openthoughts-agent-sft", "2026.09.23", "1c1f3fc7"),
        ("checkpoints/snowball-openthoughts-agent-sft", "2026.11.19.1", "6a7e5ad4"),
        ("checkpoints/snowball/mixture-262k-32node", "2026.11.19.1", "300cf21f"),
        ("checkpoints/delphi-1e22-magpie-warmup-levanter-sft", "2026.07.15", "0ca53670"),
        ("checkpoints/delphi-1e22-wc50m-warmup-levanter-sft", "2026.07.15", "9c258a76"),
        ("checkpoints/iceball-micro-sft", "2026.11.19.1", "03dbbead"),
        ("users/sft-fixture/checkpoints/curriculum-sft/1c980189d90b/snowball", "2026.11.19.1", "13f7f0db"),
        (
            "checkpoints/hf-to-levanter/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026-09-20-snowball",
            "2026.09.25.2",
            "cc7d2ded",
        ),
    ]
