# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from levanter.models.snowball import SnowballConfig
from levanter.utils.mesh import MeshConfig

from experiments.post_training.curriculum_sft import math_trial, trial
from experiments.post_training.iceball_micro import build_workflow
from experiments.post_training.snowball_sft import OPENTHOUGHTS_SFT
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
    old_snowball = replace(OPENTHOUGHTS_SFT.recipe, version="2026.09.23", mesh=old_mesh)
    tpu = resources_from_accelerator("v6e-64")
    steps = (
        sft_step(old_snowball, OPENTHOUGHTS_SFT.resources),
        OPENTHOUGHTS_SFT.model().step,
        sft_step(delphi_1e22.SPEC, tpu),
        sft_step(delphi_1e22.SPEC_WC50M, tpu),
        build_workflow(version="2026.11.19.1").sft,
        curriculum,
        conversion,
    )

    assert [(step.name, step.version, step.fingerprint()) for step in steps] == [
        ("checkpoints/snowball-openthoughts-agent-sft", "2026.09.23", "bf7978a7"),
        ("checkpoints/snowball-openthoughts-agent-sft", "2026.11.19.1", "734e0b40"),
        ("checkpoints/delphi-1e22-magpie-warmup-levanter-sft", "2026.07.15", "681c760f"),
        ("checkpoints/delphi-1e22-wc50m-warmup-levanter-sft", "2026.07.15", "cb1e1554"),
        ("checkpoints/iceball-micro-sft", "2026.11.19.1", "03dbbead"),
        ("users/sft-fixture/checkpoints/curriculum-sft/1c980189d90b/snowball", "2026.11.19.1", "13f7f0db"),
        (
            "checkpoints/hf-to-levanter/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026-09-20-snowball",
            "2026.09.25.2",
            "cc7d2ded",
        ),
    ]


@pytest.mark.timeout(120)
def test_june_grug_sft_recipe_identity():
    # The June Grug and current Grug packages register different optimizers under the same Draccus name.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
from experiments.june_tpu_67b_a2b.moe import sft_67b_a2b_2stage

step = sft_67b_a2b_2stage.build_job1("2026.11.19.1")
print(json.dumps([step.name, step.version, step.fingerprint()]))
""",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == [
        "grug/grug_67b_a2b_sft_s1_wildchat",
        "2026.11.19.1",
        "4ff51899",
    ]
