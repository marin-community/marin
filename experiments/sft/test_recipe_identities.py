# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

from levanter.models.snowball import SnowballConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep import launch_hero_sft
from experiments.post_training.curriculum_sft import math_trial, trial
from experiments.post_training.iceball_micro import build_workflow
from experiments.post_training.mochi_sft import MIXTURE_262K_176NODE
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
    monkeypatch.setattr("experiments.grug.moe_hero_ep.sft_recipe.checkpoint_step", lambda _: 393491)
    anchors = {
        text: [token_id]
        for (_, text), token_id in zip(
            launch_hero_sft.SPECIAL_TOKEN_ANCHORS, launch_hero_sft.EXPECTED_ANCHOR_IDS, strict=True
        )
    }
    tokenizer = SimpleNamespace(encode=lambda text, *, add_special_tokens: anchors[text])
    monkeypatch.setattr(launch_hero_sft, "load_tokenizer", lambda _: tokenizer)
    production = MIXTURE_262K_176NODE
    mochi_export = production.model().step
    relocated = production.recipe.train_artifact(production.resources, prefix="/tmp/mochi-production")
    config = relocated.build_config(StepContext.for_fingerprint(relocated.runtime_args.keys(), relocated.deps))
    assert config.stop_step == 400407
    assert config.trainer.save_checkpoints
    assert config.trainer.trainer.checkpointer.keep == [{"every": 500}]

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
        *mochi_export.deps,
        mochi_export,
    )

    assert [(step.name, step.version, step.fingerprint()) for step in steps] == [
        ("checkpoints/snowball-openthoughts-agent-sft", "2026.09.23", "fa196db1"),
        ("checkpoints/snowball-openthoughts-agent-sft", "2026.11.19.1", "fff2b724"),
        ("checkpoints/snowball/mixture-262k-32node", "2026.11.19.1", "80452e82"),
        ("checkpoints/delphi-1e22-magpie-warmup-levanter-sft", "2026.07.15", "72dd9435"),
        ("checkpoints/delphi-1e22-wc50m-warmup-levanter-sft", "2026.07.15", "0b4d0e44"),
        ("checkpoints/iceball-micro-sft", "2026.11.19.1", "d7ef5765"),
        ("users/sft-fixture/checkpoints/curriculum-sft/1c980189d90b/snowball", "2026.11.19.1", "895d3bdf"),
        (
            "checkpoints/hf-to-levanter/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026-09-20-snowball",
            "2026.09.25.2",
            "c939e384",
        ),
        ("grug/hero-sft-fullab-step393k-v2", "2026.11.10", "8d43d707"),
        ("grug/hero-sft-fullab-step393k-v2-hf", "2026.11.10", "df950f55"),
    ]
