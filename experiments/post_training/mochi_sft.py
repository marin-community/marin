# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the production Mochi SFT run ``hero-sft-fullab-step393k-v2`` from issue #8089.

The production launch in comment 6771184334 pins step 393491, 11 racks, both
learning rates, clipping, replay share, expert placement, evaluation cadence, and
capacity factor. The launcher's frozen router bias and full mixture schedule
produce the permanent step 400407 reported in comment 6841382920.
"""

from dataclasses import replace
from pathlib import Path

from fray.types import ResourceConfig

from experiments.grug.moe_hero_ep import launch_hero_sft
from experiments.grug.moe_hero_ep.expert_placement import ExpertPlacement
from experiments.grug.moe_hero_ep.sft_recipe import GrugHeroSFT
from experiments.grug.moe_hero_ep.train import RouterBiasMode
from experiments.sft.launcher import SFTPreset, main

MOCHI_SPEC = GrugHeroSFT(
    run_id="hero-sft-fullab-step393k-v2",
    initialize_from=(
        "s3://marin-us-east-02a/marin/grug/hero-lce-log65536sq-step390k/2026.08.19.2/checkpoints/step-393491"
    ),
    muon_lr=3.2905e-5,
    adam_lr=3.2905e-5,
    max_grad_norm=1.0,
    replay_fraction=0.2035,
    dp_racks=11,
    router_bias_mode=RouterBiasMode.FROZEN,
    capacity_factor=1.625,
    eval_every=500,
    expert_placement=ExpertPlacement.from_json(
        Path(launch_hero_sft.__file__).with_name("hero_sft_expert_placement_step393491.json").read_text()
    ),
    version="2026.11.10",
    source_revision="406dff47f217637ceca499b266989588cb0df5ab",
    export_resources=ResourceConfig.with_cpu(
        cpu=64, ram="256g", disk="512g", regions=("US-EAST-08A",), target_cluster="cw-us-east-08a"
    ),
    # The pinned Hub tokenizer, config, and chat template match the training copy at S3 version 2026.09.18 by SHA-256.
    tokenizer_uri="marin-community/marin-tokenizer-tools",
    tokenizer_revision="3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09",
)
MIXTURE_262K_176NODE = SFTPreset(
    "mixture-262k-176node",
    MOCHI_SPEC,
    replace(launch_hero_sft.sft_resources(MOCHI_SPEC.dp_racks), regions=("US-EAST-08A",), target_cluster="cw-us-east-08a"),
)
MOCHI_SFT = (MIXTURE_262K_176NODE,)


if __name__ == "__main__":
    main(MOCHI_SFT)
