# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and export a one-rack Mochi SFT recipe."""

from fray.types import ResourceConfig

from experiments.grug.moe_hero_ep.launch_hero_sft import SFT_CAPACITY_FACTOR, sft_resources
from experiments.grug.moe_hero_ep.sft_model import MochiSft
from experiments.grug.moe_hero_ep.train import RouterBiasMode
from experiments.sft.launcher import run_sft_cli

SFT = MochiSft(
    run_id="mochi-sft",
    initialize_from=(
        "s3://marin-us-east-02a/marin/grug/hero-lce-log65536sq-step390k/2026.08.19.2/checkpoints/step-393491"
    ),
    start_step=393491,
    muon_lr=3.2905e-5,
    adam_lr=7.5934e-6,
    max_grad_norm=1.0,
    replay_fraction=0.2035,
    dp_racks=1,
    router_bias_mode=RouterBiasMode.FROZEN,
    capacity_factor=SFT_CAPACITY_FACTOR,
    version="2026.11.19.1",
    source_revision="2515cb1404e03b444a5e427fd7b9aaef2c034af1",
    export_resources=ResourceConfig.with_cpu(
        cpu=64, ram="256g", disk="512g", regions=("US-EAST-08A",), target_cluster="cw-us-east-08a"
    ),
    tokenizer_uri="marin-community/marin-tokenizer-tools",
    tokenizer_revision="3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09",
)
RESOURCES = sft_resources(SFT.dp_racks)


if __name__ == "__main__":
    run_sft_cli(SFT, RESOURCES)
