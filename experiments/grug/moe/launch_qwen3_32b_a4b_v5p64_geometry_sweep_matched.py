# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Overnight Grug MoE geometry sweep with approximate baseline-FLOP matching."""

from __future__ import annotations

from dataclasses import dataclass

from marin.execution.executor import executor_main

from experiments.grug.moe.launch_qwen3_32b_a4b_v5p64_ep_profile import build_step


@dataclass(frozen=True)
class MatchedVariant:
    name: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    shared_expert_intermediate_dim: int = 2048
    match_total_active_flops: bool = False


VARIANTS: tuple[MatchedVariant, ...] = (
    MatchedVariant("m01", hidden_dim=2048, num_layers=48, num_heads=16, num_kv_heads=4),
    MatchedVariant("m02", hidden_dim=2304, num_layers=41, num_heads=18, num_kv_heads=6),
    MatchedVariant("m03", hidden_dim=2560, num_layers=36, num_heads=20, num_kv_heads=4),
    MatchedVariant("m04", hidden_dim=2688, num_layers=33, num_heads=21, num_kv_heads=7),
    MatchedVariant("m05", hidden_dim=3072, num_layers=29, num_heads=24, num_kv_heads=4),
    MatchedVariant("m06", hidden_dim=3200, num_layers=27, num_heads=25, num_kv_heads=5),
    MatchedVariant(
        "m07",
        hidden_dim=3072,
        num_layers=32,
        num_heads=24,
        num_kv_heads=4,
        shared_expert_intermediate_dim=3072,
        match_total_active_flops=True,
    ),
    MatchedVariant(
        "m08",
        hidden_dim=3072,
        num_layers=36,
        num_heads=24,
        num_kv_heads=4,
        shared_expert_intermediate_dim=4096,
        match_total_active_flops=True,
    ),
    MatchedVariant(
        "m09",
        hidden_dim=3840,
        num_layers=30,
        num_heads=30,
        num_kv_heads=5,
        shared_expert_intermediate_dim=5120,
        match_total_active_flops=True,
    ),
    MatchedVariant(
        "m10",
        hidden_dim=3200,
        num_layers=34,
        num_heads=25,
        num_kv_heads=5,
        shared_expert_intermediate_dim=4096,
        match_total_active_flops=True,
    ),
)


def main() -> None:
    steps = [
        build_step(
            expert_axis_size=4,
            batch_size=320,
            capacity_factor=1.25,
            block_remat="offload_mlp",
            report_capacity_overflow=False,
            num_experts=64,
            num_experts_per_token=4,
            match_activated_params=True,
            shared_expert_intermediate_dim=variant.shared_expert_intermediate_dim,
            match_total_active_flops=variant.match_total_active_flops,
            block_shuffle=False,
            synthetic_data=True,
            loader_prefetch_size=32,
            loader_max_buffered_batches=64,
            steps=18,
            profiler_start_step=8,
            profiler_num_steps=5,
            cross_entropy_implementation="xla",
            cross_entropy_v_block_divisor=1,
            hidden_dim=variant.hidden_dim,
            num_layers=variant.num_layers,
            num_heads=variant.num_heads,
            num_kv_heads=variant.num_kv_heads,
            run_suffix=variant.name,
        )
        for variant in VARIANTS
    ]

    executor_main(
        steps=steps,
        description=(
            "Overnight v5p-64 geometry sweep with variants chosen to stay approximately FLOP-matched "
            "to the E64/topk4/shared2048 baseline. This focuses on width/layer tradeoffs and "
            "shared-heavier matched-active-FLOP variants under the stable XLA CE path."
        ),
    )


if __name__ == "__main__":
    main()
