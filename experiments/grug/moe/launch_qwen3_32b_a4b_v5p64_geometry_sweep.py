# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Overnight Grug MoE geometry/remat sweep on v5p-64."""

from __future__ import annotations

from dataclasses import dataclass

from marin.execution.executor import executor_main

from experiments.grug.moe.launch_qwen3_32b_a4b_v5p64_ep_profile import build_step


@dataclass(frozen=True)
class SweepVariant:
    name: str
    block_remat: str = "offload_mlp"
    hidden_dim: int = 2048
    num_layers: int = 48
    num_heads: int = 32
    num_kv_heads: int = 4
    shared_expert_intermediate_dim: int = 2048
    match_total_active_flops: bool = False


VARIANTS: tuple[SweepVariant, ...] = (
    SweepVariant("g01"),
    SweepVariant("g02", block_remat="full"),
    SweepVariant("g03", block_remat="offload_mlp_inputs"),
    SweepVariant("g04", block_remat="offload_mlp_outputs"),
    SweepVariant("g05", num_layers=40),
    SweepVariant("g06", num_layers=36),
    SweepVariant("g07", shared_expert_intermediate_dim=3072, match_total_active_flops=True),
    SweepVariant("g08", shared_expert_intermediate_dim=4096, match_total_active_flops=True),
    SweepVariant("g09", shared_expert_intermediate_dim=5120, match_total_active_flops=True),
    SweepVariant("g10", num_layers=40, shared_expert_intermediate_dim=4096, match_total_active_flops=True),
    SweepVariant("g11", num_layers=36, shared_expert_intermediate_dim=4096, match_total_active_flops=True),
    SweepVariant("g12", hidden_dim=2560, num_layers=40, num_heads=20, num_kv_heads=5),
    SweepVariant("g13", hidden_dim=2560, num_layers=36, num_heads=20, num_kv_heads=5),
    SweepVariant(
        "g14",
        hidden_dim=2560,
        num_layers=40,
        num_heads=20,
        num_kv_heads=5,
        shared_expert_intermediate_dim=4096,
        match_total_active_flops=True,
    ),
    SweepVariant("g15", hidden_dim=3072, num_layers=32, num_heads=24, num_kv_heads=6),
    SweepVariant(
        "g16",
        hidden_dim=3072,
        num_layers=32,
        num_heads=24,
        num_kv_heads=6,
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
            block_remat=variant.block_remat,
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
            "Overnight v5p-64 geometry sweep for the E64/topk4 matched-active hybrid path. "
            "This sequence explores fewer layers, wider hidden states, shared-heavier matched-FLOPs "
            "variants, and conservative remat/offload policies under the stable XLA CE path."
        ),
    )


if __name__ == "__main__":
    main()
