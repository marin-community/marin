# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Targeted shared-expert A/B/C/D around the 3072-wide baseline."""

from __future__ import annotations

from marin.execution.executor import executor_main

from experiments.grug.moe.launch_qwen3_32b_a4b_v5p64_ep_profile import build_step


def main() -> None:
    baseline = build_step(
        expert_axis_size=4,
        batch_size=320,
        capacity_factor=1.25,
        block_remat="offload_mlp",
        report_capacity_overflow=False,
        num_experts=64,
        num_experts_per_token=4,
        match_activated_params=True,
        shared_expert_intermediate_dim=2048,
        match_total_active_flops=False,
        block_shuffle=False,
        synthetic_data=True,
        loader_prefetch_size=32,
        loader_max_buffered_batches=64,
        steps=18,
        profiler_start_step=8,
        profiler_num_steps=5,
        cross_entropy_implementation="xla",
        cross_entropy_v_block_divisor=1,
        hidden_dim=3072,
        num_layers=32,
        num_heads=24,
        num_kv_heads=4,
        run_suffix="sab1",
    )
    shared_heavier_3072x40 = build_step(
        expert_axis_size=4,
        batch_size=320,
        capacity_factor=1.25,
        block_remat="offload_mlp",
        report_capacity_overflow=False,
        num_experts=64,
        num_experts_per_token=4,
        match_activated_params=True,
        shared_expert_intermediate_dim=4096,
        match_total_active_flops=True,
        block_shuffle=False,
        synthetic_data=True,
        loader_prefetch_size=32,
        loader_max_buffered_batches=64,
        steps=18,
        profiler_start_step=8,
        profiler_num_steps=5,
        cross_entropy_implementation="xla",
        cross_entropy_v_block_divisor=1,
        hidden_dim=3072,
        num_layers=40,
        num_heads=24,
        num_kv_heads=4,
        run_suffix="sab2",
    )
    shared_heavier_3584x37 = build_step(
        expert_axis_size=4,
        batch_size=320,
        capacity_factor=1.25,
        block_remat="offload_mlp",
        report_capacity_overflow=False,
        num_experts=64,
        num_experts_per_token=4,
        match_activated_params=True,
        shared_expert_intermediate_dim=5120,
        match_total_active_flops=True,
        block_shuffle=False,
        synthetic_data=True,
        loader_prefetch_size=32,
        loader_max_buffered_batches=64,
        steps=18,
        profiler_start_step=8,
        profiler_num_steps=5,
        cross_entropy_implementation="xla",
        cross_entropy_v_block_divisor=1,
        hidden_dim=3584,
        num_layers=37,
        num_heads=28,
        num_kv_heads=4,
        run_suffix="sab3",
    )
    shared_heavier_4096x27 = build_step(
        expert_axis_size=4,
        batch_size=320,
        capacity_factor=1.25,
        block_remat="offload_mlp",
        report_capacity_overflow=False,
        num_experts=64,
        num_experts_per_token=4,
        match_activated_params=True,
        shared_expert_intermediate_dim=4096,
        match_total_active_flops=True,
        block_shuffle=False,
        synthetic_data=True,
        loader_prefetch_size=32,
        loader_max_buffered_batches=64,
        steps=18,
        profiler_start_step=8,
        profiler_num_steps=5,
        cross_entropy_implementation="xla",
        cross_entropy_v_block_divisor=1,
        hidden_dim=4096,
        num_layers=27,
        num_heads=32,
        num_kv_heads=4,
        run_suffix="sab4",
    )

    executor_main(
        steps=[baseline, shared_heavier_3072x40, shared_heavier_3584x37, shared_heavier_4096x27],
        description=(
            "Compare the 3072x32 E64/topk4 baseline against the three closest shared-heavier "
            "FLOP-matched variants: 3072x40 shared=4096, 3584x37 shared=5120, and 4096x27 shared=4096."
        ),
    )


if __name__ == "__main__":
    main()
