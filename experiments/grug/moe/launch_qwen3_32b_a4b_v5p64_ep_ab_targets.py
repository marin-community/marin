# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quick EP=4/8 A/B for the sab4 and es3r2 target shapes."""

from __future__ import annotations

from marin.execution.executor import executor_main

from experiments.grug.moe.launch_qwen3_32b_a4b_v5p64_ep_profile import build_step


def main() -> None:
    es3_ep4 = build_step(
        expert_axis_size=4,
        batch_size=320,
        capacity_factor=1.25,
        block_remat="offload_mlp",
        report_capacity_overflow=False,
        num_experts=64,
        num_experts_per_token=4,
        intermediate_dim=1024,
        match_activated_params=False,
        shared_expert_intermediate_dim=1024,
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
        hidden_dim=4096,
        num_layers=27,
        num_heads=32,
        num_kv_heads=4,
        run_suffix="es3e4",
    )
    es3_ep8 = build_step(
        expert_axis_size=8,
        batch_size=320,
        capacity_factor=1.25,
        block_remat="offload_mlp",
        report_capacity_overflow=False,
        num_experts=64,
        num_experts_per_token=4,
        intermediate_dim=1024,
        match_activated_params=False,
        shared_expert_intermediate_dim=1024,
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
        hidden_dim=4096,
        num_layers=27,
        num_heads=32,
        num_kv_heads=4,
        run_suffix="es3e8",
    )
    sab4_ep4 = build_step(
        expert_axis_size=4,
        batch_size=320,
        capacity_factor=1.25,
        block_remat="offload_mlp",
        report_capacity_overflow=False,
        num_experts=64,
        num_experts_per_token=4,
        intermediate_dim=None,
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
        run_suffix="sab4e4t",
    )
    sab4_ep8 = build_step(
        expert_axis_size=8,
        batch_size=320,
        capacity_factor=1.25,
        block_remat="offload_mlp",
        report_capacity_overflow=False,
        num_experts=64,
        num_experts_per_token=4,
        intermediate_dim=None,
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
        run_suffix="sab4e8t",
    )

    executor_main(
        steps=[es3_ep4, es3_ep8, sab4_ep4, sab4_ep8],
        description="Quick EP=4 vs EP=8 A/B for es3r2 and sab4 on v5p-64.",
    )


if __name__ == "__main__":
    main()
