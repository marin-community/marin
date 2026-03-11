# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated rerun of the shared-heavy 3072x40 A/B leg."""

from __future__ import annotations

from marin.execution.executor import executor_main

from experiments.grug.moe.launch_qwen3_32b_a4b_v5p64_ep_profile import build_step


def main() -> None:
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
        run_suffix="sab2r",
    )

    executor_main(
        steps=[shared_heavier_3072x40],
        description="Isolated rerun of the shared-heavy 3072x40 geometry after forcing full-gang TPU retries.",
    )


if __name__ == "__main__":
    main()
