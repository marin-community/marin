# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retry equal-shared es3 as a single-step job."""

from __future__ import annotations

from marin.execution.executor import executor_main

from experiments.grug.moe.launch_qwen3_32b_a4b_v5p64_ep_profile import build_step


def main() -> None:
    executor_main(
        steps=[
            build_step(
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
                run_suffix="es3r3",
            )
        ],
        description="Retry es3 equal-shared geometry as a single-step job.",
    )


if __name__ == "__main__":
    main()
