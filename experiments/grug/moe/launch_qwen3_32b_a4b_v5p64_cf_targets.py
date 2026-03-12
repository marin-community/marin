# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quick capacity-factor sweep on the leading sab4 and es3r2 target shapes."""

from __future__ import annotations

from marin.execution.executor import executor_main

from experiments.grug.moe.launch_qwen3_32b_a4b_v5p64_ep_profile import build_step


def main() -> None:
    steps = []

    for cf, slug in ((1.0, "c1p0"), (1.1, "c1p1"), (1.25, "c1p25"), (1.5, "c1p5")):
        steps.append(
            build_step(
                expert_axis_size=4,
                batch_size=320,
                capacity_factor=cf,
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
                run_suffix=f"es3{slug}",
            )
        )
        steps.append(
            build_step(
                expert_axis_size=4,
                batch_size=320,
                capacity_factor=cf,
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
                run_suffix=f"sab4{slug}",
            )
        )

    executor_main(
        steps=steps,
        description="Quick capacity-factor sweep on sab4 and es3r2 target shapes on v5p-64.",
    )


if __name__ == "__main__":
    main()
