# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-run smaller hybrid profile on v4-128."""

from __future__ import annotations

import dataclasses

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main, versioned

from experiments.grug.moe.launch_qwen3_32b_a4b_v5p64_ep_profile import (
    build_step,
    run_grug_moe_v5p_ep_profile,
)


def main() -> None:
    step = build_step(
        expert_axis_size=4,
        batch_size=320,
        capacity_factor=1.25,
        block_remat="offload_mlp",
        report_capacity_overflow=False,
        num_experts=64,
        num_experts_per_token=3,
        intermediate_dim=2048,
        match_activated_params=False,
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
        run_suffix="v4128k3s2",
    )
    config = dataclasses.replace(step.config, resources=versioned(ResourceConfig.with_tpu("v4-128")))
    executor_main(
        steps=[dataclasses.replace(step, fn=run_grug_moe_v5p_ep_profile, config=config)],
        description="Single smaller hybrid Grug MoE profile on v4-128 for cross-hardware comparison.",
    )


if __name__ == "__main__":
    main()
