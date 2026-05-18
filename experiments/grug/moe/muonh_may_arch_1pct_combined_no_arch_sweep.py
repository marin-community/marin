# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combined kitchen-sink minus architecture changes: K=4 / full shared MLP.

Companion to ``muonh_may_arch_1pct_combined_kitchen_sink_sweep`` (#5806)
which stacked four feature flags plus K=5 + half-sized shared MLP.
This variant **reverts the two architecture changes** (K back to 4,
shared MLP back to ``hidden_dim``) but keeps the four feature flags:

- ``routing_renorm_sum=2.5`` (#5797)
- ``split_w_gate_up=True`` (#5794)
- ``pko_norm_order="pko_first_bos_zero"``
- ``embed_adam_lr_scale=1.0`` (token_embed -> plain Adam at 1.0x adam_lr) (#5804)

Goal: isolate whether the kitchen-sink wins come from the feature
flags alone, or whether K=5 + half-shared are pulling additional
weight. 3 runs at d512 / d768 / d1024.

Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_combined_no_arch_sweep
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeMuonHMayArchGNMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
    (1024, 9.00e18),
)

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_K: int = 4  # reverted from 5
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-1pct-combined-no-arch-sweep"
_RUN_SUFFIX: str = "v1"
_ROUTING_RENORM_SUM: float = 2.5
_EMBED_ADAM_LR_SCALE: float = 1.0


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _build_step(hidden_dim: int, budget: float) -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    model = dataclasses.replace(
        model,
        num_experts=_NUM_EXPERTS,
        num_experts_per_token=_K,  # K=4 (reverted from 5)
        # shared_expert_intermediate_dim left at heuristic default (= hidden_dim)
        partial_key_offset="every_4th",
        use_partial_rope=True,
        last_layer_pko=True,
        router_z_loss_coef=0.0,
        routing_renorm_sum=_ROUTING_RENORM_SUM,
        split_w_gate_up=True,
        pko_norm_order="pko_first_bos_zero",
    )
    optimizer = GrugMoeMuonHMayArchGNMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=_WARMUP_FRACTION,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
        embed_adam_lr_scale=_EMBED_ADAM_LR_SCALE,
    )

    run_id = f"muonh-may-arch-1pct-combined-no-arch-{_RUN_SUFFIX}-d{hidden_dim}-{_format_budget(budget)}"
    step_name = f"grug/muonh_may_arch_1pct_combined_no_arch_sweep/{run_id}"

    return ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=[
                    "moe",
                    "muonh_may_arch_1pct_combined_no_arch_sweep",
                    f"d{hidden_dim}",
                ],
                group=_GROUP_NAME,
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=0.0,
                    ema_beta=None,
                    log_every=1,
                )
            ),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
            enable_cross_region_ckpt_read=True,
        ),
    )


if __name__ == "__main__":
    steps = [_build_step(hidden_dim=d, budget=c) for d, c in _POINTS]
    executor_main(
        steps=steps,
        description=(
            "MoE may_arch + 1pct-noclip combined (no arch changes): K=4, full shared, "
            "routing-renorm X=2.5, split, pko_first_bos_zero, embed->Adam (1.0x). "
            f"d512/d768/d1024 (run_suffix={_RUN_SUFFIX!r})."
        ),
    )
