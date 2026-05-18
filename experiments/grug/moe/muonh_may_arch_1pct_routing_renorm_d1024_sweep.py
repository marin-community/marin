# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + 1pct-noclip + routing-renorm X in {2.0, 2.5, 3.0} at d1024.

Confirmation sweep for the d768 winners from #5797 (X=2.5 best at
Delta -0.0081; X=3.0 / X=2.0 tied for second cluster). Targets d1024 /
9.00e18 FLOPs.

Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_routing_renorm_d1024_sweep
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

_HIDDEN_DIM: int = 1024
_BUDGET: float = 9.00e18
_X_VALUES: tuple[float, ...] = (2.0, 2.5, 3.0)

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-1pct-routing-renorm-sweep"
_RUN_SUFFIX: str = "v1"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_x(x: float) -> str:
    return f"{x:.1f}".replace(".", "p")


def _build_step(x_value: float) -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    model = dataclasses.replace(
        model,
        num_experts=_NUM_EXPERTS,
        partial_key_offset="every_4th",
        use_partial_rope=True,
        last_layer_pko=True,
        router_z_loss_coef=0.0,
        routing_renorm_sum=x_value,
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
    )

    run_id = (
        f"muonh-may-arch-1pct-routing-renorm-x{_format_x(x_value)}-{_RUN_SUFFIX}-"
        f"d{_HIDDEN_DIM}-{_format_budget(_BUDGET)}"
    )
    step_name = f"grug/muonh_may_arch_1pct_routing_renorm_sweep/{run_id}"

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
                    "muonh_may_arch_1pct_routing_renorm_sweep",
                    f"d{_HIDDEN_DIM}",
                    f"x{_format_x(x_value)}",
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
    steps = [_build_step(x) for x in _X_VALUES]
    executor_main(
        steps=steps,
        description=(
            f"MoE may_arch + 1pct-noclip + routing-renorm X in {list(_X_VALUES)} at d{_HIDDEN_DIM} "
            f"(run_suffix={_RUN_SUFFIX!r})."
        ),
    )
