# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + 1pct-noclip + token_embed on plain Adam, LR multiplier sweep.

Tests whether moving ``token_embed`` from AdamH (the scale-invariant
Frobenius-hyperball variant) back to plain Adam helps. AdamH on the
embed projects updates to keep ``||p_embed||_F`` constant; plain Adam
lets the embed norm drift freely.

Sweep three LR multipliers on top of the default ``adam_lr``:

    embed_adam_lr = embed_adam_lr_scale * adam_lr

multiplier in {0.7, 1.0, 1.3}, at d512 / d768 / d1024. Three multipliers
x three scales = 9 runs.

All other routing identical to the 1pct-noclip baseline (#5763); only
the ``token_embed`` group changes.

Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_embed_adam_lr_sweep
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
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHMayArchGNMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
    (1024, 9.00e18),
)

_LR_SCALES: tuple[float, ...] = (0.7, 1.0, 1.3)

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-1pct-embed-adam-lr-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_lr_scale(scale: float) -> str:
    return f"{scale:.1f}".replace(".", "p")


def _format_run_id(hidden_dim: int, budget: float, lr_scale: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-may-arch-1pct-embed-adam-lr-{_format_lr_scale(lr_scale)}-" f"{suffix}d{hidden_dim}-{budget_label}"


def _muonh_optimizer(base_optimizer: GrugMoeAdamHConfig, lr_scale: float) -> GrugMoeMuonHMayArchGNMuonHConfig:
    return GrugMoeMuonHMayArchGNMuonHConfig(
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
        embed_adam_lr_scale=lr_scale,
    )


def _build_step(hidden_dim: int, budget: float, lr_scale: float, run_suffix: str = "") -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    model = dataclasses.replace(
        model,
        num_experts=_NUM_EXPERTS,
        partial_key_offset="every_4th",
        use_partial_rope=True,
        last_layer_pko=True,
        router_z_loss_coef=0.0,
    )
    optimizer = _muonh_optimizer(base_optimizer, lr_scale=lr_scale)

    run_id = _format_run_id(hidden_dim, budget, lr_scale, run_suffix=run_suffix)
    step_name = f"grug/muonh_may_arch_1pct_embed_adam_lr_sweep/{run_id}"

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
                    "muonh_may_arch_1pct_embed_adam_lr_sweep",
                    f"d{hidden_dim}",
                    f"lr{_format_lr_scale(lr_scale)}",
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


_RUN_SUFFIX: str = "v1"


if __name__ == "__main__":
    steps = [
        _build_step(hidden_dim=d, budget=c, lr_scale=lr, run_suffix=_RUN_SUFFIX) for d, c in _POINTS for lr in _LR_SCALES
    ]
    executor_main(
        steps=steps,
        description=(
            "MoE may_arch + 1pct-noclip + token_embed on plain Adam, LR scale sweep "
            f"(lr_scales={list(_LR_SCALES)}, d512/d768/d1024, run_suffix={_RUN_SUFFIX!r})."
        ),
    )
