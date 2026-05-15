# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + MuonH + GatedNorms routed through MuonH-with-column-normalization.

Builds on the may_arch architecture (256 experts, PKO, partial rope,
last_layer_pko, no router/logit z-loss, 2% warmup, embed -> AdamH) and
the may_arch optimizer skeleton, with one change: the 4 GatedNorm
instances route to a new ``muonh_col_norm`` group that inserts a
per-row/col norm equalization step between NS and the Frobenius
hyperball update.

Intuition: NS on a ``(hidden, 128)`` GatedNorm matrix unit-normalizes
the 128 columns (each ``hidden`` long), but leaves the ``hidden`` rows
(each 128 long) with non-uniform norms. The column-normalization step
equalizes those row norms to their mean before hyperball re-projects
to ``||W||_F``. Goal: cleaner update geometry for the rectangular
rank-128 matrices.

All LR scales default to 1.0x (unperturbed heuristic).

Single trial at d512 only, on us-central1 instead of the usual
us-east5-a (test bed for a different zone).

Submit with:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-central1-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_gn_muonh_colnorm_sweep
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
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHMayArchGNMuonHColNormConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_HIDDEN_DIM: int = 512
_BUDGET: float = 2.19e17
_WARMUP_FRACTION: float = 0.02
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-gn-muonh-colnorm-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-may-arch-gn-muonh-colnorm-{suffix}d{hidden_dim}-{budget_label}"


def _muonh_optimizer(base_optimizer: GrugMoeAdamHConfig) -> GrugMoeMuonHMayArchGNMuonHColNormConfig:
    return GrugMoeMuonHMayArchGNMuonHColNormConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=_WARMUP_FRACTION,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=base_optimizer.max_grad_norm,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )


def _build_step(hidden_dim: int, budget: float, run_suffix: str = "") -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    # may_arch architecture: 256 experts, PKO, partial rope, last_layer_pko, no z-losses.
    model = dataclasses.replace(
        model,
        num_experts=_NUM_EXPERTS,
        partial_key_offset="every_4th",
        use_partial_rope=True,
        last_layer_pko=True,
        router_z_loss_coef=0.0,
    )
    optimizer = _muonh_optimizer(base_optimizer)

    run_id = _format_run_id(hidden_dim, budget, run_suffix=run_suffix)
    step_name = f"grug/muonh_may_arch_gn_muonh_colnorm_sweep/{run_id}"

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
                tags=["moe", "muonh_may_arch_gn_muonh_colnorm_sweep", f"d{hidden_dim}"],
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
    step = _build_step(hidden_dim=_HIDDEN_DIM, budget=_BUDGET, run_suffix=_RUN_SUFFIX)
    executor_main(
        steps=[step],
        description=(
            f"MoE may_arch + MuonH + GN -> muonh-col-norm (d{_HIDDEN_DIM} only, " f"run_suffix={_RUN_SUFFIX!r})."
        ),
    )
