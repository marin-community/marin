# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH + all GatedNorms -> MuonH routing sweep.

Same recipe as ``muonh_gn_adamh_sweep`` (MuonH for matrices, baseline
10% warmup) except the four GatedNorm instances (per-block
``attn_gated_norm`` and ``mlp_gated_norm``; model-level
``embed_gated_norm`` and ``final_gated_norm``) route to the ``muonh``
group — i.e., NS orthogonalization on their rank-128 rectangular
matrices, same treatment as the rest of the model's 2-D weight leaves.

Fixes the asymmetric default routing where only ``attn_gated_norm``
went to ``adam`` (via the ``attn_gate`` substring collision bug) and
the other three went to ``muonh``. This is the symmetric "muonh-up"
companion to:
- ``muonh_gn_adamh_sweep``: all four -> AdamH
- ``muonh_gn_adam_sweep``: all four -> plain Adam

Submit with:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_gn_muonh_sweep
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHGNMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

# d512, d768, d1024 — gate 1 + first gate-2 point.
_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
    (1024, 9.00e18),
)

_BASELINE_TARGET_STEPS: int = 2**14
_GROUP: str = "muonh-gn-muonh-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-gn-muonh-{suffix}d{hidden_dim}-{budget_label}"


def _muonh_optimizer_from_baseline(base_optimizer: GrugMoeAdamHConfig) -> GrugMoeMuonHGNMuonHConfig:
    return GrugMoeMuonHGNMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=base_optimizer.warmup,
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
    optimizer = _muonh_optimizer_from_baseline(base_optimizer)

    run_id = _format_run_id(hidden_dim=hidden_dim, budget=budget, run_suffix=run_suffix)
    step_name = f"grug/muonh_gn_muonh_sweep/{run_id}"

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
                tags=["moe", "muonh_gn_muonh_sweep", f"d{hidden_dim}"],
                group=_GROUP,
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=1e-4,
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
    steps = [_build_step(hidden_dim=d, budget=c, run_suffix=_RUN_SUFFIX) for d, c in _POINTS]
    executor_main(
        steps=steps,
        description=f"MoE MuonH + all GatedNorms -> MuonH (d512/d768/d1024, run_suffix={_RUN_SUFFIX!r}).",
    )
