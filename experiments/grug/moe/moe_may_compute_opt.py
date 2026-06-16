# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""May Recipe compute-optimal baseline at d=512 and d=768.

Compute budgets derived from the drop-1e18 isoflop fit (issue #6074):

    opt_d(C) = 0.137 · C^0.2033   ->   C(d) = (d / 0.137)^(1 / 0.2033)

For d=512  -> C ≈ 3.82e17, tokens ≈ 1.44e9
For d=768  -> C ≈ 2.81e18, tokens ≈ 4.42e9

Batch sizes hardcoded to the README pattern (d=512 -> 32, d=768 -> 64).
LR / beta2 / epsilon come from the refit ``MoeAdamHHeuristic`` (issue #5951).

Submit on us-east5-a, interactive priority, v5p-8:

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt
"""


from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic_adamh import MoeAdamHHeuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_SEQ_LEN: int = 4096
_TPU: str = "v5p-8"
_GROUP_NAME: str = "moe-may-compute-opt"
_WARMUP_FRACTION: float = 0.01

# (hidden_dim, batch_size, num_steps) — from drop-1e18 isoflop fit + README bs pattern.
_POINTS: tuple[tuple[int, int, int], ...] = (
    (512, 32, 10980),
    (768, 64, 16875),
)


def _build_step(hidden_dim: int, batch_size: int, num_steps: int) -> ExecutorStep:
    h = MoeAdamHHeuristic()
    model = h.build_model_config(hidden_dim, seq_len=_SEQ_LEN)
    tokens = float(num_steps * batch_size * _SEQ_LEN)
    base_optimizer = h.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)

    optimizer = GrugMoeMuonHConfig(
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

    run_id = f"moe_may_compute_opt_d{hidden_dim}"
    step_name = f"grug/moe_may_compute_opt/{run_id}"

    return ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(_TPU)),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=[
                    "moe",
                    "moe_may_compute_opt",
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
            checkpoint_keep_every=1000,
        ),
    )


if __name__ == "__main__":
    steps = [_build_step(d, bs, n) for (d, bs, n) in _POINTS]
    executor_main(
        steps=steps,
        description=(
            "May Recipe compute-optimal at d=512/768 using drop-1e18 isoflop fit (#6074) "
            f"and refit LR heuristic (#5951): d/bs/steps in {_POINTS}, TPU={_TPU}, "
            "1pct-noclip schedule (warmup=1%, max_grad_norm=None), permanent checkpoint every 1000 steps."
        ),
    )
