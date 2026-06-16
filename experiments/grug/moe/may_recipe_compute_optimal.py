# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""May Recipe baseline at compute-optimal points, with the refit LR heuristic.

Three runs at the README compute-optimal anchors, using the **exact** step
counts the 1pct-noclip baselines (#5763) ran for so loss is directly
comparable. The LR comes from the refit ``MoeAdamHHeuristic`` (issue #5951):

    muonh_lr = 26.9 * tokens^-0.395 * dim^-0.121 * batch_size^0.5

Submit on us-east5-a, interactive priority, v5p-8:

    .venv/bin/iris --config lib/iris/config/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.may_recipe_compute_optimal
"""


from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic_v1 import MoeAdamHHeuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_SEQ_LEN: int = 4096
_TPU: str = "v5p-8"
_RUN_SUFFIX: str = "newlr-v2"
_GROUP_NAME: str = "may-recipe-compute-optimal-newlr"
_WARMUP_FRACTION: float = 0.01

# (hidden_dim, batch_size, num_steps) — step counts match the 1pct-noclip
# baselines: d512=6387, d768=10343, d1024=12649 (from
# `muonh-may-arch-gn-muonh-1pct-noclip-v1-d{512,768,1024}-*` configs).
_POINTS: tuple[tuple[int, int, int], ...] = (
    (512, 32, 6387),
    (768, 64, 10343),
    (1024, 128, 12649),
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

    run_id = f"grug-moe-may-recipe-newlr-d{hidden_dim}-{_RUN_SUFFIX}"
    step_name = f"grug/may_recipe_compute_optimal/{run_id}"

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
                    "may_recipe_compute_optimal",
                    "newlr",
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
            checkpoint_keep_every=None,
        ),
    )


if __name__ == "__main__":
    steps = [_build_step(d, bs, n) for (d, bs, n) in _POINTS]
    executor_main(
        steps=steps,
        description=(
            "May Recipe compute-optimal baseline with refit LR heuristic (#5951): "
            f"d/bs/steps in {_POINTS}, TPU={_TPU}, 1pct-noclip schedule (warmup=1%, "
            "max_grad_norm=None), no permanent step-interval checkpoints."
        ),
    )
