# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""June 10T-token d=2560 MoE run on v4-1024 us-central2.

Big-batch d=2560 (2.01B active / 66.4B total core; +0.66B embed+lm_head) on
10T tokens via the May Recipe, with optimizer hyperparameters from the refit
``MoeAdamHHeuristic`` (issue #5951). bs=2048 (TPB = 8.39M), 1.19M steps,
~1.21e23 FLOPs total compute. EP=4.

Submit on us-central2, production priority, v4-1024:

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --region us-central2 \\
      --priority production \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_d2560_10T
"""


from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import MoeAdamHHeuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_SEQ_LEN: int = 4096
_TPU: str = "v4-1024"
_GROUP_NAME: str = "moe-may-d2560-10T"
_WARMUP_FRACTION: float = 0.01

_HIDDEN_DIM: int = 2560
_BATCH_SIZE: int = 2048
_NUM_STEPS: int = 1_192_093  # ~10T tokens = NUM_STEPS · BATCH_SIZE · SEQ_LEN
_EXPERT_PARALLEL: int = 4
_CHECKPOINT_KEEP_EVERY: int = 5000


def _build_step() -> ExecutorStep:
    h = MoeAdamHHeuristic()
    model = h.build_model_config(_HIDDEN_DIM, seq_len=_SEQ_LEN)
    tokens = float(_NUM_STEPS * _BATCH_SIZE * _SEQ_LEN)
    base_optimizer = h.build_optimizer_config(_BATCH_SIZE, tokens, _HIDDEN_DIM, seq_len=_SEQ_LEN)

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

    run_id = "marin-big-run-moe_may_d2560_10T"
    step_name = f"grug/moe_may_d2560_10T/{run_id}"

    return ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(_TPU)),
            steps=versioned(_NUM_STEPS),
            batch_size=versioned(_BATCH_SIZE),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=[
                    "moe",
                    "moe_may_d2560_10T",
                    "marin_big_run",
                    f"d{_HIDDEN_DIM}",
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
            checkpoint_keep_every=_CHECKPOINT_KEEP_EVERY,
            expert_parallel=_EXPERT_PARALLEL,
        ),
    )


if __name__ == "__main__":
    executor_main(
        steps=[_build_step()],
        description=(
            f"June 10T d={_HIDDEN_DIM} MoE: bs={_BATCH_SIZE}, steps={_NUM_STEPS:,}, "
            f"tokens={_NUM_STEPS * _BATCH_SIZE * _SEQ_LEN:.3e}, TPU={_TPU}, "
            f"EP={_EXPERT_PARALLEL}, 1pct-noclip schedule, refit LR heuristic (#5951), "
            f"permanent checkpoint every {_CHECKPOINT_KEEP_EVERY} steps."
        ),
    )
