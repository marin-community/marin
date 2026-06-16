# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""10x-overtrain d=512 (TPP ≈ 700) on v4-32 us-central2.

Same d=512 model as the compute-optimal d=512 run (``moe_may_compute_opt`` /
``moe_may_compute_opt_small_v4``), but with 10x more tokens (= 10x compute) and
batch size 4x'd from 32 to 128. Checkpoint every 3,000 steps.

Submit on us-central2, production priority, v4-32:

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --region us-central2 \\
      --priority production \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_d512_10x
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
_TPU: str = "v4-32"
_GROUP_NAME: str = "moe-may-compute-opt"
_WARMUP_FRACTION: float = 0.01

_HIDDEN_DIM: int = 512
# 10x compute-optimal tokens (= 10 × 10980 × 32 × 4096 = 1.44e10) with bs 4x'd
# from the compute-optimal bs=32 to bs=128, so steps = 109800 / 4 = 27450.
_BATCH_SIZE: int = 128
_NUM_STEPS: int = 27450


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

    run_id = f"marin-big-run-moe_may_compute_opt_d{_HIDDEN_DIM}_10x"
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
            steps=versioned(_NUM_STEPS),
            batch_size=versioned(_BATCH_SIZE),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=[
                    "moe",
                    "moe_may_compute_opt",
                    f"d{_HIDDEN_DIM}",
                    "10x_overtrain",
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
            checkpoint_keep_every=3000,
            expert_parallel=2,
        ),
    )


if __name__ == "__main__":
    executor_main(
        steps=[_build_step()],
        description=(
            f"10x-overtrain d={_HIDDEN_DIM}: bs={_BATCH_SIZE}, steps={_NUM_STEPS}, "
            f"tokens={_NUM_STEPS * _BATCH_SIZE * _SEQ_LEN:.3e}. TPU={_TPU}, "
            "1pct-noclip schedule, EP=2, permanent checkpoint every 3k steps."
        ),
    )
