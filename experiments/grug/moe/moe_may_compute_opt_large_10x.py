# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""10x-overtrain d=1024 and d=1280 on v4-32 us-central2.

Same d=1024 / d=1280 models as the compute-optimal moe_may_compute_opt_large,
but with 10x more tokens (= 10x compute) and batch sizes doubled (so steps go
up by 5x rather than 10x). Checkpoint every 5,000 steps.

For d=1024 -> bs=256 (= 2x compute-opt 128), steps=80,400 (= 5x 16,080), tokens=8.43e10
For d=1280 -> bs=512 (= 2x compute-opt 256), steps=71,625 (= 5x 14,325), tokens=1.50e11

Submit on us-central2, production priority, v4-32:

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --region us-central2 \\
      --priority production \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_large_10x
"""


from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution import ExecutorStep, executor_main, this_output_path, versioned

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

# (hidden_dim, batch_size, num_steps) — 10x compute-optimal tokens with bs doubled.
_POINTS: tuple[tuple[int, int, int], ...] = (
    (1024, 256, 80400),
    (1280, 512, 71625),
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

    run_id = f"marin-big-run-moe_may_compute_opt_d{hidden_dim}_10x"
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
            checkpoint_keep_every=5000,
            expert_parallel=2,
        ),
    )


if __name__ == "__main__":
    steps = [_build_step(d, bs, n) for (d, bs, n) in _POINTS]
    executor_main(
        steps=steps,
        description=(
            f"10x-overtrain d=1024/1280 on {_TPU} us-central2 with doubled batch sizes. "
            f"d/bs/steps in {_POINTS}, EP=2, 1pct-noclip schedule, "
            "permanent checkpoint every 5k steps."
        ),
    )
