# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""May Recipe compute-optimal baseline at d=512 and d=768, with bf16 Newton-Schulz.

This launcher is the bf16-NS variant of the May Recipe compute-optimal
baseline. Architecturally identical to ``moe_may_compute_opt`` (same model,
same heuristic, same data, same TPB / steps); the only change is in
``lib/levanter/src/levanter/optim/grugmuon.py`` where the Newton-Schulz
iteration body is cast to ``bfloat16`` (modded-nanogpt pattern). The momentum
buffer stays in fp32 — only the matmul precision inside NS is reduced.

Compute budgets derived from the drop-1e18 isoflop fit (issue #6074):

    opt_d(C) = 0.137 · C^0.2033   ->   C(d) = (d / 0.137)^(1 / 0.2033)

For d=512  -> C ≈ 3.82e17, tokens ≈ 1.44e9
For d=768  -> C ≈ 2.81e18, tokens ≈ 4.42e9

Batch sizes hardcoded to the README pattern (d=512 -> 32, d=768 -> 64).
LR / beta2 / epsilon come from the refit ``MoeHeuristicV1`` (issue #5951).

Submit on us-east5-a, interactive priority, v5p-8 (per agent.md)::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_bf16_ns
"""


from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_v1 import MoeHeuristicV1
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_SEQ_LEN: int = 4096
_TPU: str = "v5p-8"
_GROUP_NAME: str = "moe-may-compute-opt-bf16-ns"
_WARMUP_FRACTION: float = 0.01

# (hidden_dim, batch_size, num_steps) — same compute-optimal points as the
# baseline ``moe_may_compute_opt`` so paired comparisons are 1:1.
_POINTS: tuple[tuple[int, int, int], ...] = (
    (512, 32, 10980),
    (768, 64, 16875),
)


def _build_step(hidden_dim: int, batch_size: int, num_steps: int) -> ExecutorStep:
    h = MoeHeuristicV1()
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

    run_id = f"moe_may_compute_opt_bf16_ns_d{hidden_dim}"
    step_name = f"grug/moe_may_compute_opt_bf16_ns/{run_id}"

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
                    "bf16_ns",
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
        ),
    )


if __name__ == "__main__":
    steps = [_build_step(d, bs, n) for (d, bs, n) in _POINTS]
    executor_main(
        steps=steps,
        description=(
            "May Recipe compute-optimal at d=512/768 with bf16 Newton-Schulz (modded-nanogpt "
            "pattern): NS iteration body in bf16, momentum buffer fp32. Same drop-1e18 "
            f"isoflop fit (#6074) + refit LR heuristic (#5951) as baseline; d/bs/steps in {_POINTS}, "
            f"TPU={_TPU}, 1pct-noclip schedule."
        ),
    )
