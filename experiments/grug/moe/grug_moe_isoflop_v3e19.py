# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isoflop sweep at 3e19 FLOPs across d=512..1792 on v5p-32.

6 runs. The smaller four sizes use cut batch size + boosted step count to
keep them fittable on v5p-32 while preserving total tokens:

- d=512: bs=512 (heuristic 2048 / 4), steps=53956 (heuristic 13489 * 4)
- d=768: bs=256 (heuristic 1024 / 4), steps=45124 (heuristic 11281 * 4)
- d=1024: bs=256 (heuristic 512 / 2), steps=20882 (heuristic 10441 * 2)
- d=1280: bs=128 (heuristic 256 / 2), steps=24824 (heuristic 12412 * 2)
- d=1536: heuristic-derived (bs=128, steps=14834)
- d=1792: heuristic-derived (bs=128, steps=10294)

Submit on us-east5-a, interactive priority, v5p-32:

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.grug_moe_isoflop_v3e19
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic_v1 import MoeHeuristicV1, build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_BUDGET: float = 3e19
_SEQ_LEN: int = 4096
_TPU: str = "v5p-32"
_RUN_SUFFIX: str = "v1"
_GROUP_NAME: str = "grug-moe-isoflop-v3e19"
_WARMUP_FRACTION: float = 0.01
_HIDDEN_DIMS: tuple[int, ...] = (512, 768, 1024, 1280, 1536, 1792)

# Manual (bs, steps) for sizes where the heuristic-derived bs is too big for
# v5p-32. Each entry preserves the heuristic's total tokens (= 3e19 budget).
_OVERRIDES: dict[int, tuple[int, int]] = {
    512: (512, 53956),
    768: (256, 45124),
    1024: (256, 20882),
    1280: (128, 24824),
    # d=1792 at bs=128 OOMs on v5p-32; halve bs and double steps to fit.
    1792: (64, 20588),
}


def _build_step(hidden_dim: int) -> ExecutorStep:
    if hidden_dim in _OVERRIDES:
        h = MoeHeuristicV1()
        model = h.build_model_config(hidden_dim, seq_len=_SEQ_LEN)
        batch_size, num_steps = _OVERRIDES[hidden_dim]
        tokens = float(num_steps * batch_size * _SEQ_LEN)
        base_optimizer = h.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)
    else:
        model, base_optimizer, batch_size, num_steps = build_from_heuristic(budget=_BUDGET, hidden_dim=hidden_dim)

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

    run_id = f"grug-moe-isoflop-v3e19-d{hidden_dim}-{_RUN_SUFFIX}"
    step_name = f"grug/grug_moe_isoflop_v3e19/{run_id}"

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
                    "grug_moe_isoflop_v3e19",
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
    steps = [_build_step(d) for d in _HIDDEN_DIMS]
    executor_main(
        steps=steps,
        description=(
            f"Isoflop sweep at {_BUDGET:.1e} FLOPs across d={_HIDDEN_DIMS} on "
            f"{_TPU}. d in {sorted(_OVERRIDES.keys())} use cut bs + boosted "
            "steps to fit; others use heuristic-derived (bs, steps). Refit LR "
            "heuristic (#5951), no permanent step-interval checkpoints, "
            "1pct-noclip schedule (warmup=1%, max_grad_norm=None)."
        ),
    )
