# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate 1 for learned norm-preserving residual mixing on the July Baseline.

The model, optimizer, data, budgets, seed, and evaluation schedule match the
canonical ``july_baseline`` commit ``52d8a9eb8``. The only model change is one
learned positive residual-mixing parameter per layer, shared by attention and
MoE merges.

- Attention: 4:1 GQA (num_kv_heads = NH // 4 via the heuristic), half-RoPE on short layers.
- Long (every-4th + last) layers: **no PKO**, **no RoPE** (un-rotated Q/K, full causal).
- Gate-1 cells at seq_len=8192: d512 (3.82e17) and d768 (2.81e18).

Submit on us-east5-a, interactive priority, v5p-8::

    .venv/bin/iris --cluster=marin job run --no-wait --zone us-east5-a --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe_norm_preserving_residual.launch_norm_preserving_residual
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe_norm_preserving_residual.heuristic import MoeHeuristic
from experiments.grug.moe_norm_preserving_residual.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe_norm_preserving_residual.train import GrugEvalConfig, GrugTrainerConfig

_SEQ_LEN: int = 8192
_TPU: str = "v5p-8"
_GROUP_NAME: str = "MOE-NPR-gate1-issue-8860"

# May-Recipe compute-optimal cells (seq=8192; bs halved from the seq=4096 cells so
# tokens_per_batch, tokens, steps, and muonh_lr are unchanged).
_POINTS: tuple[tuple[str, int, int, int], ...] = (
    ("MOE-NPR-001", 512, 16, 10_980),
    ("MOE-NPR-002", 768, 32, 16_875),
)


def _build_step(experiment_id: str, hidden_dim: int, batch_size: int, num_steps: int) -> ExecutorStep:
    h = MoeHeuristic()
    model = dataclasses.replace(
        h.build_model_config(hidden_dim, seq_len=_SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
    )
    tokens = float(num_steps * batch_size * _SEQ_LEN)
    optimizer = h.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)

    run_id = f"{experiment_id}-d{hidden_dim}"
    step_name = f"grug/{run_id}"

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
                project="dial_moe",
                tags=[
                    "moe",
                    "moe_may_compute_opt",
                    "july_baseline",
                    "norm_preserving_residual",
                    "issue_8860",
                    "gqa",
                    "no_pko",
                    "no_long_rope",
                    f"d{hidden_dim}",
                ],
                group=_GROUP_NAME,
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1)),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=256,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


if __name__ == "__main__":
    steps = [
        _build_step(experiment_id, hidden_dim, batch_size, num_steps)
        for experiment_id, hidden_dim, batch_size, num_steps in _POINTS
    ]
    executor_main(
        steps=steps,
        description=("July Baseline Gate 1 with learned norm-preserving residual mixing. " f"{_POINTS=}, TPU={_TPU}."),
    )
