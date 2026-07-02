# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""July Baseline: May-Recipe MoE, 4:1 GQA, no long-window RoPE, no PKO. d512/d768/d1024.

A refreshed baseline off ``moe_may_pr``: standard GQA attention (not MLA), with the two
"long layer" tricks removed. The moe_may_pr model already defaults ``disable_pko=True`` and
``disable_long_rope=True``; this launcher sets them explicitly and runs the May-Recipe
compute-optimal cells (MoeHeuristic, MuonH, 256 experts, gqa_ratio=4).

- Attention: 4:1 GQA (num_kv_heads = NH // 4 via the heuristic), half-RoPE on short layers.
- Long (every-4th + last) layers: **no PKO**, **no RoPE** (un-rotated Q/K, full causal).
- Compute-optimal cells at seq_len=8192 (bs halved vs the seq=4096 cells so tokens/lr match):
  d512 (3.82e17), d768 (2.81e18), d1024 (1.16e19).

Submit on us-east5-a, interactive priority, v5p-8::

    .venv/bin/iris --cluster=marin job run --no-wait --zone us-east5-a --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_july_baseline
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic import MoeHeuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_SEQ_LEN: int = 8192
_TPU: str = "v5p-8"
_GROUP_NAME: str = "moe-may-july-baseline"

# May-Recipe compute-optimal cells (seq=8192; bs halved from the seq=4096 cells so
# tokens_per_batch, tokens, steps, and muonh_lr are unchanged).
_POINTS: tuple[tuple[int, int, int], ...] = (
    (512, 16, 10_980),
    (768, 32, 16_875),
    (1024, 64, 16_080),
)


def _build_step(hidden_dim: int, batch_size: int, num_steps: int) -> ExecutorStep:
    h = MoeHeuristic()
    model = dataclasses.replace(
        h.build_model_config(hidden_dim, seq_len=_SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
    )
    tokens = float(num_steps * batch_size * _SEQ_LEN)
    optimizer = h.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)

    run_id = f"moe_may_july_baseline_d{hidden_dim}"
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
    steps = [_build_step(d, bs, n) for (d, bs, n) in _POINTS]
    executor_main(
        steps=steps,
        description=(
            f"July Baseline: May-Recipe MoE, 4:1 GQA, no long-window RoPE, no PKO. " f"{_POINTS=}, TPU={_TPU}."
        ),
    )
