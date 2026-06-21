# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""May Recipe compute-optimal at d=512: GQA + XSA only (drop PKO, attn_gate, sw).

Architectural minimum on top of the May Recipe gate-1 baseline:

  - Drop: PKO, attn_gate, sliding window (so no short/long layer split).
  - Keep: GQA (4:1 -> KV=1 at d=512), XSA (exclusive self-attention), half-RoPE,
    GatedNorm, MoE.

So this run differs from the (hypothetical) full GQA-only baseline only by
having XSA on. Pairs with ``moe_may_compute_opt_no_pko_no_attn_gate.py``
(same things dropped plus XSA also kept and sliding window kept) and with
the ``gqa_simplified`` branch's launcher (drops XSA too but keeps sliding
window).

Submit on us-east5-a, interactive priority, v5p-8 (per agent.md)::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_gqa_xsa
"""

import dataclasses

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
_GROUP_NAME: str = "moe-may-compute-opt-gqa-xsa"
_WARMUP_FRACTION: float = 0.01

_POINTS: tuple[tuple[int, int, int], ...] = ((512, 32, 10980),)


def _build_step(hidden_dim: int, batch_size: int, num_steps: int) -> ExecutorStep:
    h = MoeHeuristicV1()
    base_model = h.build_model_config(hidden_dim, seq_len=_SEQ_LEN)
    model = dataclasses.replace(
        base_model,
        disable_pko=True,
        disable_attn_gate=True,
        disable_sliding_window=True,
    )
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

    run_id = f"moe_may_compute_opt_gqa_xsa_d{hidden_dim}"
    step_name = f"grug/moe_may_compute_opt_gqa_xsa/{run_id}"

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
                    "gqa",
                    "xsa",
                    "no_pko",
                    "no_attn_gate",
                    "no_sw",
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
            "May Recipe compute-optimal at d=512: GQA + XSA only. Drop PKO, attn_gate, "
            "sliding window. All layers run full causal attention. Pairs with "
            "compute_opt_no_pko_no_attn_gate (keeps sliding window) and gqa_simplified "
            "(drops XSA too). Wandb project: dial_moe."
        ),
    )
