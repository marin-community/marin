# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""May Recipe compute-optimal at d=512 / d=768: GQA baseline for the MLA ablation.

Sibling of ``moe_may_compute_opt_mla`` (on the ``mla_simplified`` branch).
This branch (``gqa_simplified``) leaves the attention mechanism as plain GQA
(``MoeHeuristicV1.gqa_ratio = 4``, which yields ``num_kv_heads = 1`` at
d=512 / d=768) but applies the same simplifications as the MLA variant:

  - Removed: attention gate, XSA (exclusive self-attention v-parallel subtract),
    PKO (partial key offset / doc-start zero).
  - Kept: sliding window + short/long mask split (every 4th + last layer runs
    full-causal; other layers use the sliding window), half-RoPE, GatedNorm.

So this is the **same simplified architecture** as the MLA branch except the
attention block is GQA-4:1 instead of MLA. A direct apples-to-apples
comparison of MLA vs GQA on the May Recipe gate-1 setting.

Compute budgets / batch sizes / steps / heuristic identical to baseline and
to the MLA sibling. Submit on us-east5-a, interactive priority, v5p-8::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_gqa
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
_GROUP_NAME: str = "moe-may-compute-opt-gqa"
_WARMUP_FRACTION: float = 0.01

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

    run_id = f"moe_may_compute_opt_gqa_d{hidden_dim}"
    step_name = f"grug/moe_may_compute_opt_gqa/{run_id}"

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
                    "gqa",
                    "no_attn_gate",
                    "no_xsa",
                    "no_pko",
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
            "May Recipe compute-optimal at d=512/768 with simplified GQA attention (no attn_gate, "
            "no XSA, no PKO; sliding window + short/long split kept). Companion to the MLA branch "
            f"for direct MLA-vs-GQA comparison on the gate-1 setup. {_POINTS=}, TPU={_TPU}."
        ),
    )
