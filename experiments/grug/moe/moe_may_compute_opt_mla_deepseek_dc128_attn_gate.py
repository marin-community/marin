# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-style MLA at d=512, d_c=128, with attention gate enabled.

Sibling of ``moe_may_compute_opt_mla_deepseek_dc128``. Same architecture
(additive rope, NH=8 at d=512, DeepSeek-strict norm, no Q compression,
``kv_compression_dim = 128``) but with ``attn_gate = True`` -- a per-head
sigmoid gate applied to the attention output (before w_o):

    gate[t, h] = 2 * sigmoid(x[t] @ w_attn_gate[:, h])    # (D, NH) weights
    attn_out  *= gate[..., None]                          # scalar per (token, head)

Zero-init means gate = 1.0 at step 0 (identical to the no-gate run), so any
difference comes from learning the gate over the course of training.
Absorption-friendly: the gate depends only on the current query, so at decode
it adds one (D, NH) matmul and a per-head scalar multiply.

Only d=512 per user instruction.

Submit on us-east5-a, interactive priority, v5p-8::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_mla_deepseek_dc128_attn_gate
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
_GROUP_NAME: str = "moe-may-compute-opt-mla-deepseek-dc128-attn-gate"
_WARMUP_FRACTION: float = 0.01
_KV_COMPRESSION_DIM: int = 128
_QK_ROPE_HEAD_DIM: int = 64
_HEAD_DIM: int = 128
_HIDDEN_HEAD_RATIO: int = 64

_POINTS: tuple[tuple[int, int, int], ...] = ((512, 32, 10980),)


def _build_step(hidden_dim: int, batch_size: int, num_steps: int) -> ExecutorStep:
    h = MoeHeuristicV1()
    base_model = h.build_model_config(hidden_dim, seq_len=_SEQ_LEN)
    num_heads = hidden_dim // _HIDDEN_HEAD_RATIO
    model = dataclasses.replace(
        base_model,
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        kv_compression_dim=_KV_COMPRESSION_DIM,
        mla_norm_compressed=True,
        attn_gate=True,
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

    run_id = f"moe_may_compute_opt_mla_deepseek_dc{_KV_COMPRESSION_DIM}_attn_gate_d{hidden_dim}"
    step_name = f"grug/moe_may_compute_opt_mla_deepseek_dc{_KV_COMPRESSION_DIM}_attn_gate/{run_id}"

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
                    "mla",
                    "mla_deepseek",
                    "additive_rope",
                    "attn_gate",
                    f"d_c{_KV_COMPRESSION_DIM}",
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
            f"DeepSeek-style MLA at d=512 with d_c={_KV_COMPRESSION_DIM} and per-head sigmoid "
            "attention gate (zero-init). Same architecture as compute_opt_mla_deepseek_dc128 "
            "otherwise. Wandb project: dial_moe."
        ),
    )
