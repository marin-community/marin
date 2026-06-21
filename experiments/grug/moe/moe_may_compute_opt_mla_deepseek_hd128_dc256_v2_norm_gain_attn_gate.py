# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V2 MLA hd128 dc256 with learnable c_kv norm + attn_gate at d=512.

Builds on ``moe_may_compute_opt_mla_deepseek_hd128_dc256_v2_norm_gain`` and
turns on ``attn_gate=True``: a per-head sigmoid gate
``gate = 2 * sigmoid(x @ w_attn_gate)`` is applied (scalar per (token, head))
to the attention output before the output projection. Zero-init means
``gate=1.0`` at step 0, identical to no-gate model.

Companion to the no-PKO ``_norm_gain`` baseline and the
``_norm_gain_xsa`` and ``_norm_gain_pko_last`` sibling runs.
d=512 only.

Submit on us-east5-a, interactive priority, v5p-8::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_mla_deepseek_hd128_dc256_v2_norm_gain_attn_gate
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_v2 import MoeHeuristicV2
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_SEQ_LEN: int = 4096
_TPU: str = "v5p-8"
_GROUP_NAME: str = "moe-may-compute-opt-mla-deepseek-hd128-dc256-v2-norm-gain-attn-gate"
_KV_COMPRESSION_DIM: int = 256
_QK_ROPE_HEAD_DIM: int = 64
_HEAD_DIM: int = 128
_HIDDEN_HEAD_RATIO: int = 64

_POINTS: tuple[tuple[int, int, int], ...] = ((512, 32, 10980),)


def _build_step(hidden_dim: int, batch_size: int, num_steps: int) -> ExecutorStep:
    h = MoeHeuristicV2()
    base_model = h.build_model_config(hidden_dim, seq_len=_SEQ_LEN)
    num_heads = hidden_dim // _HIDDEN_HEAD_RATIO
    model = dataclasses.replace(
        base_model,
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        kv_compression_dim=_KV_COMPRESSION_DIM,
        mla_norm_compressed=True,
        mla_norm_compressed_learnable=True,
        attn_gate=True,
    )
    tokens = float(num_steps * batch_size * _SEQ_LEN)
    optimizer = h.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)

    run_id = (
        f"moe_may_compute_opt_mla_deepseek_hd{_HEAD_DIM}_dc{_KV_COMPRESSION_DIM}"
        f"_v2_norm_gain_attn_gate_d{hidden_dim}"
    )
    step_name = (
        f"grug/moe_may_compute_opt_mla_deepseek_hd{_HEAD_DIM}_dc{_KV_COMPRESSION_DIM}"
        f"_v2_norm_gain_attn_gate/{run_id}"
    )

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
                    f"hd{_HEAD_DIM}",
                    f"d_c{_KV_COMPRESSION_DIM}",
                    "heuristic_v2",
                    "norm_gain",
                    "attn_gate",
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
            f"V2 MLA hd{_HEAD_DIM} dc{_KV_COMPRESSION_DIM} at d=512: learnable c_kv RMSNorm + "
            "attn_gate (zero-init). Wandb project: dial_moe."
        ),
    )
