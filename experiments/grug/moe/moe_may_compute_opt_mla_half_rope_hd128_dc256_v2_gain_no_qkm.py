# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V2 MLA half-RoPE hd128 dc256 at d=512 with learnable c_kv norm + no qk_mult.

Builds on ``moe_may_compute_opt_mla_half_rope_hd128_dc256_v2`` (legacy half-RoPE
path: per-head Q/K=128 split 64 rope + 64 no-rope, V=128, d_c=256) and turns on
two flags simultaneously:

  - ``mla_norm_compressed_learnable=True``: c_kv normalization uses the
    parameterized RMSNorm class with a learnable ``(d_c,)=(256,)`` per-channel
    weight (DeepSeek's kv_a_layernorm), instead of the gainless ``rms_norm()``
    function.
  - ``qk_mult=1.0``: drops the grug-default 1.3 temperature boost on Q,
    leaving the standard ``1/sqrt(q_head_dim)`` SDPA scale only (DeepSeek-V2 /
    V3 convention).

Only d=512 per request.

Submit on us-east5-a, interactive priority, v5p-8::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_mla_half_rope_hd128_dc256_v2_gain_no_qkm
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
_GROUP_NAME: str = "moe-may-compute-opt-mla-half-rope-hd128-dc256-v2-gain-no-qkm"
_KV_COMPRESSION_DIM: int = 256
_HEAD_DIM: int = 128
_HIDDEN_HEAD_RATIO: int = 64
_QK_MULT: float = 1.0  # default 1.3; dropped here.

_POINTS: tuple[tuple[int, int, int], ...] = ((512, 32, 10980),)


def _build_step(hidden_dim: int, batch_size: int, num_steps: int) -> ExecutorStep:
    h = MoeHeuristicV2()
    base_model = h.build_model_config(hidden_dim, seq_len=_SEQ_LEN)
    num_heads = hidden_dim // _HIDDEN_HEAD_RATIO
    # Legacy half-RoPE: qk_rope_head_dim left as None.
    model = dataclasses.replace(
        base_model,
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        kv_compression_dim=_KV_COMPRESSION_DIM,
        mla_norm_compressed=True,
        mla_norm_compressed_learnable=True,
        qk_mult=_QK_MULT,
    )
    tokens = float(num_steps * batch_size * _SEQ_LEN)
    optimizer = h.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)

    run_id = f"moe_may_compute_opt_mla_half_rope_hd{_HEAD_DIM}_dc{_KV_COMPRESSION_DIM}_v2_gain_no_qkm_d{hidden_dim}"
    step_name = f"grug/moe_may_compute_opt_mla_half_rope_hd{_HEAD_DIM}_dc{_KV_COMPRESSION_DIM}_v2_gain_no_qkm/{run_id}"

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
                    "mla_half_rope",
                    f"hd{_HEAD_DIM}",
                    f"d_c{_KV_COMPRESSION_DIM}",
                    "heuristic_v2",
                    "norm_gain",
                    "no_qk_mult",
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
            f"V2 MLA half-RoPE hd{_HEAD_DIM} dc{_KV_COMPRESSION_DIM} at d=512 with learnable c_kv "
            "norm + qk_mult=1.0 (DeepSeek-style). Wandb project: dial_moe."
        ),
    )
