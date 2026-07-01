# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LatentMoE (½ compression) + latent RMSNorm + latent GatedNorm, at d512/d768.

Extends ``moe_may_compute_opt_mla_hd192_vd256_qc_latentmoe_half_norm_xsa`` by adding a
GatedNorm on the MoE latent ``c`` (after the learnable RMSNorm, before dispatch), to see
whether gating the compressed latent on top of normalizing it recovers the loss gap the
½-compression LatentMoE showed vs the MLA baseline. Everything else identical: wide-V /
low-rank-Q MLA attention, ``moe_latent_dim=D/2``, full-width (D) experts, router + shared
expert unchanged, 256 experts top-4. d512 and d768 only.

Submit on us-east5-a, interactive priority, v5p-8::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_mla_hd192_vd256_qc_latentmoe_half_norm_gatednorm_xsa
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
_GROUP_NAME: str = "moe-may-compute-opt-mla-hd192-vd256-qc-latentmoe-half-norm-gatednorm-xsa"
_KV_COMPRESSION_DIM: int = 512
_QK_ROPE_HEAD_DIM: int = 64
_HEAD_DIM: int = 192
_V_HEAD_DIM: int = 256
_HIDDEN_HEAD_RATIO: int = 128

# Two smaller compute-optimal cells from the README V2 baseline.
_POINTS: tuple[tuple[int, int, int], ...] = (
    (512, 32, 10980),
    (768, 64, 16875),
)


def _build_step(hidden_dim: int, batch_size: int, num_steps: int) -> ExecutorStep:
    h = MoeHeuristicV2()
    base_model = h.build_model_config(hidden_dim, seq_len=_SEQ_LEN)
    num_heads = hidden_dim // _HIDDEN_HEAD_RATIO
    model = dataclasses.replace(
        base_model,
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        kv_compression_dim=_KV_COMPRESSION_DIM,
        q_compression_dim=hidden_dim // 2,
        mla_norm_compressed=True,
        mla_norm_compressed_learnable=True,
        q_norm_learnable=True,
        xsa=True,
        # LatentMoE: routed experts in D/2, full-width (D) intermediate, latent RMSNorm + GatedNorm.
        intermediate_dim=hidden_dim,
        moe_latent_dim=hidden_dim // 2,
        moe_latent_norm=True,
        moe_latent_gated_norm=True,
    )
    tokens = float(num_steps * batch_size * _SEQ_LEN)
    optimizer = h.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)

    _stem = f"moe_may_compute_opt_mla_hd{_HEAD_DIM}_vd{_V_HEAD_DIM}_qc_latentmoe_half_norm_gatednorm_xsa"
    run_id = f"{_stem}_d{hidden_dim}"
    step_name = f"grug/{_stem}/{run_id}"

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
                    f"vd{_V_HEAD_DIM}",
                    "q_compression",
                    f"d_c{_KV_COMPRESSION_DIM}",
                    "heuristic_v2",
                    "norm_gain",
                    "q_norm_gain",
                    "xsa",
                    "latentmoe",
                    "latentmoe_half",
                    "latentmoe_norm",
                    "latentmoe_gatednorm",
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
            f"LatentMoE ½-compression + latent RMSNorm + latent GatedNorm on the MLA "
            f"hd{_HEAD_DIM}/vd{_V_HEAD_DIM} q_c/XSA stack. {_POINTS=}, TPU={_TPU}."
        ),
    )
