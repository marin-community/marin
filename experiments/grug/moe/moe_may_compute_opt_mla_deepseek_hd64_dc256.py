# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-style MLA at d=512 with head_dim=64, qk_rope=64, d_c=256.

Variant of ``moe_may_compute_opt_mla_deepseek`` (which used head_dim=128,
d_c=512). This one shrinks both the per-head dimensions and the latent:

  - head_dim = 64                    -> v_head_dim = qk_nope_head_dim = 64
  - qk_rope_head_dim = 64
  - per-head Q/K = head_dim + qk_rope_head_dim = 128
  - per-head V = head_dim = 64
  - num_heads = 8 (D // 64)
  - kv_compression_dim = 256
  - mla_norm_compressed = True (DeepSeek-strict: norm only on c_kv)

Per-token cache = d_c + qk_rope = 256 + 64 = 320.

Submit on us-east5-a, interactive priority, v5p-8::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_mla_deepseek_hd64_dc256
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
_GROUP_NAME: str = "moe-may-compute-opt-mla-deepseek-hd64-dc256"
_WARMUP_FRACTION: float = 0.01
_KV_COMPRESSION_DIM: int = 256
_QK_ROPE_HEAD_DIM: int = 64
_HEAD_DIM: int = 64
_HIDDEN_HEAD_RATIO: int = 64  # NH = D // 64 (= 8 at d=512).

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

    run_id = f"moe_may_compute_opt_mla_deepseek_hd{_HEAD_DIM}_dc{_KV_COMPRESSION_DIM}_d{hidden_dim}"
    step_name = f"grug/moe_may_compute_opt_mla_deepseek_hd{_HEAD_DIM}_dc{_KV_COMPRESSION_DIM}/{run_id}"

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
            f"DeepSeek-style MLA at d=512 with head_dim={_HEAD_DIM}, qk_rope={_QK_ROPE_HEAD_DIM}, "
            f"d_c={_KV_COMPRESSION_DIM}, NH=D//64. Per-head Q/K=128, V=64. Cache=320. "
            "Wandb project: dial_moe."
        ),
    )
