# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V2 MLA with legacy half-RoPE at d=512+768, head_dim=128, d_c=256.

Sibling of ``moe_may_compute_opt_mla_deepseek_hd128_dc256_v2`` (additive
rope: per-head Q/K=192) — this one uses the **legacy half-RoPE** path
(``qk_rope_head_dim=None``) so the rope channels share the head_dim budget
instead of being added on top:

  - per-head V = head_dim = 128
  - per-head Q/K total = head_dim = 128
      - first 64 channels: rope-rotated (and on K side, shared via k_r across heads)
      - last 64 channels:  no-rope (on K side: per-head up-projected from c_kv)
  - shared k_r width = head_dim // 2 = 64
  - per-token cache = d_c + half = 256 + 64 = **320** (identical to the additive
    sibling, so this is a fixed-cache A/B on per-head Q/K capacity)

V2 / MuonH heuristic (256 experts, May Recipe LR), NH=D//64 (8 at d=512,
12 at d=768), ``mla_norm_compressed=True`` so RMSNorm is on c_kv (DeepSeek
recipe) instead of on the post-up-projection k_nr.

Submit on us-east5-a, interactive priority, v5p-8::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_mla_half_rope_hd128_dc256_v2
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
_GROUP_NAME: str = "moe-may-compute-opt-mla-half-rope-hd128-dc256-v2"
_KV_COMPRESSION_DIM: int = 256
_HEAD_DIM: int = 128
_HIDDEN_HEAD_RATIO: int = 64  # NH = D // 64.

_POINTS: tuple[tuple[int, int, int], ...] = (
    (512, 32, 10980),
    (768, 64, 16875),
)


def _build_step(hidden_dim: int, batch_size: int, num_steps: int) -> ExecutorStep:
    h = MoeHeuristicV2()
    base_model = h.build_model_config(hidden_dim, seq_len=_SEQ_LEN)
    num_heads = hidden_dim // _HIDDEN_HEAD_RATIO
    # Legacy half-RoPE: leave ``qk_rope_head_dim`` as None. Rope is applied to
    # the first head_dim // 2 channels of Q and to the shared k_r (also half wide).
    model = dataclasses.replace(
        base_model,
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        kv_compression_dim=_KV_COMPRESSION_DIM,
        mla_norm_compressed=True,
    )
    tokens = float(num_steps * batch_size * _SEQ_LEN)
    optimizer = h.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)

    run_id = f"moe_may_compute_opt_mla_half_rope_hd{_HEAD_DIM}_dc{_KV_COMPRESSION_DIM}_v2_d{hidden_dim}"
    step_name = f"grug/moe_may_compute_opt_mla_half_rope_hd{_HEAD_DIM}_dc{_KV_COMPRESSION_DIM}_v2/{run_id}"

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
            f"V2 MLA half-RoPE at d=512+768, head_dim={_HEAD_DIM}, d_c={_KV_COMPRESSION_DIM}. "
            "Per-head Q/K=128 (rope/no-rope share head_dim), V=128. Cache=320. "
            f"{_POINTS=}, TPU={_TPU}."
        ),
    )
