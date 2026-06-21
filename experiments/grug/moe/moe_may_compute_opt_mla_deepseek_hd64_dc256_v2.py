# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-style MLA at d=512 (hd=64, qk_rope=64, d_c=256) on V2 / MuonH heuristic.

V2 sibling of ``moe_may_compute_opt_mla_deepseek_hd64_dc256`` (V1).
Identical MLA architecture knobs (head_dim=64, qk_rope_head_dim=64,
kv_compression_dim=256, num_heads=8, mla_norm_compressed=True), but the
optimizer / model config comes from ``MoeHeuristicV2``:

  - num_experts left at the GrugModelConfig default (**256**) instead of V1's
    hardcoded 64.
  - LR formula uses the May Recipe MuonH ISOFlop refit (issue #5951) rather
    than V1's AdamH-derived coefficients. At d=512 the peak MuonH LR is
    ~0.00980 (V2) instead of ~0.01069 (V1).
  - ``build_optimizer_config`` already returns ``GrugMoeMuonHConfig`` directly
    on V2 (no rewrap needed, unlike V1 which returns AdamH and we rewrap).

The compute-optimal cell (BS=32, 10,980 steps) is the same as V1, matching
the d=512 baseline row in ``experiments/grug/moe/README.md`` (current May
Recipe baseline). Pairs cleanly with that baseline since both use V2 / 256
experts / MuonH LR formula.

Submit on us-east5-a, interactive priority, v5p-8::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_mla_deepseek_hd64_dc256_v2
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
_GROUP_NAME: str = "moe-may-compute-opt-mla-deepseek-hd64-dc256-v2"
_KV_COMPRESSION_DIM: int = 256
_QK_ROPE_HEAD_DIM: int = 64
_HEAD_DIM: int = 64
_HIDDEN_HEAD_RATIO: int = 64  # NH = D // 64 (= 8 at d=512).

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
    )
    tokens = float(num_steps * batch_size * _SEQ_LEN)
    # V2.build_optimizer_config returns GrugMoeMuonHConfig directly (1pct-noclip
    # schedule, May Recipe defaults) -- no rewrap step needed.
    optimizer = h.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)

    run_id = f"moe_may_compute_opt_mla_deepseek_hd{_HEAD_DIM}_dc{_KV_COMPRESSION_DIM}_v2_d{hidden_dim}"
    step_name = f"grug/moe_may_compute_opt_mla_deepseek_hd{_HEAD_DIM}_dc{_KV_COMPRESSION_DIM}_v2/{run_id}"

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
            f"DeepSeek-style MLA at d=512 (head_dim={_HEAD_DIM}, qk_rope={_QK_ROPE_HEAD_DIM}, "
            f"d_c={_KV_COMPRESSION_DIM}, NH=D//64) on V2 / MuonH heuristic (256 experts, May "
            "Recipe LR). Wandb project: dial_moe."
        ),
    )
