# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V2 / DeepSeek-style MLA hd64 dc256 at d=512 + d=768 with qk_mult dropped (=1.0).

Same MLA architecture and heuristic as
``moe_may_compute_opt_mla_deepseek_hd64_dc256_v2(_d768)`` but
``qk_mult = 1.0`` instead of the grug-default 1.3. This removes the inherited
attention-temperature multiplier so the effective Q-K scaling is the standard
``1 / sqrt(q_head_dim)`` only (what DeepSeek-V2 / V3 actually use).

The two paired V2 runs with qk_mult=1.3 sit in the same wandb group ``...v2``;
this no-qk-mult sweep lives in its own group so the A/B is clean per cell.

Submit on us-east5-a, interactive priority, v5p-8::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_mla_deepseek_hd64_dc256_v2_no_qk_mult
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
_GROUP_NAME: str = "moe-may-compute-opt-mla-deepseek-hd64-dc256-v2-no-qk-mult"
_KV_COMPRESSION_DIM: int = 256
_QK_ROPE_HEAD_DIM: int = 64
_HEAD_DIM: int = 64
_HIDDEN_HEAD_RATIO: int = 64
_QK_MULT: float = 1.0  # default is 1.3; here we drop it.

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
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        kv_compression_dim=_KV_COMPRESSION_DIM,
        mla_norm_compressed=True,
        qk_mult=_QK_MULT,
    )
    tokens = float(num_steps * batch_size * _SEQ_LEN)
    optimizer = h.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)

    run_id = f"moe_may_compute_opt_mla_deepseek_hd{_HEAD_DIM}_dc{_KV_COMPRESSION_DIM}_v2_no_qk_mult_d{hidden_dim}"
    step_name = f"grug/moe_may_compute_opt_mla_deepseek_hd{_HEAD_DIM}_dc{_KV_COMPRESSION_DIM}_v2_no_qk_mult/{run_id}"

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
            f"DeepSeek-style MLA at d=512+768 with qk_mult=1.0 (no grug-default 1.3 boost). "
            f"V2 heuristic. {_POINTS=}, TPU={_TPU}."
        ),
    )
