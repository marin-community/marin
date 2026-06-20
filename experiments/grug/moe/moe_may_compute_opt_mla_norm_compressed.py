# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""May Recipe compute-optimal at d=512 with DeepSeek-V2 MLA norm placement.

Sibling of ``moe_may_compute_opt_mla``. Same model / heuristic / batch / steps,
but ``GrugModelConfig.mla_norm_compressed = True`` so RMSNorm runs on the
compressed KV latent ``c_kv`` (right after ``w_dkv``) instead of on the
post-up-projection ``k_nr``. This matches the DeepSeek-V2 recipe and preserves
the matrix-absorption inference optimization (cache is just c_kv + k_r, with no
per-token ``rms_norm(k_nr)`` reconstruction at decode).

Other RMSNorms unchanged:
  - ``q``: still normalized post-projection (query side, no cache interaction).
  - ``k_r``: still normalized post-projection (rope-bearing K, computed once per
    token and cached after rope; safe).

This run isolates the norm-placement effect on training quality. Compare against
``moe_may_compute_opt_mla`` (legacy norm on ``k_nr``) and the GQA / baseline
gate-1 anchors.

Submit on us-east5-a, interactive priority, v5p-8 (per agent.md)::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.moe_may_compute_opt_mla_norm_compressed
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
_GROUP_NAME: str = "moe-may-compute-opt-mla-norm-compressed"
_WARMUP_FRACTION: float = 0.01

# Only the d=512 cell; user requested kicking off just one for this variant.
_POINTS: tuple[tuple[int, int, int], ...] = ((512, 32, 10980),)


def _build_step(hidden_dim: int, batch_size: int, num_steps: int) -> ExecutorStep:
    h = MoeHeuristicV1()
    base_model = h.build_model_config(hidden_dim, seq_len=_SEQ_LEN)
    model = dataclasses.replace(base_model, mla_norm_compressed=True)
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

    run_id = f"moe_may_compute_opt_mla_norm_compressed_d{hidden_dim}"
    step_name = f"grug/moe_may_compute_opt_mla_norm_compressed/{run_id}"

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
                    "mla",
                    "mla_norm_compressed",
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
            "May Recipe compute-optimal at d=512 with DeepSeek-V2 MLA norm placement: "
            "RMSNorm on c_kv (post-w_dkv), no norm on k_nr. Same heuristic / batch / steps "
            f"as moe_may_compute_opt_mla. {_POINTS=}, TPU={_TPU}."
        ),
    )
