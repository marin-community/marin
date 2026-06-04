# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wall-attention-vs-SWA architecture comparison at d512 (May Recipe), identical MuonH recipe.

Same as ``muonh_combined_d512.py`` (the SWA baseline, public run
``muonh-d512-combined-reproduce``, c4 val 3.5931) except BOTH the sliding-window (short)
and full-causal (long) attention layers are replaced with Wall Attention
(tilde-research/wall-attention-release): softmax attention with a data-dependent
per-channel multiplicative decay, a per-head attention sink, and (optionally) a FoX
scalar gate. Short layers keep a sliding window (sliding_window//2); long layers are full
causal. GQA head structure mirrors the baseline so params match (~+0.01%). Wall attention
respects segment_ids (document masking) like the baseline.

Submit (bump WALL_RUN_TAG for a clean rerun after code changes):

    .venv/bin/iris --config lib/iris/config/marin.yaml job run --no-wait \\
      --preemptible --reserve v5p-8 \\
      -e WANDB_API_KEY "$WANDB_API_KEY" -e WALL_USE_SCALAR_GATE 0 \\
      -- python -m experiments.grug.moe.wall_combined_d512
"""

import dataclasses
import os

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import this_output_path, versioned

from experiments.grug.moe.direct_launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeDirectLaunchConfig,
    _resolve_run_id,
    train_grug_moe,
)
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_HIDDEN_DIM: int = 512
_BUDGET: float = 2.19e17
_TARGET_STEPS: int = 2**14
_TPU: str = "v5p-8"
_WALL_CHUNK_SIZE: int = 64
_COMBINED_K: int = 5
_COMBINED_SHARED_INT_DIM_RATIO: int = 2  # shared_int_dim = hidden_dim // RATIO

_USE_SCALAR_GATE: bool = os.environ.get("WALL_USE_SCALAR_GATE", "0") == "1"
_USE_SINK: bool = os.environ.get("WALL_USE_SINK", "1") == "1"
_RUN_TAG: str = os.environ.get("WALL_RUN_TAG", "v1")


def _build_launch() -> tuple[str, GrugMoeDirectLaunchConfig]:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
    )
    model = dataclasses.replace(
        model,
        num_experts_per_token=_COMBINED_K,
        shared_expert_intermediate_dim=_HIDDEN_DIM // _COMBINED_SHARED_INT_DIM_RATIO,
        use_wall_attention=True,
        wall_chunk_size=_WALL_CHUNK_SIZE,
        wall_use_scalar_gate=_USE_SCALAR_GATE,
        wall_use_sink=_USE_SINK,
    )

    optimizer = GrugMoeMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=0.01,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )

    feats = []
    if _USE_SCALAR_GATE:
        feats.append("sg")
    if not _USE_SINK:
        feats.append("nosink")
    feat_tag = ("-" + "-".join(feats)) if feats else ""
    suffix = f"wall{feat_tag}-{_RUN_TAG}"
    run_id = _resolve_run_id(f"wall-d{_HIDDEN_DIM}-{_BUDGET:.2e}-{suffix}".replace("+", ""))
    name = f"grug/wall-d{_HIDDEN_DIM}-{suffix}"

    launch = GrugMoeDirectLaunchConfig(
        model=versioned(model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=ResourceConfig.with_tpu(_TPU),
        steps=versioned(num_steps),
        batch_size=versioned(batch_size),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            entity="marin-community",
            project="marin_moe",
            tags=["moe", "muonh", "may_recipe", "wall_attention", *feats, f"d{_HIDDEN_DIM}"],
            group="wall-d512-combined",
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
                steps_per_eval=200,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
        checkpoint_keep_every=1000,
    )
    return name, launch


if __name__ == "__main__":
    name, launch = _build_launch()
    print(f"Submitting Wall-attention d512 (scalar_gate={_USE_SCALAR_GATE}, sink={_USE_SINK}, tag={_RUN_TAG})")
    job_id = train_grug_moe(name=name, launch=launch, wait=True)
    print(f"Training job finished: {job_id}")
