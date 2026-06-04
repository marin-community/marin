# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLA-vs-SWA architecture comparison at d512 (May Recipe), identical MuonH recipe.

Same as ``muonh_combined_d512.py`` (the SWA baseline, public run
``muonh-d512-combined-reproduce``, c4 val 3.5931) except the sliding-window
attention on the "short" layers is replaced with Gated Linear Attention
(arXiv 2312.06635). The periodic full-attention layers (every 4th + last) stay
softmax. GLA uses the same GQA head structure as SWA (``num_heads`` query heads,
``num_kv_heads`` KV heads, ``head_dim``) so the parameter footprint matches; the
value/output width is set by ``GLA_EXPAND_V``:

  * ``GLA_EXPAND_V=0.5`` -> ~0.96x SWA attn params (true param-match)
  * ``GLA_EXPAND_V=1.0`` -> ~1.41x SWA attn params (full-capacity GLA; +0.15% model)

GLA respects ``segment_ids`` (document-boundary state resets), matching the SWA
layers' cross-document masking.

Submit (one per expand_v; bump GLA_RUN_TAG for a clean rerun after code changes):

    GLA_EXPAND_V=0.5 .venv/bin/iris --config lib/iris/config/marin.yaml job run --no-wait \\
      --preemptible --reserve v5p-8 \\
      -e WANDB_API_KEY "$WANDB_API_KEY" -e GLA_EXPAND_V -e GLA_RUN_TAG \\
      -- python -m experiments.grug.moe.gla_combined_d512
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
_GLA_CHUNK_SIZE: int = 64
_COMBINED_K: int = 5
_COMBINED_SHARED_INT_DIM_RATIO: int = 2  # shared_int_dim = hidden_dim // RATIO

_EXPAND_V: float = float(os.environ.get("GLA_EXPAND_V", "1.0"))
_USE_SHORT_CONV: bool = os.environ.get("GLA_USE_SHORT_CONV", "0") == "1"
_USE_XSA: bool = os.environ.get("GLA_USE_XSA", "0") == "1"
_RUN_TAG: str = os.environ.get("GLA_RUN_TAG", "v1")


def _build_launch() -> tuple[str, GrugMoeDirectLaunchConfig]:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
    )
    # Combined-arm overrides + GLA on the short layers (matches SWA recipe otherwise).
    model = dataclasses.replace(
        model,
        num_experts_per_token=_COMBINED_K,
        shared_expert_intermediate_dim=_HIDDEN_DIM // _COMBINED_SHARED_INT_DIM_RATIO,
        use_gla=True,
        gla_chunk_size=_GLA_CHUNK_SIZE,
        gla_expand_v=_EXPAND_V,
        gla_use_short_conv=_USE_SHORT_CONV,
        gla_use_xsa=_USE_XSA,
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

    ev_tag = f"ev{_EXPAND_V:g}".replace(".", "p")
    conv_tag = "-sconv" if _USE_SHORT_CONV else ""
    xsa_tag = "-xsa" if _USE_XSA else ""
    suffix = f"gla-{ev_tag}{conv_tag}{xsa_tag}-{_RUN_TAG}"
    run_id = _resolve_run_id(f"gla-d{_HIDDEN_DIM}-{_BUDGET:.2e}-{suffix}".replace("+", ""))
    name = f"grug/gla-d{_HIDDEN_DIM}-{suffix}"

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
            tags=[
                "moe",
                "muonh",
                "gla",
                ev_tag,
                *(["short_conv"] if _USE_SHORT_CONV else []),
                *(["xsa"] if _USE_XSA else []),
            ],
            group="gla-d512-combined",
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
    print(f"Submitting GLA d512 comparison (expand_v={_EXPAND_V}, tag={_RUN_TAG})")
    job_id = train_grug_moe(name=name, launch=launch, wait=True)
    print(f"Training job finished: {job_id}")
