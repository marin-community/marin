# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""May Recipe LR sweep at d1280 — companion to the d512/d768/d1024 sweep.

**Resubmit subset**: only Rin{120, 240} x LRin{0.4, 0.7, 1.0, 1.3, 1.6}.
Iris job ``/larry/iris-run-job-20260521-185623`` covered the full 6x5
grid; the R=4/10/20/60 cells (20 runs) finished cleanly on WandB. The
R=120 and R=240 cells (10 runs) all show ``state=crashed`` after the
coordinator was killed. This script re-runs just those 10. With
``enable_cross_region_ckpt_read=True``, surviving temp checkpoints (if
any) will be picked up; otherwise the cells restart from step 0.

Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/config/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority batch \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_recipe_lr_d1280_sweep
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic_v1 import SEQ_LEN, moe_adamh_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeMuonHMayArchGNMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_NUM_EXPERTS: int = 256
_K: int = 4
_ROUTING_RENORM_SUM: float = 2.5
_EMBED_ADAM_LR_SCALE: float = 1.0
_WARMUP_FRACTION: float = 0.01

_GATED_NORM_RANK: int = 128  # mirrors model._GATED_NORM_RANK

_RUN_SUFFIX: str = "v1"
_GROUP_NAME: str = "muonh-may-recipe-lr-d1280-sweep"

_HIDDEN_DIM: int = 1280
_BATCH_SIZE: int = 256
_TPU: str = "v5p-32"

# Resubmit subset: only the cells that crashed in the original 6x5 grid.
_TOKEN_RATIOS: tuple[int, ...] = (120, 240)
_LR_MULTIPLIERS: tuple[float, ...] = (0.4, 0.7, 1.0, 1.3, 1.6)

# Hardcoded hashes from the original sweep so the resubmit lands on the
# existing output dir (whose temp checkpoints get picked up). Editing this
# file shifts the executor's auto-hash; pinning ``override_output_path``
# per cell keeps the resume target stable.
_ORIGINAL_HASHES: dict[tuple[int, str], str] = {
    (120, "0p4"): "ffec1c",
    (120, "0p7"): "1c1cde",
    (120, "1p0"): "87fe75",
    (120, "1p3"): "1b5dd5",
    (120, "1p6"): "61d352",
    (240, "0p4"): "6802f6",
    (240, "0p7"): "331fc9",
    (240, "1p0"): "a0be5c",
    (240, "1p3"): "aeb7bd",
    (240, "1p6"): "08d23b",
}
_OVERRIDE_BUCKET: str = "gs://marin-us-east5"


def _active_params(cfg: GrugModelConfig, k_active: int) -> int:
    """Count active params per token: only ``k_active`` routed experts."""
    d = cfg.hidden_dim
    nh = cfg.num_heads
    nkv = cfg.num_kv_heads
    head = d // nh
    inter = cfg.intermediate_dim
    s_inter = cfg.shared_expert_intermediate_dim
    e = cfg.num_experts
    nl = cfg.num_layers

    attn = d * nh * head + 2 * d * nkv * head + nh * head * d
    gated_norm_per = 2 * d * _GATED_NORM_RANK
    block_norms = 2 * d + 2 * gated_norm_per
    moe_router = d * e + e
    moe_routed_active = k_active * 3 * d * inter
    shared = 3 * d * s_inter

    per_block = attn + block_norms + moe_router + moe_routed_active + shared
    top_level = 2 * d + 2 * gated_norm_per
    return top_level + nl * per_block


def _build_no_arch_model(hidden_dim: int) -> GrugModelConfig:
    base = moe_adamh_heuristic.build_model_config(hidden_size=hidden_dim, seq_len=SEQ_LEN)
    return dataclasses.replace(
        base,
        num_experts=_NUM_EXPERTS,
        num_experts_per_token=_K,
        partial_key_offset="every_4th",
        use_partial_rope=True,
        last_layer_pko=True,
        router_z_loss_coef=0.0,
        routing_renorm_sum=_ROUTING_RENORM_SUM,
        split_w_gate_up=True,
        pko_norm_order="pko_first_bos_zero",
    )


def _format_lr_mult(mult: float) -> str:
    return f"{mult:.1f}".replace(".", "p")


def _build_step(hidden_dim: int, token_ratio: int, lr_mult: float) -> ExecutorStep:
    model = _build_no_arch_model(hidden_dim)
    p_active = _active_params(model, k_active=_K)
    tokens = float(token_ratio) * p_active
    num_steps = max(1, round(tokens / (_BATCH_SIZE * SEQ_LEN)))

    base_optimizer = moe_adamh_heuristic.build_optimizer_config(
        batch_size=_BATCH_SIZE, tokens=tokens, hidden_dim=hidden_dim, seq_len=SEQ_LEN
    )

    scaled_lr = base_optimizer.learning_rate * lr_mult
    scaled_adam_lr = base_optimizer.adam_lr * lr_mult

    optimizer = GrugMoeMuonHMayArchGNMuonHConfig(
        learning_rate=scaled_lr,
        adam_lr=scaled_adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=_WARMUP_FRACTION,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
        embed_adam_lr_scale=_EMBED_ADAM_LR_SCALE,
    )

    run_id = f"muonh-may-recipe-lr-{_RUN_SUFFIX}-d{hidden_dim}-R{token_ratio}-lr{_format_lr_mult(lr_mult)}"
    step_name = f"grug/muonh_may_recipe_lr_d1280_sweep/{run_id}"
    lr_tag = _format_lr_mult(lr_mult)
    pinned_hash = _ORIGINAL_HASHES[(token_ratio, lr_tag)]
    override_output_path = f"{_OVERRIDE_BUCKET}/{step_name}-{pinned_hash}"

    return ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        override_output_path=override_output_path,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(_TPU)),
            steps=versioned(num_steps),
            batch_size=versioned(_BATCH_SIZE),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=[
                    "moe",
                    "muonh_may_recipe_lr_d1280_sweep",
                    f"d{hidden_dim}",
                    f"R{token_ratio}",
                    f"lr{_format_lr_mult(lr_mult)}",
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
            enable_cross_region_ckpt_read=True,
            checkpoint_keep_every=None,
        ),
    )


if __name__ == "__main__":
    steps = [
        _build_step(hidden_dim=_HIDDEN_DIM, token_ratio=r, lr_mult=m) for r in _TOKEN_RATIOS for m in _LR_MULTIPLIERS
    ]
    executor_main(
        steps=steps,
        description=(
            f"May Recipe LR d1280 resubmit: R in {_TOKEN_RATIOS} (failed cells from "
            f"job /larry/iris-run-job-20260521-185623), LR multipliers in {_LR_MULTIPLIERS}, "
            f"B={_BATCH_SIZE}, TPU={_TPU}."
        ),
    )
