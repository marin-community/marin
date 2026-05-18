# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + 1pct-noclip + learned q rotating/stationary rescale on PKO layers.

In PKO layers each token has q split into a rotating half (gets RoPE)
and a stationary half (no RoPE; PKO uses this half via the shift). This
adds a learned per-block weight ``q_split_rescale_weight`` of shape
``(hidden_dim, num_kv_heads)`` (zero-init) that produces a scalar per
(token, kv-head):

    s = 2 * sigmoid(x @ w)            # stationary weight in [0, 2]
    r = 2 - s                          # rotating weight in [0, 2]
    q_rot = q[..., :half]  * r
    q_stat = q[..., half:] * s

Sum ``s + r = 2`` always. At init ``s = r = 1.0`` (no-op vs baseline).
Per-kv-head scalar broadcast across the GQA group's q-heads (group_size
= num_q_heads / num_kv_heads).

Two variants, both at d512:

| trial    | rescale applied | effect of rms_norm |
|---|---|---|
| pre-norm  | before ``rms_norm(q)`` | rms_norm normalises but the per-half ratio is preserved |
| post-norm | after  ``rms_norm(q)`` | post-norm half magnitudes are directly scaled |

Branch off the 1pct-noclip baseline; only the PKO-layer q path changes.
``q_split_rescale_weight`` routes to the small-LR ``adam`` group
(alongside ``attn_gate`` / ``router_bias``).

Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_pko_q_split_rescale_sweep
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHMayArchGNMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_BUDGET: float = 2.19e17
_HIDDEN_DIM: int = 512

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-1pct-pko-q-split-rescale-sweep"
_RUN_SUFFIX: str = "v1"


def _muonh_optimizer(base_optimizer: GrugMoeAdamHConfig) -> GrugMoeMuonHMayArchGNMuonHConfig:
    return GrugMoeMuonHMayArchGNMuonHConfig(
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


def _format_trial_label(mode: str) -> str:
    return {"pre_norm": "prenorm", "post_norm": "postnorm"}[mode]


def _build_step(mode: str) -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    model = dataclasses.replace(
        model,
        num_experts=_NUM_EXPERTS,
        partial_key_offset="every_4th",
        use_partial_rope=True,
        last_layer_pko=True,
        router_z_loss_coef=0.0,
        pko_q_split_rescale_mode=mode,
    )
    optimizer = _muonh_optimizer(base_optimizer)

    label = _format_trial_label(mode)
    run_id = f"muonh-may-arch-1pct-pko-q-split-rescale-{label}-{_RUN_SUFFIX}-d{_HIDDEN_DIM}"
    step_name = f"grug/muonh_may_arch_1pct_pko_q_split_rescale_sweep/{run_id}"

    return ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=[
                    "moe",
                    "muonh_may_arch_1pct_pko_q_split_rescale_sweep",
                    f"d{_HIDDEN_DIM}",
                    label,
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
        ),
    )


if __name__ == "__main__":
    steps = [_build_step(mode) for mode in ("pre_norm", "post_norm")]
    executor_main(
        steps=steps,
        description=(
            "MoE may_arch + 1pct-noclip + PKO q-half learned rescale (pre/post qk-norm), "
            f"d{_HIDDEN_DIM}, run_suffix={_RUN_SUFFIX!r}."
        ),
    )
