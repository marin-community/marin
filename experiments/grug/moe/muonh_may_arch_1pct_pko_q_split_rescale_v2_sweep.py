# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + 1pct-noclip + PKO q-split-rescale V2: two independent sigmoid gates.

V2 variant of the q-half rescale idea. Two SEPARATE per-q-head weight
matrices (each shape ``(hidden_dim, num_heads)``, zero-init); each
produces a per-token, per-q-head ``sigmoid`` in [0, 1] (NOT 2*sigmoid).

    stat_w = sigmoid(x @ W_stat)      # (B, S, num_heads), values in [0, 1]
    rot_w  = sigmoid(x @ W_rot)       # (B, S, num_heads), values in [0, 1]
    q_stat = q[..., half:] * stat_w
    q_rot  = q[..., :half] * rot_w

At init both halves are scaled by 0.5 (q magnitude halved). Sum is
NOT constrained -- the two halves can independently grow/shrink.

Weight matrices routed to a dedicated optimizer group:
- Adam (NOT AdamH), b1=0.95, b2=0.999
- LR = 0.1 x normal adam_lr

Two variants, both at d512: pre_norm and post_norm.

Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_pko_q_split_rescale_v2_sweep
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
_GROUP_NAME: str = "muonh-may-arch-1pct-pko-q-split-rescale-v2-sweep"
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
        # V2 q-split-rescale optimizer hyperparameters.
        q_split_v2_beta1=0.95,
        q_split_v2_beta2=0.999,
        q_split_v2_lr_scale=0.1,
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
        pko_q_split_rescale_v2_mode=mode,
    )
    optimizer = _muonh_optimizer(base_optimizer)

    label = _format_trial_label(mode)
    run_id = f"muonh-may-arch-1pct-pko-q-split-rescale-v2-{label}-{_RUN_SUFFIX}-d{_HIDDEN_DIM}"
    step_name = f"grug/muonh_may_arch_1pct_pko_q_split_rescale_v2_sweep/{run_id}"

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
                    "muonh_may_arch_1pct_pko_q_split_rescale_v2_sweep",
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
            "MoE may_arch + 1pct-noclip + PKO q-half learned rescale V2 (two independent sigmoids per q-head, "
            f"pre/post qk-norm), d{_HIDDEN_DIM}, run_suffix={_RUN_SUFFIX!r}."
        ),
    )
