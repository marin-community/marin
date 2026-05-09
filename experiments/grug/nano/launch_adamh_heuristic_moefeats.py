# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Heuristic-AdamH baseline + moe-style architectural features.

Walks one step away from `launch_adamh_heuristic.py:nano_adamh_heuristic_trial`
by turning on the architectural conventions used in `experiments/grug/moe`:

- `logit_cap = 0.0`        — drop the logit soft-cap entirely.
- `qk_mult = 1.3`          — multiply q after QK norm + RoPE (moe convention).
- `use_attn_gate = True`   — per-head zero-init sigmoid gate on attention output.
- `use_gated_norm = True`  — rank-128 learnable gate after every parametric RMSNorm.
- `use_bias = False`       — drop biases on every Linear in the model.
- `initializer_std = 0.5 / sqrt(hidden_dim)` — moe init scale (≈0.0180 for dim=768).
- `zero_init_proj = False` — required since AdamH preserves Frobenius norm.
- `init_scheme = "default"` — truncated_normal everywhere with the new std.

Optimizer: heuristic AdamH (no clip, β/ε/lrs computed at this scale). The mask
in `NanoHeuristicAdamHConfig._create_mask` routes `attn_gate -> AdamW` (matching
moe), `gated_norm -> AdamH` (per the user spec — differs from moe), and
everything else 2D+ except `embed` -> AdamH.

Run length: 3350 steps. Optimizer is rebuilt at this step count so the
heuristic-derived hyperparameters reflect the new total token count.
"""

import math

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch_adamh_heuristic import (
    NanoAdamHHeuristicLaunchConfig,
    _fineweb_gpt2_data,
    _resolve_run_id,
    build_heuristic_optimizer,
    run_nano_adamh_heuristic_trial,
)
from experiments.grug.nano.model import NanoModelConfig
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

NANO_ADAMH_HEURISTIC_MOEFEATS_TRAIN_STEPS = 3350

_HIDDEN_DIM = 768
_INITIALIZER_STD = 0.5 / math.sqrt(_HIDDEN_DIM)


_SEQ_LEN = 4096
_BATCH_SIZE = 128


NANO_124M_ADAMH_HEURISTIC_MOEFEATS_MODEL = NanoModelConfig(
    vocab_size=50304,
    hidden_dim=_HIDDEN_DIM,
    intermediate_dim=3072,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    max_seq_len=_SEQ_LEN,
    # moe-style architectural features:
    logit_cap=0.0,
    qk_mult=1.3,
    use_attn_gate=True,
    use_gated_norm=True,
    gated_norm_rank=128,
    use_bias=False,
    # 3-short / 1-long sliding-window pattern (matches grug/moe).
    sliding_window=_SEQ_LEN,
    # moe init:
    init_scheme="default",
    zero_init_proj=False,  # AdamH needs non-zero matrices to update.
    initializer_std=_INITIALIZER_STD,
)


# Optimizer rebuilt at the new (batch_size, num_train_steps) so heuristic LR/beta2/eps
# reflect the new total tokens and tokens-per-batch. Note: tpb is unchanged
# (was 512*1024=524288, now 128*4096=524288), so beta2 and eps_scale stay put,
# only sqrt(tpb)*lr scaling is different (it isn't, since sqrt(tpb) is the same).
# Total tokens = 524288 * 3350 = 1.756e9.
NANO_124M_ADAMH_HEURISTIC_MOEFEATS_OPTIMIZER = build_heuristic_optimizer(
    batch_size=_BATCH_SIZE,
    num_train_steps=NANO_ADAMH_HEURISTIC_MOEFEATS_TRAIN_STEPS,
    seq_len=NANO_124M_ADAMH_HEURISTIC_MOEFEATS_MODEL.max_seq_len,
    hidden_dim=NANO_124M_ADAMH_HEURISTIC_MOEFEATS_MODEL.hidden_dim,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-moefeats-sw")


nano_adamh_heuristic_moefeats_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-moefeats-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(NANO_124M_ADAMH_HEURISTIC_MOEFEATS_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(NANO_ADAMH_HEURISTIC_MOEFEATS_TRAIN_STEPS),
        batch_size=versioned(_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "moefeats"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NANO_124M_ADAMH_HEURISTIC_MOEFEATS_OPTIMIZER),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,  # keep heuristic z-loss
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=_BATCH_SIZE,
                steps_per_eval=125,
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_adamh_heuristic_moefeats_trial],
        description="Heuristic AdamH + moe-style features, 3350 steps on fineweb10B-gpt2.",
    )
