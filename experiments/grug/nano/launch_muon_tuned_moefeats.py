# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tuned-Muon baseline + moe-style architectural features.

Walks one step away from `launch_muon_tuned.py:nano_muon_tuned_trial` by
turning on the architectural conventions used in `experiments/grug/moe`:

- `logit_cap = 0.0`        — drop the logit soft-cap entirely.
- `qk_mult = 1.3`          — multiply q after QK norm + RoPE (moe convention).
- `use_attn_gate = True`   — per-head zero-init sigmoid gate on attention output.
- `use_gated_norm = True`  — rank-128 learnable gate after every parametric RMSNorm.
- `use_bias = False`       — drop biases on every Linear in the model.
- `initializer_std = 0.5 / sqrt(hidden_dim)` — moe init scale (≈0.0180 for dim=768).
- `zero_init_proj = False` — moe doesn't zero-init "proj" weights.
- `init_scheme = "default"` — truncated_normal everywhere with the new std.

Optimizer: identical to `nano_muon_tuned_trial` (AdamW + Muon, lr=0.035, wd=0.025).
The mask in `NanoAdamWMuonConfig._create_mask` already routes `attn_gate` and
`gated_norm` to the AdamW (`adam_norm`) group, matching the user spec.

Run length: 3350 steps (matches the muon-tuned baseline).
"""

import math

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch import (
    NanoAdamWMuonConfig,
    NanoLaunchConfig,
    _fineweb_gpt2_data,
    _resolve_run_id,
    run_nano_trial,
)
from experiments.grug.nano.model import NanoModelConfig
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

NANO_MUON_TUNED_MOEFEATS_TRAIN_STEPS = 3350
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025

_HIDDEN_DIM = 768
_INITIALIZER_STD = 0.5 / math.sqrt(_HIDDEN_DIM)


_SEQ_LEN = 4096
_BATCH_SIZE = 128

NANO_124M_MUON_TUNED_MOEFEATS_MODEL = NanoModelConfig(
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
    # 3-short / 1-long sliding-window pattern (window = seq_len // 2 on the short blocks,
    # full causal on every 4th block). Matches `experiments/grug/moe`'s convention.
    sliding_window=_SEQ_LEN,
    # moe init:
    init_scheme="default",
    zero_init_proj=False,
    initializer_std=_INITIALIZER_STD,
)


NANO_MUON_TUNED_MOEFEATS_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-moefeats-sw")


nano_muon_tuned_moefeats_trial = ExecutorStep(
    name="grug/nano-muon-tuned-moefeats-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
        model=versioned(NANO_124M_MUON_TUNED_MOEFEATS_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(NANO_MUON_TUNED_MOEFEATS_TRAIN_STEPS),
        batch_size=versioned(_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "muon", "fineweb-gpt2", "tuned", "moefeats"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NANO_MUON_TUNED_MOEFEATS_OPTIMIZER),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,
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
        steps=[nano_muon_tuned_moefeats_trial],
        description="Tuned Muon + moe-style features, 3350 steps on fineweb10B-gpt2.",
    )
