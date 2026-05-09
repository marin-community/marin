# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bare-architecture heuristic-AdamH baseline at 3350 steps.

Same model and optimizer as `launch_adamh_heuristic.py:nano_adamh_heuristic_trial`,
just with `steps=3350` (and the heuristic optimizer rebuilt at the new step
count). Useful as a clean baseline for ablations against
`launch_adamh_heuristic_moefeats.py` (which adds attn_gate, gated_norm,
bias-free, qk_mult=1.3, no softcap, and sliding window) at matched step count.

Architecture and routing are unchanged from the bare heuristic:
- modded-nanogpt model with logit_cap=15 (rsqrt form), biases on every Linear,
  qk_mult=1.0, no attn_gate, no gated_norm, no sliding window.
- Mask: `lm_head -> AdamH`, `embed -> AdamW`, `1D -> AdamW`, block 2D -> AdamH.

At 3350 steps, batch 512, seq 1024:
    tokens_per_batch = 524,288
    total_tokens     = 1.756e9
    compute (no lm_head) ≈ 1.70e18 FLOPs (matches moe README's d768 anchor).
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch_adamh_heuristic import (
    NANO_124M_HEURISTIC_MODEL,
    NanoAdamHHeuristicLaunchConfig,
    _fineweb_gpt2_data,
    _resolve_run_id,
    build_heuristic_optimizer,
    run_nano_adamh_heuristic_trial,
)
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

NANO_ADAMH_HEURISTIC_3350_TRAIN_STEPS = 3350


# Optimizer rebuilt at 3350 steps so heuristic-derived hyperparameters reflect
# the new total token count.
NANO_124M_HEURISTIC_3350_OPTIMIZER = build_heuristic_optimizer(
    batch_size=512,
    num_train_steps=NANO_ADAMH_HEURISTIC_3350_TRAIN_STEPS,
    seq_len=NANO_124M_HEURISTIC_MODEL.max_seq_len,
    hidden_dim=NANO_124M_HEURISTIC_MODEL.hidden_dim,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-3350")


nano_adamh_heuristic_3350_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-3350-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(NANO_124M_HEURISTIC_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(NANO_ADAMH_HEURISTIC_3350_TRAIN_STEPS),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "3350"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NANO_124M_HEURISTIC_3350_OPTIMIZER),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512,
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
        steps=[nano_adamh_heuristic_3350_trial],
        description="Bare heuristic AdamH baseline, 3350 steps on fineweb10B-gpt2.",
    )
