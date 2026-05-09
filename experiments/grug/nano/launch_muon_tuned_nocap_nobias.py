# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tuned-Muon ablation: drop the logit soft-cap and all biases.

Step 2 in the muon walk toward moefeats. Identical to
`launch_muon_tuned_nocap.py:nano_muon_tuned_nocap_trial` plus `use_bias=False`.
No attn_gate, no gated_norm, no qk_mult, no sliding window.

3350 steps, batch 512, seq 1024 — same compute as `nano_muon_tuned_trial`.
"""

import dataclasses

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
from experiments.grug.nano.launch_muon_tuned_nocap import NANO_124M_MUON_TUNED_NOCAP_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

NANO_MUON_TUNED_NOCAP_NOBIAS_TRAIN_STEPS = 3350
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025

NANO_124M_MUON_TUNED_NOCAP_NOBIAS_MODEL = dataclasses.replace(
    NANO_124M_MUON_TUNED_NOCAP_MODEL,
    use_bias=False,
)

NANO_MUON_TUNED_NOCAP_NOBIAS_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-nocap-nobias")


nano_muon_tuned_nocap_nobias_trial = ExecutorStep(
    name="grug/nano-muon-tuned-nocap-nobias-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
        model=versioned(NANO_124M_MUON_TUNED_NOCAP_NOBIAS_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(NANO_MUON_TUNED_NOCAP_NOBIAS_TRAIN_STEPS),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "muon", "fineweb-gpt2", "tuned", "nocap", "nobias"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NANO_MUON_TUNED_NOCAP_NOBIAS_OPTIMIZER),
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
        steps=[nano_muon_tuned_nocap_nobias_trial],
        description="Muon-tuned minus rsqrt cap and biases, 3350 steps.",
    )
