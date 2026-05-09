# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon walk p12: p11 architecture, batch halved (128 -> 64) and steps doubled
(3350 -> 6700).

Total tokens are unchanged at 1.756B (= 64 * 4096 * 6700). The point is to
ablate batch size at fixed compute before introducing MoE in p13. For the muon
walk specifically the LR / WD are not heuristic-derived, so they're held at
the same constants as p11 (lr=0.035, wd=0.025); the LR vs batch interaction
will show up cleanly in the loss curves.
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
from experiments.grug.nano.launch_muon_tuned_walk_p11 import P11_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P12_TRAIN_STEPS = 6700
P12_BATCH_SIZE = 64
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025

# Architecture is identical to p11. The schedule (batch / num_steps) is what
# changes; downstream walks (p13) inherit this model unchanged and add MoE.
P12_MODEL = dataclasses.replace(P11_MODEL)

P12_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-p12")


nano_muon_tuned_p12_trial = ExecutorStep(
    name="grug/nano-muon-tuned-p12-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
        model=versioned(P12_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P12_TRAIN_STEPS),
        batch_size=versioned(P12_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "muon", "fineweb-gpt2", "tuned", "p12", "halfbs"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P12_OPTIMIZER),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1)),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=P12_BATCH_SIZE,
                # Doubling steps means doubling the eval interval so we keep
                # the same number of eval points as p11 (~27 evals over the run).
                steps_per_eval=250,
                # 40 batches x 64 (BS) x 4096 (seq) = 10.49M tokens per eval pass.
                max_eval_batches=40,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_muon_tuned_p12_trial],
        description="muon-tuned p12: batch=64, steps=6700 (halved BS, doubled steps).",
    )
