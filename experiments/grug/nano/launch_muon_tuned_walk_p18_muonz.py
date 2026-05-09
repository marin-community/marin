# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon walk p18 with final-logit z-loss enabled (z_loss_weight = 1e-4).

Identical to ``launch_muon_tuned_walk_p18.py`` except for the
``z_loss_weight`` knob. p18 is p17 with the shared expert reshaped to
ReLU2 @ 1.5*D; this run adds the same final-logit z-loss the adamh
side uses.
"""

from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch import (
    NanoAdamWMuonConfig,
    NanoLaunchConfig,
    _resolve_run_id,
    run_nano_trial,
)
from experiments.grug.nano.launch_muon_tuned_walk_p16 import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
from experiments.grug.nano.launch_muon_tuned_walk_p18 import P18_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P18_MUONZ_TRAIN_STEPS = 10343
P18_MUONZ_BATCH_SIZE = 64
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025

P18_MUONZ_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-p18-muonz")


nano_muon_tuned_p18_muonz_trial = ExecutorStep(
    name="grug/nano-muon-tuned-p18-muonz-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
        model=versioned(P18_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P18_MUONZ_TRAIN_STEPS),
        batch_size=versioned(P18_MUONZ_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "muon", "nemotron", "tuned", "p18", "moe", "shared-relu2", "muonz"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P18_MUONZ_OPTIMIZER),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
                train_batch_pspec=P(("data", "expert")),
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=P18_MUONZ_BATCH_SIZE,
                steps_per_eval=250,
                max_eval_batches=40,
                eval_current=True,
                eval_ema=False,
                eval_batch_pspec=P(("data", "expert")),
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_muon_tuned_p18_muonz_trial],
        description="muon-tuned p18 (relu2 shared expert) + final-logit z_loss=1e-4.",
    )
